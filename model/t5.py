# -*- coding: utf-8 -*-

import math
from turtle import position
import torch
from torch import nn
import torch.nn.functional as F

def _expand_mask(mask, tgt_len = None):
    """
        Inputs
            mask.shape = (B, S_L)
        Outputs
            output.shape = (B, 1, T_L, S_L)
    """
    batch_size, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).to(torch.float)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(torch.float).min)

def _make_causal_mask(dec_ids, past_key_values_length: int = 0):
    """
        Inputs
            dec_ids.shape = (B, D_L) or (B, 1)
    """
    batch_size, tgt_len = dec_ids.size()
    device = dec_ids.device

    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(torch.float).to(device)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=torch.float, device = device), mask], dim=-1)
    return mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len + past_key_values_length)


class T5LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.d_model))
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

class T5DenseReluDense(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wi = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.wo = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.act = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class T5LayerFF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.DenseReluDense = T5DenseReluDense(cfg)
        self.layer_norm = T5LayerNorm(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

class T5Attention(nn.Module):
    def __init__(self, cfg, is_decoder, is_cross_attention, has_relative_attention_bias = False):
        super().__init__()
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention
        self.has_relative_attention_bias = has_relative_attention_bias

        self.d_model = cfg.d_model
        self.num_heads = cfg.num_heads
        self.d_head = self.d_model // self.num_heads
        self.dropout = nn.Dropout(cfg.dropout)

        self.q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o = nn.Linear(self.d_model, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(cfg.relative_attention_num_buckets, self.num_heads)
            self.relative_attention_num_buckets = cfg.relative_attention_num_buckets
            self.relative_attention_max_distance = cfg.relative_attention_max_distance

    def _relative_position_bucket(self, relative_position):
        num_buckets = self.relative_attention_num_buckets
        max_distance = self.relative_attention_max_distance
        relative_buckets = 0

        if not self.is_decoder:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position) 
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)

        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)

        return relative_buckets

    def compute_bias(self, tgt_len, src_len):
        device = self.relative_attention_bias.weight.device

        context_position = torch.arange(tgt_len, dtype = torch.long, device = device)[:, None]
        memory_position = torch.arange(src_len, dtype = torch.long, device = device)[None, :]
        relative_position = memory_position - context_position

        relative_position_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)

        return values

    def _shape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2).contiguous()

    def forward(self, query_states, key_value_states, past_key_value = None, attention_mask = None, position_bias = None, real_tgt_len = None):
        batch_size, tgt_len, d_model = query_states.size()
        _, src_len, _ = key_value_states.size()

        if real_tgt_len is not None: # cross_attention
            real_tgt_len = real_tgt_len
        elif past_key_value is None:
            real_tgt_len = tgt_len
        elif past_key_value is not None:
            real_tgt_len = tgt_len + past_key_value[0].shape[2]

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(real_tgt_len, attn_scores.shape[-1])
            else:
                position_bias = torch.zeros(
                    (1, self.num_heads, real_tgt_len, attn_scores.shape[-1]), device = attn_scores.device, dtype=attn_scores.dtype
                )
                if self.training:
                    position_bias.requires_grad = True

            if past_key_value is not None:
                position_bias = position_bias[:, :, -tgt_len : , :]

            if attention_mask is not None:
                position_bias = position_bias + attention_mask  # (batch_size, n_heads, tgt_len, src_len)

        query_states = self._shape(self.q(query_states), tgt_len, batch_size)
        if self.is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif self.is_cross_attention:
            key_states = self._shape(self.k(key_value_states), -1, batch_size)
            value_states = self._shape(self.v(key_value_states), -1, batch_size)
        elif past_key_value is not None:
            key_states = self._shape(self.k(key_value_states), -1, batch_size)
            value_states = self._shape(self.v(key_value_states), -1, batch_size)
            key_states = torch.cat([past_key_value[0], key_states], dim = 2)
            value_states = torch.cat([past_key_value[1], value_states], dim = 2)
        else:
            key_states = self._shape(self.k(key_value_states), -1, batch_size)
            value_states = self._shape(self.v(key_value_states), -1, batch_size)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        attn_scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )

        attn_scores += position_bias

        attn_weights = F.softmax(attn_scores, dim = -1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_output = self.o(attn_output)

        return attn_output, past_key_value, position_bias

class T5LayerSelfAttention(nn.Module):
    def __init__(self, cfg, is_decoder, has_relative_attention_bias = False):
        super().__init__()
        self.SelfAttention = T5Attention(
            cfg, is_decoder = is_decoder, is_cross_attention = False, has_relative_attention_bias = has_relative_attention_bias
        )
        self.layer_norm = T5LayerNorm(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, hidden_states, attention_mask = None, position_bias = None, past_key_value = None):
        normed_hidden_states = self.layer_norm(hidden_states)
        attn_output, past_key_value, position_bias = self.SelfAttention(
            query_states = normed_hidden_states, 
            key_value_states = normed_hidden_states, 
            past_key_value = past_key_value, 
            attention_mask = attention_mask, 
            position_bias = position_bias            
        )
        hidden_states = hidden_states + self.dropout(attn_output)

        return hidden_states, past_key_value, position_bias

class T5LayerCrossAttention(nn.Module):
    def __init__(self, cfg, has_relative_attention_bias = False):
        super().__init__()
        self.EncDecAttention = T5Attention(
            cfg, is_decoder  = True, is_cross_attention = True, has_relative_attention_bias = has_relative_attention_bias
        )
        self.layer_norm = T5LayerNorm(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, hidden_states, enc_hidden_states, attention_mask = None, position_bias = None, real_tgt_len = None, past_key_value = None):
        normed_hidden_states = self.layer_norm(hidden_states)
        attn_output, past_key_value, position_bias = self.EncDecAttention(
            query_states = normed_hidden_states,
            key_value_states = enc_hidden_states,
            position_bias = position_bias,
            past_key_value = past_key_value,
            attention_mask = attention_mask,
            real_tgt_len = real_tgt_len
        )
        hidden_states = hidden_states + self.dropout(attn_output)

        return hidden_states, past_key_value, position_bias

class T5Block(nn.Module):
    def __init__(self, cfg, is_decoder, has_relative_attention_bias = False):
        super().__init__()
        self.is_decoder = is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(cfg, is_decoder = is_decoder, has_relative_attention_bias = has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(cfg))
        self.layer.append(T5LayerFF(cfg))

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        position_bias = None,
        enc_hidden_states = None,
        enc_dec_mask = None,
        enc_dec_position_bias = None,
        past_key_value = None
    ):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        cross_attn_past_key_value = past_key_value[2:] if past_key_value is not None else None

        self_attn_position_bias, cross_attn_position_bias = position_bias, enc_dec_position_bias

        hidden_states, self_attn_present_key_value, self_attn_position_bias = self.layer[0](
            hidden_states,
            attention_mask = attention_mask,
            position_bias = position_bias,
            past_key_value = self_attn_past_key_value
        )

        if self.is_decoder:
            real_tgt_len = self_attn_present_key_value[0].shape[2]

            hidden_states, cross_attn_present_key_value, cross_attn_position_bias = self.layer[1](
                hidden_states,
                enc_hidden_states = enc_hidden_states,
                attention_mask = enc_dec_mask,
                position_bias = enc_dec_position_bias,
                past_key_value = cross_attn_past_key_value,
                real_tgt_len = real_tgt_len
            )
        
        hidden_states = self.layer[-1](hidden_states)
        present_key_value = self_attn_present_key_value + cross_attn_present_key_value if self.is_decoder else None

        return (hidden_states, present_key_value, self_attn_position_bias, cross_attn_position_bias)

class T5Stack(nn.Module):
    def __init__(self, cfg, is_decoder, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.is_decoder = is_decoder

        self.block = nn.ModuleList(
            [T5Block(cfg, is_decoder, has_relative_attention_bias = bool(i==0)) for i in range(cfg.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        enc_ids = None,
        enc_mask = None,
        dec_ids = None,
        dec_mask = None,
        enc_hidden_states = None,
        past_key_values = None
    ):
        if not self.is_decoder:
            token_embedding = self.embed_tokens(enc_ids)
            hidden_states = self.dropout(token_embedding)
            enc_self_mask = _expand_mask(enc_mask)
            position_bias = None

            for layer in self.block:
                layer_outputs = layer(
                    hidden_states,
                    enc_self_mask,
                    position_bias
                )

                hidden_states = layer_outputs[0]
                position_bias =layer_outputs[2]

            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)

            return {
                'enc_hidden_states' : hidden_states
            }

        else:
            token_embedding = self.embed_tokens(dec_ids)
            hidden_states = self.dropout(token_embedding)
            enc_dec_mask = _expand_mask(enc_mask, dec_ids.shape[-1])
            dec_self_mask = None
            if dec_ids.shape[-1] != 1:
                temp1 = _make_causal_mask(dec_ids)
                temp2 = _expand_mask(dec_mask)
                dec_self_mask = temp1 + temp2
            position_bias = None
            enc_dec_position_bias = None

            cache = ()

            for idx, layer in enumerate(self.block):
                past_key_value = past_key_values[idx] if past_key_values is not None else None

                layer_outputs = layer(
                    hidden_states,
                    dec_self_mask,
                    position_bias,
                    enc_hidden_states,
                    enc_dec_mask,
                    enc_dec_position_bias,
                    past_key_value
                )

                hidden_states = layer_outputs[0]
                cache += (layer_outputs[1],)
                position_bias = layer_outputs[2]
                enc_dec_position_bias = layer_outputs[3]

            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)

            past_key_values = cache

            return {
                'dec_hidden_states' : hidden_states,
                'past_key_values' : past_key_values
            }

class T5Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.shared = nn.Embedding(cfg.plm_vocab_size, cfg.d_model)

        self.encoder = T5Stack(cfg, is_decoder = False, embed_tokens = self.shared)
        self.decoder = T5Stack(cfg, is_decoder = True, embed_tokens = self.shared)

    def forward(
        self, enc_ids, enc_mask, dec_ids, dec_mask = None, enc_hidden_states = None, past_key_values = None
    ):
        if enc_hidden_states is None:
            enc_outputs = self.encoder(
                enc_ids = enc_ids,
                enc_mask = enc_mask
            )
            enc_hidden_states = enc_outputs['enc_hidden_states']
        
        dec_outputs = self.decoder(
            enc_ids = None,
            enc_mask = enc_mask,
            dec_ids = dec_ids,
            dec_mask = dec_mask,
            enc_hidden_states = enc_hidden_states,
            past_key_values = past_key_values
        )

        return {
            'enc_hidden_states' : enc_hidden_states,
            'dec_hidden_states' : dec_outputs['dec_hidden_states'],
            'past_key_values' : dec_outputs['past_key_values']
        }

class T5ForConditionalGeneration(nn.Module):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.transformer = T5Model(cfg)
        self.lm_head = nn.Linear(cfg.d_model, len(tokenizer), bias = False)
        self.lm_head.weight.data.normal_(mean = 0.0, std = 0.02)

        self.load_plm(cfg.cache_filename)

    def load_plm(self, cache_filename = None):
        state_dict = torch.load(cache_filename)
        self.transformer.load_state_dict(state_dict, strict = False)

    def forward(
        self, enc_ids, enc_mask, dec_ids, dec_mask = None, enc_hidden_states = None, past_key_values = None, label_ids = None
    ):
        transformer_outputs = self.transformer(
            enc_ids = enc_ids,
            enc_mask = enc_mask,
            dec_ids = dec_ids,
            dec_mask = dec_mask,
            enc_hidden_states = enc_hidden_states,
            past_key_values = past_key_values
        )
        lm_logits = self.lm_head(transformer_outputs['dec_hidden_states'])
        lm_loss = None
        if label_ids is not None:
            criterion = nn.CrossEntropyLoss()
            lm_loss = criterion(lm_logits.view(-1, self.cfg.plm_vocab_size), label_ids.view(-1))

        return {
            'enc_hidden_states' : transformer_outputs['enc_hidden_states'],
            'dec_hidden_states' : transformer_outputs['dec_hidden_states'],
            'past_key_values' : transformer_outputs['past_key_values'],
            'lm_logits' : lm_logits,
            'lm_loss' : lm_loss
        }

    def generate(self, enc_ids, enc_mask):
        batch_size = enc_ids.shape[0]
        device = enc_ids.device
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id=  self.tokenizer.eos_token_id

        outputs = []
        has_eos = torch.zeros(batch_size, dtype = torch.bool).to(device)

        dec_ids = torch.tensor([[bos_token_id]] * batch_size, dtype = torch.long, device = device)
        enc_hidden_states = None
        past_key_values = None

        for _ in range(self.cfg.generate_max_length):
            transformer_outputs = self.forward(
                enc_ids, enc_mask, dec_ids, enc_hidden_states = enc_hidden_states, past_key_values = past_key_values
            )

            new_token_ids = torch.argmax(transformer_outputs['lm_logits'][:, -1, :], dim = -1)

            has_eos = has_eos | (new_token_ids == eos_token_id)
            new_token_ids = new_token_ids.masked_fill(has_eos, eos_token_id)
            outputs.append(new_token_ids)

            dec_ids = new_token_ids.unsqueeze(-1)
            enc_hidden_states = transformer_outputs['enc_hidden_states']
            past_key_values = transformer_outputs['past_key_values']

            if torch.all(has_eos):
                break

        outputs = torch.stack(outputs, dim = -1).tolist()
        generated_outputs = []

        for output in outputs:
            generated_outputs.append(self.tokenizer.decode(output, skip_special_tokens = True))
        
        return {
            'generated_outputs' : generated_outputs
        }

