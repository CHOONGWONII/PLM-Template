# -*- coding: utf-8 -*-

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
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

def _make_causal_mask(input_ids, past_len = 0):
    """
        Inputs
            dec_ids.shape = (B, D_L) or (B, 1)
    """
    batch_size, tgt_len = input_ids.size()
    device = input_ids.device

    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(torch.float).to(device)

    if past_len > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_len, dtype=torch.float, device = device), mask], dim=-1)

    return mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len + past_len)

class Conv1D(nn.Module):
    def __init__(self, output_dim, input_dim):
        super().__init__()
        self.output_dim = output_dim
        w = torch.empty(input_dim, output_dim)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.output_dim,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

class GPT2Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.num_heads = cfg.num_heads
        self.d_head = self.d_model // self.num_heads
        self.scaling = self.d_head ** -0.5
        self.attn_dropout = cfg.attn_dropout
        self.residual_dropout = cfg.residual_dropout

        self.c_attn = Conv1D(3 * self.d_model, self.d_model)
        self.c_proj = Conv1D(self.d_model, self.d_model)

    def _split_shape(self, tensor):
        '''
            Inputs  
                tensor.shape = (B, L, H)
            Outputs
                output.shape = (B, num_heads, L, H // num_heads)
        '''
        batch_size, seq_len = tensor.shape[0], tensor.shape[1]
        return tensor.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2).contiguous()

    def _merge_shape(self, tensor):
        '''
            Inputs
                tensor.shape = (B * num_heads, L, H // num_heads)
            Outputs
                output.shape = (B, L, H)
        '''
        seq_len = tensor.shape[1]
        output = tensor.view(-1, self.num_heads, seq_len, self.d_head)
        output = output.transpose(1, 2)
        output = output.reshape(-1, seq_len, self.d_model)
        return output

    def forward(self, hidden_states, past_key_value = None, attention_mask = None):
        '''
            Inputs
                hidden_states.shape = (B, L, H)
                past_key_value.shape = ((B, num_heads, L, H // num_heads), (B, num_heads, L, H // num_heads))
                attention_mask.shape = (B, 1, T_L, S_L)
            Outputs
                attn_output.shape = (B, L, H)
                present_key_value
        '''
        batch_size = hidden_states.shape[0]

        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.d_model, dim = -1)
        query_states, key_states, value_states = self._split_shape(query_states), self._split_shape(key_states), self._split_shape(value_states)
        if past_key_value is not None:
            key_states = torch.cat((past_key_value[0], key_states), dim = 2)
            value_states = torch.cat((past_key_value[1], value_states), dim = 2)

        present_key_value = (key_states, value_states)

        proj_shape = (batch_size * self.num_heads, -1, self.d_head)
        query_states, key_states, value_states = query_states.view(*proj_shape), key_states.view(*proj_shape), value_states.view(*proj_shape)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        attn_weights = attn_weights * torch.tensor(self.scaling, dtype = attn_weights.dtype, device = attn_weights.device)

        if attention_mask is not None:
            tgt_len, src_len = attn_weights.shape[-2], attn_weights.shape[-1]
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim = -1)
        attn_weights = F.dropout(attn_weights, p = self.attn_dropout, training = self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = self._merge_shape(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = F.dropout(attn_output, p = self.residual_dropout, training = self.training)

        return attn_output, present_key_value

class GPT2MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        self.residual_dropout = cfg.residual_dropout

        self.c_fc = Conv1D(4 * d_model, d_model)
        self.c_proj = Conv1D(d_model, 4 * d_model)
        self.act = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = F.dropout(hidden_states, p = self.residual_dropout, training = self.training)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model

        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = GPT2Attention(cfg)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = GPT2MLP(cfg)

    def forward(
        self,
        hidden_states,
        past_key_value = None,
        attention_mask = None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, present_key_value = self.attn(
            hidden_states,
            past_key_value = past_key_value,
            attention_mask=attention_mask,
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states, present_key_value 

class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model

        self.wte = nn.Embedding(cfg.plm_vocab_size, self.d_model)
        self.wpe = nn.Embedding(cfg.max_position_embeddings, self.d_model)

        self.embed_dropout = cfg.embed_dropout
        self.h = nn.ModuleList([GPT2Block(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(self.d_model)

    def forward(self, input_ids, attention_mask = None, past_key_values = None):
        '''
            Inputs
                input_ids.shape = (B, L) or (B, 1)
                attention_mask.shape = (B, L)
                past_key_values
            Outputs

        '''
        batch_size, tgt_len = input_ids.shape[0], input_ids.shape[1]
        device = input_ids.device

        cache = ()

        if past_key_values is None:
            past_len = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_len = past_key_values[0][0].shape[2]
        position_ids = torch.arange(past_len, tgt_len + past_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, tgt_len)

        if attention_mask is not None:
            temp1 = _make_causal_mask(input_ids, past_len)
            temp2 = _expand_mask(attention_mask, tgt_len = tgt_len)
            attention_mask = temp1 + temp2
            # attention_mask.shape = (B, 1, T_L, S_L)
        else:
            attention_mask = _make_causal_mask(input_ids)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = F.dropout(hidden_states, p = self.embed_dropout, training = self.training)

        for i, (block, past_key_value) in enumerate(zip(self.h, past_key_values)):
            hidden_states, present_key_value = block(
                hidden_states,
                past_key_value = past_key_value,
                attention_mask = attention_mask,
            )

            cache += (present_key_value,)

        hidden_states = self.ln_f(hidden_states)

        past_key_values = cache

        return {
            'hidden_states' : hidden_states,
            'past_key_values' : past_key_values
        }

class GPT2LMHeadModel(nn.Module):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.transformer = GPT2Model(cfg)
        self.lm_head = nn.Linear(cfg.d_model, len(tokenizer), bias=False)

    def forward(self, input_ids, past_key_values = None, attention_mask = None, label_ids = None,):
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values = past_key_values,
            attention_mask = attention_mask,
        )
        hidden_states = transformer_outputs['hidden_states']
        past_key_values = transformer_outputs['past_key_values']

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if label_ids is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = label_ids[:, 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {
            'hidden_states' : hidden_states,
            'past_key_values' : past_key_values,
            'lm_logits' : lm_logits,
            'loss' : loss
        }

    def load_plm(self):
        state_dict = torch.load(self.cfg.plm_name)
        self.transformer.load_state_dict(state_dict, strict = False)

    def resize_token_embeddings(self):
        old_embeddings = self.transformer.wte
        
        new_embeddings = nn.Embedding(len(self.tokenizer), old_embeddings.shape[1])
        new_embeddings.to(old_embeddings.weight.device, dtype = old_embeddings.weight.dtype)

        n = min(old_embeddings.shape[0], new_embeddings.shape[0])

        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        self.transformer.wte = new_embeddings

    def generate(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        has_eos = torch.zeros(batch_size, dtype = torch.bool).to(self.device)
        outputs = []

        bos_ids = torch.tensor([[bos_token_id]] * batch_size, dtype = torch.long, device = self.device)
        bos_mask = torch.ones_like(bos_ids)
        input_ids = torch.cat((input_ids, bos_ids), dim = -1)
        attention_mask = torch.cat((attention_mask, bos_mask), dim = -1)
        past_key_values = None

        for _ in range(self.cfg.generate_max_length):
            gpt_output = self.forward(
                input_ids = input_ids,
                past_key_values = past_key_values,
                attention_mask = attention_mask
            )

            new_token_ids = torch.argmax(gpt_output['lm_logits'][:, -1, :], dim = -1)
            
            has_eos = has_eos | (new_token_ids == eos_token_id)
            new_token_ids = new_token_ids.masked_fill(has_eos, eos_token_id)
            outputs.append(new_token_ids)

            input_ids = new_token_ids.unsqueeze(-1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(input_ids)), dim = -1)
            past_key_values = gpt_output['past_key_values']

            if torch.all(has_eos):
                break

        outputs = torch.stack(outputs, dim = -1).tolist()
        generated_outputs = []

        for output in outputs:
            generated_outputs.append(self.tokenizer.decode(output, skip_special_tokens = True))

        return {
            'generated_outputs' : generated_outputs
        }