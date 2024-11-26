import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from model.pref import Preference_Model, BahdanauAttention_softmax
from model.austgn import Sequential_Model
from utils.utils import MLP_BN4d, MLP_LN

class BahdanauAttention_softmax(nn.Module):
    def __init__(self, key_size, query_size, hidden_size):
        super().__init__()
        self.wk = nn.Linear(key_size, hidden_size)
        self.wq = nn.Linear(query_size, hidden_size)
        self.wv = nn.Linear(hidden_size,1)
    def forward(self, queries, keys, cum_subs_masks, num_neg):
        '''
        queries (batch_size, num_neg+1, embs)
        keys (batch_size, num_neg+1, num_keys, embs)
        '''
        queries = queries.unsqueeze(-2) #(bs, num_neg+1, 1, embs)
        scores =self.wv(F.tanh(self.wq(queries) + self.wk(keys)))
        if cum_subs_masks is not None:
            scores = scores + (cum_subs_masks*-1e9)
        attn_weight = F.softmax(scores, dim=-2) #(bs, neg_num+1, key_num, 1)
        attn_out = attn_weight * keys # attn_out/keys:(bs, num_neg+1, key_num, key_size)
        attn_out = torch.sum(attn_out, dim=-2) #(bs, num_neg+1, key_size)
        return attn_out, attn_weight

class Pref_Austgn(nn.Module):
    def __init__(self, num_x, pref_embs, stgn_embs, mlp_units, num_layers, num_head, num_rec):
        '''
        mlp_units :
            pref_mlp_units = [512, 128]
            mlp_units = (pref_mlp_units, [1024, 512, 1])
        '''
        super().__init__()
        self.hidden_size = pref_embs[0]
        self.query_size = sum(pref_embs[1:])

        self.preference_model = Preference_Model(pref_embs, num_x, num_layers, num_head, mlp_units[0])
        self.sequential_model = Sequential_Model(stgn_embs, num_x, mlp_units[1], num_rec) 
    
        self.itst_mlp_bn = MLP_BN4d(mlp_units[0], 5, self.hidden_size)
        self.inner_attn = BahdanauAttention_softmax(mlp_units[0][-1], self.query_size, self.hidden_size)
        self.out_mlp = MLP_LN(mlp_units[1], mlp_units[0][-1])

    def forward(self, inputs, num_neg):
        '''
        inputs = [austgn_inputs, pref_inputs, y_inputs, neg_inputs]
        '''
        austgn_inputs, pref_inputs, y_inputs, neg_inputs = inputs
        pref_out, shuffled_indices = self.preference_model(pref_inputs, y_inputs, neg_inputs, num_neg)
        seq_out = self.sequential_model(austgn_inputs)
        
        pref_day, pref_hour, pref_geo, queries = pref_out # (batch_size, neg_num+1, embs)
        seq_poi, seq_cat = seq_out # (batch_size, embs)

        # 需要对齐neg_num，否则没法和pref_out stack
        seq_poi = torch.unsqueeze(seq_poi, dim=1).expand(-1, num_neg+1, -1)
        seq_cat = torch.unsqueeze(seq_cat, dim=1).expand(-1, num_neg+1, -1)

        # inner attn
        inner_keys = torch.stack([pref_day, pref_hour, pref_geo, seq_poi, seq_cat], dim = 2)# (batch_size, neg_num+1, key_num, embs)
        inner_keys_bn = self.itst_mlp_bn(inner_keys) #(batch_size, neg_num, key_nums, embs)

        inner_attn_out, _ = self.inner_attn(queries, inner_keys_bn, None, num_neg) # q(batch_size, num_neg, embs);key(batch_size, key_num, embs);out(batch_size, neg_num, key_size)

        model_out = self.out_mlp(inner_attn_out) #(batch_size, neg_num, 1)
        
        return model_out, torch.stack(shuffled_indices) #返回indice张量
        
