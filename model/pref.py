import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from utils.utils import MLP_LN_SIGMOID


class BahdanauAttention_softmax(nn.Module):
    def __init__(self, key_size, query_size, hidden_size):
        super().__init__()
        self.wk = nn.Linear(key_size, hidden_size)
        self.wq = nn.Linear(query_size, hidden_size)
        self.wv = nn.Linear(hidden_size,1)
    def forward(self, queries, keys, cum_subs_masks, num_neg):
        '''
        queries (batch_size, num_neg+1, embs)
        keys (batch_size, num_keys, embs)
        '''
        queries = queries.unsqueeze(-2) #(bs, num_neg, 1, embs)
        keys = keys.unsqueeze(1).expand(-1, num_neg+1, -1, -1)
        scores =self.wv(F.tanh(self.wq(queries) + self.wk(keys)))
        if cum_subs_masks is not None:
            scores = scores + (cum_subs_masks*-1e9)
        attn_weight = F.softmax(scores, dim=-2) #(bs, neg_num+1, key_num, 1)
        attn_out = attn_weight * keys # attn_out/keys:(bs, num_neg+1, key_num, key_size)
        attn_out = torch.sum(attn_out, dim=-2) #(bs, num_neg+1, key_size)
        return attn_out, attn_weight


class Preference_Model(nn.Module):
    def __init__(self, pref_embs, num_x, num_layers, num_head, mlp_units):
        '''
        pref_embs: embedding的维度
                        0hidden;1poi;2cat;3day;4hour;5geo
            pref_embs = [256, 64, 32, 8, 16, 32]
        num_x:被embedding的个数
                    0user;1poi;2cat;3geo;4day;5hour;6num_rec
            num_x = [1078, 3906, 285, 95, 20, 8, 25]   
        '''
        super().__init__()
        self.embed_poi = nn.Embedding(num_x[1], pref_embs[1])
        self.embed_cat = nn.Embedding(num_x[2], pref_embs[2])
        self.embed_geo = nn.Embedding(num_x[3], pref_embs[3])
        self.embed_day = nn.Embedding(num_x[4], pref_embs[4])
        self.embed_hour = nn.Embedding(num_x[5], pref_embs[5])
        hidden_size = pref_embs[0]

        query_size = sum(pref_embs[1:])
        mlp_size = query_size
        self_attn_key_size = mlp_units[-1] # key经过mlp层最后得到的维度

        self.dense_day = MLP_LN_SIGMOID(mlp_units, mlp_size)
        self.dense_hour = MLP_LN_SIGMOID(mlp_units, mlp_size)
        self.dense_geo = MLP_LN_SIGMOID(mlp_units, mlp_size)
        self.attn = BahdanauAttention_softmax(mlp_units[-1], query_size, hidden_size)
        self.multihead_attn_day = nn.MultiheadAttention(self_attn_key_size, num_head, batch_first=True, add_bias_kv=True)
        self.multihead_attn_hour = nn.MultiheadAttention(self_attn_key_size, num_head, batch_first=True, add_bias_kv=True)
        self.multihead_attn_geo = nn.MultiheadAttention(self_attn_key_size, num_head, batch_first=True, add_bias_kv=True)
    def fun_avg_pooling(self,inputs_embs,subs_days_masks):
        '''
        根据样本打卡星期x，将样本划分进入不同类别和时间，并池化
        poi_embs:(batch_size, seq_len, embs_size_pois)
        ctxt_embs:
        sub_days_masks=(batch_size, seq_len, 7)
        return:
                avg_subs_days=(batch_size,seq_len,7,embs_size_pois+embs_size_ctxt)
        '''
        def divide_no_nan(x, y, large_value=1e9):
            mask = y == 0
            y = y.masked_fill(mask, large_value)
            result=x / y
            return result
        
        subs_days_masks=subs_days_masks.float()
        cum_idx=torch.cumsum(subs_days_masks,dim=1)#用户在某时间步前，在周x打卡个数(batch_size,seqlen,7)
        cum_idx_mask=(cum_idx==0.).float()#如果某时间步在该星期x没有打卡，则取1,(batch_size,seqlen,7)
        cum_idx_mask=cum_idx_mask.unsqueeze(-1)#扩展一维（bs，sq，7，1）

        # inputs_embs=torch.cat((poi_embs,ctxt_embs),2)#(b,seq,emb)
        inputs_embs_expand=inputs_embs.unsqueeze(2)#(bs,sl,1,embs_size)
        inputs_subs_idx=subs_days_masks.unsqueeze(3)#(bs,sl,7,1),用户的打卡在星期上的扩展掩码
        inputs_embs_expand_subs=inputs_embs_expand*inputs_subs_idx#(bs,sq,7,embs_size)将inputs_embs_expand划分到7类

        inputs_subs_idx_cum=torch.cumsum(inputs_subs_idx,dim=1)
        inputs_embs_expand_subs_cum=torch.cumsum(inputs_embs_expand_subs,dim=1)

        inputs_embs_expand_subs_cum_avg=divide_no_nan(inputs_embs_expand_subs_cum,inputs_subs_idx_cum)

        cum_subs_avg=inputs_embs_expand_subs_cum_avg#pooling后的每周的打卡嵌入(bs,sq,7,embs)
        cum_subs_mask=cum_idx_mask#(bs,sl,7,1)某时间步在星期x没有打卡则取1
        return cum_subs_avg[:,1,:,:].squeeze(1), cum_subs_mask[:,-1,:,:].squeeze(1)
        
    def forward(self, pref_inputs, y_inputs, neg_inputs, num_neg):

        '''
        pref_inputs = [0x_poi, 1x_cat, 2x_day, 3x_hour, 4x_geo,\
            5x_sub_day, 6x_sub_hour, 7x_sub_geo]
        neg_inputs = [neg_poi, neg_cat]
        y_inputs = [0y_poi, 1y_cat, 2y_day, 3y_hour, 4y_geo]
        num_neg = [tra_neg/tes_neg]
        '''
        x_poi = self.embed_poi(pref_inputs[0]) # (batch_size, sq_len, emb_size)
        x_cat = self.embed_cat(pref_inputs[1])
        x_day = self.embed_day(pref_inputs[2])
        x_hour = self.embed_hour(pref_inputs[3])
        x_geo = self.embed_geo(pref_inputs[4])

        #(bs, neg, emb)
        neg_poi = self.embed_poi(neg_inputs[0]) #(batch_size, neg_num, emb_size)
        neg_cat = self.embed_cat(neg_inputs[1]) #(batch_size, neg_num, emb_size)
        num_neg = num_neg #

        # y
        y_poi = self.embed_poi(y_inputs[0]) #(batch_size, emb_size)
        y_cat = self.embed_cat(y_inputs[1]) 
        y_day = self.embed_day(y_inputs[2])
        y_hour = self.embed_hour(y_inputs[3])
        y_geo = self.embed_geo(y_inputs[4])

        x_poi_embs = torch.cat((x_poi, x_cat), -1) #(bs, sl, emb)
        x_ctxt_embs = torch.cat((x_day, x_hour, x_geo), -1) #（bs, sl, emb)
        inputs_embs = torch.cat((x_poi_embs, x_ctxt_embs), 2)

        # 1. 做正样本，正样本放第一个
        neg_poi_info_embs=torch.cat((neg_poi,neg_cat),-1)#(b,neg_num,emb)
        posi_poi_info_embs=torch.cat((y_poi,y_cat),-1)#(b,emb)
        ipt_q=torch.cat((posi_poi_info_embs.unsqueeze(1),neg_poi_info_embs),dim=1)#正样本和负样本的poi_embs拼接
        #打乱正负样本位置和对应candipt_q=torch.cat((posi_poi_info_embs.unsqueeze(1),neg_poi_info_embs),dim=1)#正样本和负样本的poi_embs
        shuffled_indices=[torch.randperm(ipt_q.size(1)) for _ in range(ipt_q.size(0))]#(bs,neg_num) 
        ipt_q=torch.stack([ipt_q[i,shuffled_indices[i],:] for i in range(ipt_q.size(0))])#(b, neg_num, embs)
        ipt_q=torch.cat((ipt_q, y_day.unsqueeze(1).expand(-1,num_neg+1,-1),
                               y_hour.unsqueeze(1).expand(-1,num_neg+1,-1),
                               y_geo.unsqueeze(1).expand(-1,num_neg+1,-1)),dim=-1)#(b,neg_num+1,embs)
        
        #pooling:每个时间步都pooling了
            #day pooling
        cum_subs_avg,cum_subs_mask=self.fun_avg_pooling(inputs_embs,pref_inputs[5])#embs=512,(batch_size,sq,7,embs_size_pois+embs_size_ctxt)
        cum_subs_mask=cum_subs_mask.unsqueeze(1).expand(-1,num_neg+1,-1,-1)
            #hour pooling
        cum_subs_avg_hour,cum_subs_mask_hour=self.fun_avg_pooling(inputs_embs,pref_inputs[6]) # (batch_size, sq, 24, embs)
        cum_subs_mask_hour=cum_subs_mask_hour.unsqueeze(1).expand(-1,num_neg+1,-1,-1)
            #zone pooling
        # cum_subs_avg_hsh,cum_subs_mask_hsh=self.fun_avg_pooling(inputs_embs,pref_inputs[7]) #(batch_size, sq, 95, embs)
        # cum_subs_mask_hsh=cum_subs_mask_hsh.unsqueeze(1).expand(-1,num_neg+1,-1,-1)

         #pooling->fc
            #day
        cum_subs_avg=self.dense_day(cum_subs_avg)#(bs,7,emb_size)->(bs,7,mlp_unit[-1])
            #hour
        cum_subs_avg_hour=self.dense_hour(cum_subs_avg_hour)
            #zoom
        # cum_subs_avg_hsh=self.dense_geo(cum_subs_avg_hsh)

            #att
        #att
            #day
        keys=cum_subs_avg#(bs,sq,7,hidden_size)|(bs,7,hidden_size)
        queries=ipt_q#(bs,sq,21,embs)|(bs,neg_num,embs),公用
        attn_out_day,attn_w=self.attn(queries,keys,cum_subs_mask,num_neg)#(bs,sq,21,hidden_size),(bs,sq,21,7,1)|(bs,21,h),(bs,7,embs)
        self_attn_out_day, _ = self.multihead_attn_day(attn_out_day, attn_out_day, attn_out_day)#k, q, v (bs, neg_num+1, h)
            #hour
        keys_hour=cum_subs_avg_hour
        attn_out_hour,_=self.attn(queries,keys_hour,cum_subs_mask_hour,num_neg)
        self_attn_out_hour, _ = self.multihead_attn_hour(attn_out_hour, attn_out_hour, attn_out_hour)
            #hsh
        # keys_hsh=cum_subs_avg_hsh
        # attn_out_hsh,_=self.attn(queries,keys_hsh,cum_subs_mask_hsh,num_neg)
        # self_attn_out_geo, _ = self.multihead_attn_geo(attn_out_hsh, attn_out_hsh, attn_out_hsh)
        # pref_out = (self_attn_out_day, self_attn_out_hour, self_attn_out_geo, queries)

        pref_out = (self_attn_out_day, self_attn_out_hour, queries)
        return pref_out, shuffled_indices