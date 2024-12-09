import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import datetime
import os
import sys
sys.path.append('code/my_code/my_model/stgn_model/')
sys.path.append('code/my_code/my_model/preference_model/model_part2/')
from stgn_model.stgn import STGN,STGN_MLP
from preference_model.model_part2.preference_model import BahdanauAttention_softmax

class Abl_Two_Model(nn.Module):
    '''
    所用到的数据:
                POI id
                category id,原文中一级，此处用一级，否则丢失信息
                day of week
                slot
                geohash
                **没有**使用用户id
    embed->pooling->dense->att->mlp
    '''
    #embed,使用和sequential model一样的embed_size
    def __init__(self,pref_embs,
                 num_x_long,
                 num_layers,head_num,dropout,
                 hidden_sz,num_x_short,sz_embs,num_rec,mlp_units):
        '''
                                          0    1   2   3   4   5
        pref_embs=[128,64,32,8,16,32]#(hidden,poi,cat,day,hour,hsh5)
                           0                1             2              3                    4     
        sz_embs:列表[embed_size_user,embed_size_poi,embed_size_cat,embed_size_month/hours,embed_size_hsh5]
        sqlen_minus:seq_len-2
        '''
        super().__init__()

        embs_size=sum(pref_embs[1:])+sum(sz_embs[1:])+pref_embs[3]
        mlp_size=sum(pref_embs[1:])
        # self.sqlen_minus=sqlen_minus

        self.embed_poi_long=nn.Embedding(num_x_long[1],pref_embs[1])#embedding映射是个数，而不是最大的数
        self.embed_cats_id_long=nn.Embedding(num_x_long[2],pref_embs[2])
        # self.embed_months=nn.Embedding(num_x[3],pref_embs[3])
        self.embed_days_long=nn.Embedding(num_x_long[5],pref_embs[3])
        self.embed_slots_long=nn.Embedding(num_x_long[6],pref_embs[4])
        self.embed_geohshs_long=nn.Embedding(num_x_long[3],pref_embs[5])
        self.dense=nn.Linear(mlp_size,pref_embs[0])
        self.attn=BahdanauAttention_softmax(pref_embs[0],embs_size,pref_embs[0],dropout)
        self.multihead_attn=nn.MultiheadAttention(embs_size,head_num,batch_first=True,add_bias_kv=True)
        # self.mlp=MLP(mlp_units,mlp_size)

        ##########################################stgn
        self.num_rec=num_rec

        self.embed_user_short=nn.Embedding(num_x_short[0],sz_embs[0])
        self.embed_poi_short=nn.Embedding(num_x_short[1],sz_embs[1])
        self.embed_cat_short=nn.Embedding(num_x_short[2],sz_embs[2])
        self.embed_day_short=nn.Embedding(num_x_long[5],pref_embs[3])
        self.embed_hour_short=nn.Embedding(num_x_short[3],sz_embs[3])
        self.embed_hsh5_short=nn.Embedding(num_x_short[4],sz_embs[4])
        #组合2
        self.stgn_cat=STGN(sz_embs[0]+pref_embs[2]+sum(pref_embs[4:]),hidden_sz)#t,d,user,hsh5,hour,cat
        self.stgn_loc=STGN(sz_embs[0]+pref_embs[1]+sum(pref_embs[4:]),hidden_sz)#t,d,user,poi,hour,hsh5
        #组合1
        # self.stgn_cat=STGN(sum(sz_embs[:-1]),hidden_sz)#无loc
        # self.stgn_loc=STGN(sum(sz_embs)-sz_embs[2],hidden_sz)#无cat
        self.mlp=STGN_MLP(mlp_units,hidden_sz)

    def fun_avg_pooling(self,inputs_embs,subs_days_masks):
        '''
        根据样本打卡星期x，将样本划分进入不同类别和时间，并池化
        poi_embs:(batch_size,seq_len,embs_size_pois)
        ctxt_embs:
        sub_days_masks=(batch_size,seq_len,7)
        return:
                avg_subs_days=(batch_size,seq_len,7,embs_size_pois+embs_size_ctxt)
        '''
        def divide_no_nan(x, y, large_value=1e9):
            '''
            y (bs, sq, 7, 1),是计数, y可以不参与反向传播
            x (bs, sq, 7, embs), 是emb, 参与反向传播
            '''
            mask = y == 0 #(bs, sq, 7, 1)
            y = y.masked_fill(mask, large_value)
            result = x / y # 广播（bs, sq, 7, emb)
            return result

        subs_days_masks=subs_days_masks.float() # (batch_size, seq_len, 7) 标记是否在周x打卡
        cum_idx=torch.cumsum(subs_days_masks,dim=1)# 用户在某时间步前，在周x打卡个数(batch_size,seqlen,7)
        cum_idx_mask=(cum_idx==0.).float()# 如果某时间步在该星期x没有打卡，则取1(true),(batch_size,seqlen,7)
        cum_idx_mask=cum_idx_mask.unsqueeze(-1)#扩展一维（bs，sq，7，1）

        # inputs_embs=torch.cat((poi_embs,ctxt_embs),2)#(b,seq,emb)
        inputs_embs_expand=inputs_embs.unsqueeze(2)#(bs,sl,1,embs_size)
        inputs_subs_idx=subs_days_masks.unsqueeze(3)#(bs,sl,7,1),用户的打卡在星期上的扩展掩码
        inputs_embs_expand_subs=inputs_embs_expand*inputs_subs_idx#(bs,sq,7,embs_size)将inputs_embs_expand划分到7类 | zoom: 3g

        inputs_subs_idx_cum=torch.cumsum(inputs_subs_idx,dim=1)
        inputs_embs_expand_subs_cum=torch.cumsum(inputs_embs_expand_subs,dim=1)
        inputs_embs_expand_subs_cum_avg=divide_no_nan(inputs_embs_expand_subs_cum,inputs_subs_idx_cum)

        cum_subs_avg=inputs_embs_expand_subs_cum_avg#pooling后的每周的打卡嵌入(bs,sq,7,embs)
        cum_subs_mask=cum_idx_mask#(bs,sl,7,1)某时间步在星期x没有打卡则取1
        return cum_subs_avg[:,-1,:,:].squeeze(1), cum_subs_mask[:,-1,:,:].squeeze(1)
        
    def forward(self,pref_inputs,num_neg,input):
        '''            
                        0     1     2       3       4      5         6          7         8       9           10         11       12     
        pref_inputs=[ x_poi,x_cat,x_month,x_day,x_hour,x_hsh5,x_sub_month,x_sub_day,x_sub_hour,x_sub_hsh5,x_neg_poi,x_neg_cat,x_neg_second,\
                      13     14       15    16     17     18
                    y_cat,y_second,y_month,y_day,y_hour,y_hsh5,\
                      19
                    y_poi_int]
                    0       1     2     3      4      5         6            7         8         9       10    11    12     13
        pref_input=[x_poi,x_cat,x_day,x_hour,x_hsh5,x_sub_day,x_sub_hour,x_sub_hsh5,x_neg_poi,x_neg_cat,y_cat,y_day,y hour,y_hsh5,\
                        14
                    y_poi_int]

        x_poi_int,x_cat_int,x_day,x_hour,x_hs5_int:(bs,sqlen-2)
        x_days_subs:(bs,sqlen-2,7)
        neg_poi_int,neg_cat_int:(bs,sqlen-2,negnum)
        neg_second:|(bs,neg_num)
        y_poi_int,y_cat_int,y_day,y_hour,y_hs5:(bs,sqlen-2)

        '''
        #1.embed
        x_poi_int=self.embed_poi_long(pref_inputs[0])#->(batch_size,t_max,embs)
        x_cat_int=self.embed_cats_id_long(pref_inputs[1])
        # x_month=self.embed_months(pref_inputs[2])
        x_day=self.embed_days_long(pref_inputs[2])
        x_hour=self.embed_slots_long(pref_inputs[3])
        x_hs5=self.embed_geohshs_long(pref_inputs[4])
        print("poi维度：", pref_inputs[0].shape)
        # print("cat维度：", pref_inputs[1].shape)
        # print("day维度：", pref_inputs[2].shape)
        # print("hour维度：", input[3].shape)
        # print("hsh维度：", input[4].shape)
        #(bs,neg_num,emb)
        neg_poi_int_long=self.embed_poi_long(pref_inputs[8])
        neg_cat_int_long=self.embed_cats_id_long(pref_inputs[9])
        neg_num=neg_poi_int_long.size()[1]
        neg_poi_int_short=self.embed_poi_short(pref_inputs[8])
        neg_cat_int_short=self.embed_cat_short(pref_inputs[9])

        neg_poi_int=torch.cat((neg_poi_int_long,neg_poi_int_short),-1)
        neg_cat_int=torch.cat((neg_cat_int_long,neg_cat_int_short),-1)

        y_poi_int_long=self.embed_poi_long(pref_inputs[14])#正样本(bs,emb)
        y_cat_int_long=self.embed_cats_id_long(pref_inputs[10])#正样本
        # y_month=self.embed_months(pref_inputs[15])
        y_day_long=self.embed_days_long(pref_inputs[11])#q的context
        y_hour_long=self.embed_slots_long(pref_inputs[12])
        y_hs5_long=self.embed_geohshs_long(pref_inputs[13])

        y_poi_int_short=self.embed_poi_short(pref_inputs[14])
        y_cat_int_short=self.embed_cat_short(pref_inputs[10])
        y_day_short=self.embed_day_short(pref_inputs[11])
        y_hour_short=self.embed_hour_short(pref_inputs[12])
        y_hs5_short=self.embed_hsh5_short(pref_inputs[13])

        y_poi_int=torch.cat((y_poi_int_long,y_poi_int_short),-1)
        y_cat_int=torch.cat((y_cat_int_long,y_cat_int_short),-1)
        y_day=torch.cat((y_day_long,y_day_short),-1)
        y_hour=torch.cat((y_hour_long,y_hour_short),-1)
        y_hs5=torch.cat((y_hs5_long,y_hs5_short),-1)
        

        x_poi_embs=torch.cat((x_poi_int,x_cat_int),-1)
        x_ctxt_embs=torch.cat((x_day,x_hour,x_hs5),-1)
        inputs_embs=torch.cat((x_poi_embs,x_ctxt_embs),2)
        
        #1做正负样本拼接，正样本放在第一个
        neg_poi_info_embs=torch.cat((neg_poi_int,neg_cat_int),-1)#(b,neg_num,emb) 200M
        posi_poi_info_embs=torch.cat((y_poi_int,y_cat_int),-1)#(b,1,emb) 
        ipt_q=torch.cat((posi_poi_info_embs.unsqueeze(1),neg_poi_info_embs),dim=1)#正样本和负样本的poi_embs拼接 100M
        #打乱正负样本位置和对应candi_seceond_ints位置
        shuffled_indices=[torch.randperm(ipt_q.size(1)) for _ in range(ipt_q.size(0))]#(bs,neg_num)
        ipt_q=torch.stack([ipt_q[i,shuffled_indices[i],:] for i in range(ipt_q.size(0))])#(b,21,embs) 300M
        
        #1.2正负样本poi_info和ctxt_info拼接
        ipt_q=torch.cat((ipt_q,y_day.unsqueeze(1).expand(-1,neg_num+1,-1),
                               y_hour.unsqueeze(1).expand(-1,neg_num+1,-1),
                               y_hs5.unsqueeze(1).expand(-1,neg_num+1,-1)),dim=-1)#(b,21,embs) 100M
        
        #在pooling前加self-attn
        # self_attn_out,_=self.multihead_attn(inputs_embs,inputs_embs,inputs_embs)#q,kv

        #pooling:每个时间步都pooling了
            #day pooling
        cum_subs_avg,cum_subs_mask=self.fun_avg_pooling(inputs_embs,pref_inputs[5])#embs=512,(batch_size,sq,7,embs_size_pois+embs_size_ctxt) 
        # cum_subs_avg=cum_subs_avg[:,-1,:,:].squeeze(1)
        # cum_subs_mask=cum_subs_mask[:,-1:,:].squeeze(1)
        cum_subs_mask=cum_subs_mask.unsqueeze(1).expand(-1,num_neg,-1,-1)
            #hour pooling
        cum_subs_avg_hour,cum_subs_mask_hour=self.fun_avg_pooling(inputs_embs,pref_inputs[6])
        # cum_subs_avg_hour=cum_subs_avg_hour[:,-1,:,:].squeeze(1) #取了最后一个时间步, 在函数中返回 （bs, 7, emb)
        # cum_subs_mask_hour=cum_subs_mask_hour[:,-1:,:].squeeze(1) # 取最后一个时间步，在函数中返回(bs, 7, 1)
        cum_subs_mask_hour=cum_subs_mask_hour.unsqueeze(1).expand(-1,num_neg,-1,-1)
            #zoom pooling
        # cum_subs_avg_hsh,cum_subs_mask_hsh=self.fun_avg_pooling(inputs_embs,pref_inputs[7])
        # cum_subs_avg_hsh=cum_subs_avg_hsh[:,-1,:,:].squeeze(1) 
        # cum_subs_mask_hsh=cum_subs_mask_hsh[:,-1:,:].squeeze(1)
        # cum_subs_mask_hsh=cum_subs_mask_hsh.unsqueeze(1).expand(-1,num_neg,-1,-1)

        #pooling->fc
            #day
        cum_subs_avg=self.dense(cum_subs_avg)#(bs,sq,7,emb_size)->(bs,sq,7,hidden_size)|(bs,7,hidden_size)
            #month
        # cum_subs_avg_month=self.dense(cum_subs_avg_month)
            #hour
        cum_subs_avg_hour=self.dense(cum_subs_avg_hour)
            #zoom
        # cum_subs_avg_hsh=self.dense(cum_subs_avg_hsh)
        


        #att
            #day
        keys=cum_subs_avg#(bs,sq,7,hidden_size)|(bs,7,hidden_size)
        queries=ipt_q#(bs,sq,21,embs)|(bs,21,embs),公用

        attn_out,_=self.attn(queries,keys,cum_subs_mask,num_neg)#(bs,sq,21,hidden_size),(bs,sq,21,7,1)|(bs,21,h),(bs,7,embs)
            #month
        # keys_months=cum_subs_avg_month
        # attn_out_month,attn_w_month=self.attn(queries,keys_months,cum_subs_mask_month,num_neg)
            #hour
        keys_hour=cum_subs_avg_hour
        attn_out_hour,_=self.attn(queries,keys_hour,cum_subs_mask_hour,num_neg) # 3g
            #hsh
        # keys_hsh=cum_subs_avg_hsh
        # attn_out_hsh,_=self.attn(queries,keys_hsh,cum_subs_mask_hsh,num_neg) # 10g
        

        #month-pooling concat
        # concat=torch.cat((attn_out_month,attn_out,queries),dim=-1)#(bs,sq,21,hidden_size),(bs,sq,21,embs)->(bs,sq,21,hidden_size+embs_size)
        # return concat,shuffled_indices,shuffled_neg_seconds
        
        #hiera_attn
        # pref_out=attn_out,attn_out_hour,attn_out_hsh,queries
        pref_out=attn_out,attn_out_hour,queries


        ########################################stgn
        user=self.embed_user_short(input[0])#（u*t,emb)
        user=(torch.unsqueeze(user,dim=1)).repeat(1,self.num_rec,1)
        poi=self.embed_poi_long(input[1])#（u*t,rec,emb)
        cat=self.embed_cats_id_long(input[2])
        t=torch.unsqueeze(input[3],dim=-1)
        d=torch.unsqueeze(input[4],dim=-1)
        hour=self.embed_slots_long(input[5])
        hsh5=self.embed_geohshs_long(input[6])

        stgn_cat_input=torch.cat((t,d,user,hsh5,hour,cat),dim=-1)
        stgn_loc_input=torch.cat((t,d,user,hsh5,hour,poi),dim=-1)
        
        #做两个stgn，组合1
        # stgn_cat_input=torch.cat((t,d,user,poi,cat,hour),dim=-1)#(u*t,20,embs)
        # stgn_loc_input=torch.cat((t,d,user,poi,hour,hsh5),dim=-1)#(u*t,20,embs)

        #只做一个stgn
        # stgn_input=torch.cat((t,d,user,poi,cat,hour,hsh5),dim=-1)#(u*t,20,embs)
        
        _,(h_cat_t,_)=self.stgn_cat(stgn_cat_input)
        _,(h_loc_t,_)=self.stgn_loc(stgn_loc_input)
        # h_t=self.mlp(h_t)

        return pref_out,shuffled_indices,h_cat_t,h_loc_t#,shuffled_neg_seconds