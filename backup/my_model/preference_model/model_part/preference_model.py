import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import datetime
import os
class BahdanauAttention_softmax(nn.Module):
    def __init__(self,key_size,query_size,hidden_size,dropout):
        super(BahdanauAttention_softmax,self).__init__()
        self.wk=nn.Linear(key_size,hidden_size)
        self.wq=nn.Linear(query_size,hidden_size)
        self.wv=nn.Linear(hidden_size,1)
        self.dropout=nn.Dropout(dropout)
    def forward(self,queries,keys,cum_subs_masks):
        '''
        queries:(bs,sq,21,embs)
        keys:(bs,sq,7,hidden_size)
        values(bs,sq,7,embs),即keys
        cum_subs_masks:(bs,sq,7,1)，某时间步在星期x没有打卡则取1
        return:
                attn_out：每个q对k的分数(bs,sq,21,hidden_size)
                attn_weight:最后attn的输出分数(bs,sq,21,7,1)
        '''
        queries,keys=self.dropout(queries),self.dropout(keys)

        queries=queries.unsqueeze(-2)#(bs,sq,21,1,embs),扩展一个维度，不复制是因为在相加的时候会广播
        keys=keys.unsqueeze(2).expand(-1,-1,21,-1,-1)#为了对21个样本都进行计算分数(bs,sq,7,hidden_size)->(bs,sq,21,7,hidden)
        masks=cum_subs_masks.unsqueeze(2).expand(-1,-1,21,-1,-1)#(bs,sq,7,embs)->(bs,sq,21,7,1)
        scores=self.wv(F.tanh(self.wq(queries)+self.wk(keys)))#(bs,sq,21,1,hidden)+(bs,sq,21,7,hidden)=(bs,sq,21,7,hidden)->tanh之后不变，->(bs,sq,21,7,1)
        if masks is not None:#如果某时间步在某星期没有打卡，则在scores（注意力分数）加一个很大的负偏执，使得在softmax取值趋向0，抑制对这个元素的关注
            scores+=(masks*-1e9)
        attn_weights=F.softmax(scores,dim=-2)#(bs,sq,21,7,1)
        attn_out=attn_weights*keys#(bs,sq,21,7,1)*(bs,sq,21,7,hidden_size)->(bs,sq,21,7,hidden_size)
        attn_out=torch.sum(attn_out,dim=-2)#(bs,sq,21,hidden_size)，将7个hidden_size加到一起
        return attn_out,attn_weights


class MLP(nn.Module):
    def __init__(self,mlp_units,mlp_size,acti=torch.relu,rate=0.1):
        super(MLP,self).__init__()
        '''
        mlp_units:list,由x映射到到mlp_units[0]->mlp_units[1]->mlp_units[2]
        mlp_size:int,输入维度
        '''
        self.dropout=nn.Dropout(rate)

        self.layernorm1=nn.LayerNorm(mlp_units[0],eps=1e-6)
        self.layernorm2=nn.LayerNorm(mlp_units[1],eps=1e-6)

        self.dense1=nn.Linear(mlp_size,mlp_units[0])
        self.dense2=nn.Linear(mlp_units[0],mlp_units[1])
        self.dense_score=nn.Linear(mlp_units[1],mlp_units[2])
        self.acti1=acti
        self.acti2=F.sigmoid
    def forward(self,x):
        '''
        x:[bs,sq,hidden_size+embs_size]#x是attn concat queries结果，attn之前对keys做了dense
        '''
        x=self.dropout(x)
        x=self.acti1(self.dense1(x))
        x=self.layernorm1(x)

        x=self.dropout(x)
        x=self.acti1(self.dense2(x))
        x=self.layernorm2(x)

        y=self.dense_score(x)
        y=self.acti2(y)#(bs,sq,1)
        y=torch.squeeze(y,dim=-1)
        return y


class Preference_Model(nn.Module):
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
    def __init__(self,embed_size_poi,embed_size_cat_id,
                 embed_size_days,embed_size_slot,embed_size_geohsh,
                 hidden_size,
                 size_cat_id,size_poi,size_days,size_slots,size_geohshs,
                 num_layers,dropout=0.1):
        '''
        sqlen_minus:seq_len-2
        '''
        super(Preference_Model,self).__init__()

        embs_size=embed_size_poi+embed_size_cat_id+embed_size_days+embed_size_slot+embed_size_geohsh
        mlp_size=hidden_size+embs_size
        # self.sqlen_minus=sqlen_minus

        self.embed_poi=nn.Embedding(size_poi,embed_size_poi)
        self.embed_cats_id=nn.Embedding(size_cat_id,embed_size_cat_id)
        self.embed_days=nn.Embedding(size_days,embed_size_days)
        self.embed_slots=nn.Embedding(size_slots,embed_size_slot)
        self.embed_geohshs=nn.Embedding(size_geohshs,embed_size_geohsh)
        self.dense=nn.Linear(embs_size,hidden_size)
        self.attn=BahdanauAttention_softmax(hidden_size,embs_size,hidden_size,dropout)
        # self.mlp=MLP(mlp_units,mlp_size)

    def fun_avg_pooling(self,poi_embs,ctxt_embs,subs_days_masks):
        '''
        根据样本打卡星期x，将样本划分进入不同类别和时间，并池化
        poi_embs:(batch_size,seq_len,embs_size_pois)
        ctxt_embs:
        sub_days_masks=(batch_size,seq_len,7)
        return:
                avg_subs_days=(batch_size,seq_len,7,embs_size_pois+embs_size_ctxt)
        '''
        def divide_no_nan(x,y,nan_value=0.0):
            mask=y==0
            y=torch.where(mask,torch.ones_like(y),y)
            result=x/y
            result=torch.where(mask,torch.tensor(nan_value),result)
            return result
        subs_days_masks=subs_days_masks.float()
        cum_idx=torch.cumsum(subs_days_masks,dim=1)#用户在某时间步前，在周x打卡个数(batch_size,seqlen,7)
        cum_idx_mask=(cum_idx==0.).float()#如果某时间步在该星期x没有打卡，则取1,(batch_size,seqlen,7)
        cum_idx_mask=cum_idx_mask.unsqueeze(-1)#扩展一维（bs，sq，7，1）

        inputs_embs=torch.cat((poi_embs,ctxt_embs),2)
        inputs_embs_expand=inputs_embs.unsqueeze(2)#(bs,sl,1,embs_size)
        inputs_subs_idx=subs_days_masks.unsqueeze(3)#(bs,sl,7,1),用户的打卡在星期上的扩展掩码
        inputs_embs_expand_subs=inputs_embs_expand*inputs_subs_idx#(bs,sq,7,embs_size)将inputs_embs_expand划分到7类

        inputs_subs_idx_cum=torch.cumsum(inputs_subs_idx,dim=1)
        inputs_embs_expand_subs_cum=torch.cumsum(inputs_embs_expand_subs,dim=1)
        inputs_embs_expand_subs_cum_avg=divide_no_nan(inputs_embs_expand_subs_cum,inputs_subs_idx_cum)

        cum_subs_avg=inputs_embs_expand_subs_cum_avg#pooling后的每周的打卡嵌入(bs,sq,7,embs)
        cum_subs_mask=cum_idx_mask#(bs,sl,7,1)某时间步在星期x没有打卡则取1
        return cum_subs_avg,cum_subs_mask
        
    def forward(self,x_poi_int,x_cat_int,x_day,x_hour,x_hs5_int,x_days_subs,
                    neg_poi_int,neg_cat_int,
                    y_poi_int,y_cat_int,y_day,y_hour,y_hs5):
        '''
        x_poi_int,x_cat_int,x_day,x_hour,x_hs5_int:(bs,sqlen-2)
        x_days_subs:(bs,sqlen-2,7)
        neg_poi_int,neg_cat_int:(bs,sqlen-2,negnum)
        y_poi_int,y_cat_int,y_day,y_hour,y_hs5:(bs,sqlen-2)

        '''
        #1.embed
        x_poi_int=self.embed_poi(x_poi_int)#->(batch_size,seq_pad,embed)
        x_cat_int=self.embed_cats_id(x_cat_int)
        x_day=self.embed_days(x_day)
        x_hour=self.embed_slots(x_hour)
        x_hs5=self.embed_geohshs(x_hs5_int)

        #(bs,sq-2,neg_num,emb)
        neg_poi_int=self.embed_poi(neg_poi_int)
        neg_cat_int=self.embed_cats_id(neg_cat_int)
        neg_num=neg_poi_int.size()[2]
        
        y_poi_int=self.embed_poi(y_poi_int)#正样本(bs,sq-2,emb)
        y_cat_int=self.embed_cats_id(y_cat_int)#正样本
        y_day=self.embed_days(y_day)#q的context
        y_hour=self.embed_slots(y_hour)
        y_hs5=self.embed_geohshs(y_hs5)
         

        x_poi_embs=torch.cat((x_poi_int,x_cat_int),2)
        x_ctxt_embs=torch.cat((x_day,x_hour,x_hs5),2)

        #1做正负样本拼接，正样本放在第一个
        neg_poi_info_embs=torch.cat((neg_poi_int,neg_cat_int),-1)
        posi_poi_info_embs=torch.cat((y_poi_int,y_cat_int),-1)
        ipt_q=torch.cat((posi_poi_info_embs.unsqueeze(2),neg_poi_info_embs),dim=2)#正样本和负样本的poi_embs拼接
        #1.2正负样本poi_info和ctxt_info拼接
        ipt_q=torch.cat((ipt_q,y_day.unsqueeze(2).expand(-1,-1,neg_num+1,-1),
                               y_hour.unsqueeze(2).expand(-1,-1,neg_num+1,-1),
                               y_hs5.unsqueeze(2).expand(-1,-1,neg_num+1,-1)),dim=-1)
        

        #pooling:每个时间步都pooling了
        cum_subs_avg,cum_subs_mask=self.fun_avg_pooling(x_poi_embs,x_ctxt_embs,x_days_subs)#embs=512,(batch_size,seq_len,7,embs_size_pois+embs_size_ctxt)
        #pooling->fc
        cum_subs_avg=self.dense(cum_subs_avg)#(bs,sq,7,emb_size)->(bs,sq,7,hidden_size)
        #att
        keys=cum_subs_avg#(bs,sq,7,hidden_size)
        queries=ipt_q#(bs,sq,21,embs)

        attn_out,attn_w=self.attn(queries,keys,cum_subs_mask)#(bs,sq,21,hidden_size),(bs,sq,21,7,1)
        #mlp
        concat=torch.cat((attn_out,queries),dim=-1)#(bs,sq,21,hidden_size),(bs,sq,21,embs)->(bs,sq,21,hidden_size+embs_size)
        # mlp_out=self.mlp(concat)#(bs,seqlen,21)
        return concat


def Process_preference_model_train_state(train_x_poi_int,train_x_cat_int,train_x_day,train_x_hour,train_x_hs5_int,train_x_days_subs,
                                        negs_poi_int,negs_cat_int,
                                        train_y_poi_int,train_y_cat_int,train_y_day,train_y_hour,train_y_hs5,batch_size):
    class Pre_Dataset(Dataset):
        def __init__(self,train_x_poi_int,train_x_cat_int,train_x_day,train_x_hour,train_x_hs5_int,train_x_days_subs,
                        negs_poi_int,negs_cat_int,
                        train_y_poi_int,train_y_cat_int,train_y_day,train_y_hour,train_y_hs5):
            self.train_x_poi_int=train_x_poi_int
            self.train_x_cat_int=train_x_cat_int
            self.train_x_day=train_x_day
            self.train_x_hour=train_x_hour
            self.train_x_hs5_int=train_x_hs5_int
            self.train_x_days_subs=train_x_days_subs
            
            self.negs_poi_int=negs_poi_int
            self.negs_cat_int=negs_cat_int

            self.train_y_poi_int=train_y_poi_int
            self.train_y_cat_int=train_y_cat_int
            self.train_y_day=train_y_day
            self.train_y_hour=train_y_hour
            self.train_y_hs5=train_y_hs5

        def __len__(self):
            return len(self.train_x_poi_int)
        def __getitem__(self,index):
            x_poi_int=self.train_x_poi_int[index]
            x_cat_int=self.train_x_cat_int[index]
            x_day=self.train_x_day[index]
            x_hour=self.train_x_hour[index]
            x_hs5_int=self.train_x_hs5_int[index]
            x_days_subs=self.train_x_days_subs[index]

            neg_poi_int=self.negs_poi_int[index]
            neg_cat_int=self.negs_cat_int[index]

            y_poi_int=self.train_y_poi_int[index]
            y_cat_int=self.train_y_cat_int[index]
            y_day=self.train_y_day[index]
            y_hour=self.train_y_hour[index]
            y_hs5=self.train_y_hs5[index]
            return x_poi_int,x_cat_int,x_day,x_hour,x_hs5_int,x_days_subs,neg_poi_int,neg_cat_int,y_poi_int,y_cat_int,y_day,y_hour,y_hs5
    #创建数据集实例
    all_datasets=Pre_Dataset(train_x_poi_int,train_x_cat_int,train_x_day,train_x_hour,train_x_hs5_int,train_x_days_subs,
                                        negs_poi_int,negs_cat_int,
                                        train_y_poi_int,train_y_cat_int,train_y_day,train_y_hour,train_y_hs5)
    #创建dataloader
    dataloader=DataLoader(all_datasets,batch_size,shuffle=True)
    return dataloader
