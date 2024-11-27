import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import datetime


class BahdanauAttention_softmax(nn.Module):
    def __init__(self,key_size,query_size,hidden_size,dropout):
        super(BahdanauAttention_softmax,self).__init__()
        self.wk=nn.Linear(key_size,hidden_size)
        self.wq=nn.Linear(query_size,hidden_size)
        self.wv=nn.Linear(hidden_size,1)
        self.dropout=nn.Dropout(dropout)
    def forward(self,queries,keys,cum_subs_masks):
        '''
        queries:(bs,sq,embs)
        keys:(bs,sq,7,hidden_size)
        values(bs,sq,7,embs)
        cum_subs_masks:(bsm,sq,7,1)
        '''
        queries,keys=self.dropout(queries),self.dropout(keys)

        queries=queries.unsqueeze(2)#(bs,sq,1,embs)
        scores=self.wv(F.tanh(self.wq(queries)+self.wk(keys)))#(bs,sq,1,hidden_size广播bs,sq,7,hidden_size)+(bs,sq,7,hidden_size)最后->(bs,sq,7,1)
        if cum_subs_masks is not None:
            scores+=(cum_subs_masks*-1e9)
        attn_weights=F.softmax(scores,dim=-2)#(bs,sq,7,1)
        attn_out=attn_weights*keys#(bs,sq,7,1)*(bs,sq,7,hidden_size)->(bs,sq,7,hidden_size)
        attn_out=torch.sum(attn_out,dim=-2)#(bs,sq,hidden_size)
        return attn_out,attn_weights


class MLP(nn.Module):
    def __init__(self,mlp_units,mlp_size,acti=torch.relu,rate=0.1):
        super(MLP,self).__init__()
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
                 num_layers,size_tgt,
                 mlp_units,sqlen_minus,dropout=0.1):
        '''
        sqlen_minus:seq_len-1
        '''
        super(Preference_Model,self).__init__()

        embs_size=embed_size_poi+embed_size_cat_id+embed_size_days+embed_size_slot+embed_size_geohsh
        mlp_size=hidden_size+embs_size
        self.sqlen_minus=sqlen_minus

        self.embed_poi=nn.Embedding(size_poi,embed_size_poi)
        self.embed_cats_id=nn.Embedding(size_cat_id,embed_size_cat_id)
        self.embed_days=nn.Embedding(size_days,embed_size_days)
        self.embed_slots=nn.Embedding(size_slots,embed_size_slot)
        self.embed_geohshs=nn.Embedding(size_geohshs,embed_size_geohsh)
        self.dense=nn.Linear(embs_size,hidden_size)
        self.attn=BahdanauAttention_softmax(hidden_size,embs_size,hidden_size,dropout)
        self.mlp=MLP(mlp_units,mlp_size)

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
        
    def forward(self,inputs_pois,inputs_cats_id,inputs_days,inputs_slots,inputs_geohshs,inputs_subs_days_mask,
                # inputs_tgt_pois,inputs_tgt_cats_id,
                inputs_tgt_days,inputs_tgt_slots,inputs_tgt_hshs,
                inputs_negs_poi,inputs_negs_cats_id,bs):
        '''
        inputs_pois:(batch_size,seq_len_padded)
        inputs_second_id:(batch_size,seq_len_padded)

        inputs_tgts:(bs,sq)
        inputs_negs_pois:(bs,sq,negs) tensor
        bs:当前batch真实长度
        ...
        '''
        #1.embed
        inputs_pois=self.embed_poi(inputs_pois)#->(batch_size,seq_pad,embed)
        inputs_cats_id=self.embed_cats_id(inputs_cats_id)
        inputs_days=self.embed_days(inputs_days)
        inputs_slots=self.embed_slots(inputs_slots)
        inputs_geohshs=self.embed_geohshs(inputs_geohshs)

        # inputs_tgt_pois=self.embed_poi(inputs_tgt_pois)
        # inputs_tgt_cats_id=self.embed_cats_id(inputs_tgt_cats_id)
        inputs_tgt_days=self.embed_days(inputs_tgt_days)
        inputs_tgt_slots=self.embed_slots(inputs_tgt_slots)
        inputs_tgt_geohshs=self.embed_geohshs(inputs_tgt_hshs)

        all_pois=self.embed_poi(all_pois)#(3906,embs)
        cats_id_mapped_poi=self.embed_cats_id(cat_id_mapped_poi)#(3906,embd)

        poi_embs=torch.cat((inputs_pois,inputs_cats_id),2)
        ctxt_embs=torch.cat((inputs_days,inputs_slots,inputs_geohshs),2)

        all_pois_embs=all_pois.unsqueeze(0).repeat(self.sqlen_minus,1,1)#(sq,3906,embs)
        batch_all_pois_embs=all_pois_embs.unsqueeze(0).repeat(bs,1,1,1)

        all_cats_id_embs=cats_id_mapped_poi.unsqueeze(0).repeat(self.sqlen_minus,1,1)
        batch_all_cats_1st_embs=all_cats_id_embs.unsqueeze(0).repeat(bs,1,1,1)

        tgt_info_embs=torch.cat((batch_all_pois_embs,batch_all_cats_1st_embs,inputs_days.unsqueeze(2).repeat(1,1,size_poi,1),inputs_slots.unsqueeze(2).repeat(1,1,size_poi,1),inputs_geohshs.unsqueeze(2).repeat(1,1,size_poi,1)),3)
        #pooling
        cum_subs_avg,cum_subs_mask=self.fun_avg_pooling(poi_embs,ctxt_embs,inputs_subs_days_mask)#embs=512
        #pooling->fc
        cum_subs_avg=self.dense(cum_subs_avg)
        #att
        keys=cum_subs_avg#(bs,sq,7,hidden_size)
        queries=tgt_info_embs#(bs,sq,embs)
        attn_out,attn_w=self.attn(queries,keys,cum_subs_mask)#(bs,sq,hidden_size)
        #mlp
        concat=torch.cat((attn_out,queries),dim=-1)#(bs,sq,hidden_size)(bs,sq,embs)->(bs,sq,hidden_size+embs_size)
        mlp_out=self.mlp(concat)#(bs,seqlen,1)
        return mlp_out


embed_size_poi=64
embed_size_cat_id=100
embed_size_days=16
embed_size_slot=32
embed_size_geohsh=64

hidden_size=128
num_layers=2
epoch_size=10
batch_size=1
mlp_units=[256,128,1]
lr=0.0001

size_cat_id=285#虽然FourSquare官网有约1k种，但该数据集中出现285中，0-284
size_poi=3906#0-3906
size_days=8#原取值1-7，做embed取8
size_slots=25#1-24
size_geohsh=95#0-94
size_tgt=3906
seq_len=1111

#1.1用padded数据
'''
不是rec数据而是所有的数据
'''
dir_inputs_pois='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/padded_reindex_poi.pkl'
dir_inputs_cats_id='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/padded_reindex_cat_id.pkl'#使用最小目录否则丢失信息
dir_inputs_days='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/padded_days.pkl'
dir_inputs_slots='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/padded_hours.pkl'
dir_inputs_geohshs='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/padded_hs5.pkl'
dir_inputs_subs_days='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/padded_subs_days.pkl'

dir_inputs_tgt_pois='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/padded_tgt_reindex_poi.pkl'
dir_inputs_tgt_cats_id='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/padded_tgt_reindex_cat_id.pkl'
dir_inputs_tgt_days='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/padded_tgt_days.pkl'
dir_inputs_tgt_slots='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/padded_tgt_slots.pkl'
dir_inputs_tgt_geohshs='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/padded_tgt_hshs.pkl'
dir_cat_id_mapped_poi='/home/jovyan/datasets/tsmc_nyc_6_distinguish_cat_poi/cat_id_mapped_poi.pkl'

dir_us_negs='/home/jovyan/datasets/tsmc_nyc_11_negative_sample/tsmc_nyc_11_negative_sample.ipynb'



with open(dir_inputs_pois,'rb') as f:
    inputs_pois=pickle.load(f)
with open(dir_inputs_cats_id,'rb') as f:
    inputs_cats_id=pickle.load(f)
with open(dir_inputs_days,'rb') as f:
    inputs_days=pickle.load(f)
with open(dir_inputs_slots,'rb') as f:
    inputs_slots=pickle.load(f)
with open(dir_inputs_geohshs,'rb') as f:
    inputs_geohshs=pickle.load(f)
with open(dir_inputs_subs_days,'rb') as f:
    inputs_subs_days_mask=pickle.load(f)
    
with open(dir_inputs_tgt_pois,'rb') as f:
    inputs_tgt_pois=pickle.load(f)
with open(dir_inputs_tgt_cats_id,'rb') as f:
    inputs_tgt_cats_id=pickle.load(f)
with open(dir_inputs_tgt_days,'rb') as f:
    inputs_tgt_days=pickle.load(f)
with open(dir_inputs_tgt_slots,'rb') as f:
    inputs_tgt_slots=pickle.load(f)
with open(dir_inputs_tgt_geohshs,'rb') as f:
    inputs_tgt_geohshs=pickle.load(f)
with open(dir_cat_id_mapped_poi,'rb') as f:
    cat_id_mapped_poi=pickle.load(f)
with open('/home/jovyan/datasets/tsmc_nyc_11_negative_sample/us_negs.pkl','rb') as f:
    (us_negs_pois,us_negs_cats_id)=pickle.load(f)

#1.2写dataloader
def Process_preference_model(padded_pois,padded_cats_id,padded_days,padded_slots,
                            padded_geohshs,padded_subs_days,
                             padded_tgt_pois,padded_tgt_cats_id,padded_tgt_days,
                             padded_tgt_slots,padded_tgt_geohshs,
                             us_negs_pois,us_negs_cats_id,batch_size):
    class Pre_Dataset(Dataset):
        def __init__(self,padded_pois,padded_cats_id,padded_days,padded_slots,
                     padded_geohshs,padded_subs_days_mask,inputs_tgt_pois,inputs_tgt_cats_id,inputs_tgt_days,
                     inputs_tgt_slots,inputs_tgt_geohshs,
                     us_negs_pois,us_negs_cats_id):
            self.padded_pois=padded_pois
            self.padded_cats_id=padded_cats_id
            self.padded_days=padded_days
            self.padded_slots=padded_slots
            self.padded_geohshs=padded_geohshs
            self.padded_subs_days_mask=padded_subs_days_mask
            self.padded_tgt_pois=padded_tgt_pois
            self.padded_tgt_cats_id=padded_tgt_cats_id
            self.padded_tgt_days=padded_tgt_days
            self.padded_tgt_slots=padded_tgt_slots
            self.padded_tgt_geohshs=padded_tgt_geohshs
            self.us_negs_pois=us_negs_pois
            self.us_negs_cats_id=us_negs_cats_id
        def __len__(self):
            return len(self.padded_pois)
        def __getitem__(self,index):
            pois=self.padded_pois[index]
            cats_id=self.padded_cats_id[index]
            days=self.padded_days[index]
            slots=self.padded_slots[index]
            geohshs=self.padded_geohshs[index]
            subs_days_mask=self.padded_subs_days_mask[index]
            tgt_pois=self.padded_tgt_pois[index]
            tgt_cats_id=self.padded_tgt_cats_id[index]
            tgt_days=self.padded_tgt_days[index]
            tgt_slots=self.padded_tgt_slots[index]
            tgt_geohshs=self.padded_tgt_geohshs[index]
            negs_pois=self.us_negs_pois[index]
            negs_cats_id=self.us_negs_cats_id[index]
            return pois,cats_id,days,slots,geohshs,subs_days_mask,tgt_pois,tgt_cats_id,tgt_days,tgt_slots,tgt_geohshs,negs_pois,negs_cats_id
    #创建数据集实例
    all_datasets=Pre_Dataset(padded_pois,padded_cats_id,padded_days,padded_slots,
                            padded_geohshs,padded_subs_days,
                             padded_tgt_pois,padded_tgt_cats_id,padded_tgt_days,padded_tgt_slots,padded_tgt_geohshs,us_negs_pois,us_negs_cats_id)
    dataloader=DataLoader(all_datasets,batch_size,shuffle=True)
    return dataloader

pre_all_data=Process_preference_model(inputs_pois,inputs_cats_id,inputs_days,inputs_slots,
                                      inputs_geohshs,inputs_subs_days_mask,
                                      inputs_tgt_pois,inputs_tgt_cats_id,inputs_tgt_days,inputs_tgt_slots,inputs_tgt_geohshs,
                                      us_negs_pois,us_negs_cats_id,
                                      batch_size)

#1.2.2生成reindex pois和reindex_cat_1st_id
all_pois=torch.arange(size_poi)#.cuda()
all_cat_1st_id=torch.arange(size_cat_id)#.cuda()

#1.3传入数据
model=Preference_Model(embed_size_poi,embed_size_cat_id,
                       embed_size_days,embed_size_slot,embed_size_geohsh,
                       hidden_size,
                       size_cat_id,size_poi,size_days,size_slots,size_geohsh,
                       num_layers,size_tgt,mlp_units,
                       seq_len)
# model=model.cuda()

loss_function=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(),lr)

for epoch in range(epoch_size):
    model.train()
    epoch_loss=0.0
    for batch_step,(batch_pois,batch_cats_id,batch_days,batch_slots,batch_geohshs,padded_subs_days_mask,
                    batch_tgt_pois,batch_tgt_cats_id,tgt_days,tgt_slots,tgt_hshs,batch_negs_poi,batch_negs_cat_id) in enumerate(pre_all_data):
        len_batch=len(batch_pois)
        model.zero_grad()
        loss=0.0
        
        # batch_pois=batch_pois.cuda()
        # batch_cats_id=batch_pois.cuda()
        # batch_days=batch_days.cuda()
        # batch_slots=batch_slots.cuda()
        # batch_geohshs=batch_geohshs.cuda()
        # padded_subs_days_mask=padded_subs_days_mask.cuda()
        # tgt_days=tgt_days.cuda()
        # tgt_slots=tgt_slots.cuda()
        # tgt_hshs=tgt_slots.cuda()

        outputs=model(batch_pois,batch_cats_id,batch_days,batch_slots,
                      batch_geohshs,padded_subs_days_mask,
                    #   batch_tgt_pois,batch_tgt_cats_id,
                      tgt_days,tgt_slots,tgt_hshs,
                      batch_negs_poi,batch_negs_cat_id,len_batch)#(bs,sq)
        #计算loss
        outputs=outputs.unsqueeze(-1)
        labels=torch.ones_like(outputs)
        loss+=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        epoch_loss+=float(loss)
        # print(loss)
        # break#只执行一个循环
    print(epoch_loss)