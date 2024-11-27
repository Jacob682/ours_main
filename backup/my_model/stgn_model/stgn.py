import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import math
warnings.filterwarnings('ignore')
import sys


class STGN_MLP(nn.Module):
    def __init__(self,mlp_units,mlp_size,acti=torch.relu,rate=0.1):
        super(STGN_MLP,self).__init__()
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
        return y
    
class STGN(nn.Module):
    def __init__(self,input_sz,hidden_sz):
        super().__init__()
        self.input_sz=input_sz
        self.hidden_sz=hidden_sz
        self.Wxi=nn.Parameter(torch.Tensor(input_sz,hidden_sz))
        self.Whi=nn.Parameter(torch.Tensor(hidden_sz,hidden_sz))
        self.bi=nn.Parameter(torch.Tensor(hidden_sz))

        #f_t
        self.Wxf=nn.Parameter(torch.Tensor(input_sz,hidden_sz))
        self.Whf=nn.Parameter(torch.Tensor(hidden_sz,hidden_sz))
        self.bf=nn.Parameter(torch.Tensor(hidden_sz))

        #c_t
        self.Wxc = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.Whc = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.bc = nn.Parameter(torch.Tensor(hidden_sz))

        #T1t
        self.Wxt1=nn.Parameter(torch.Tensor(input_sz,hidden_sz))
        self.Wt1=nn.Parameter(torch.Tensor(1,hidden_sz))
        self.bt1=nn.Parameter(torch.Tensor(hidden_sz))

        #T2t
        self.Wxt2=nn.Parameter(torch.Tensor(input_sz,hidden_sz))
        self.Wt2=nn.Parameter(torch.Tensor(1,hidden_sz))
        self.bt2=nn.Parameter(torch.Tensor(hidden_sz))

        #D1t
        self.Wxd1=nn.Parameter(torch.Tensor(input_sz,hidden_sz))
        self.Wd1=nn.Parameter(torch.Tensor(1,hidden_sz))
        self.bd1=nn.Parameter(torch.Tensor(hidden_sz))

        #D2t
        self.Wxd2=nn.Parameter(torch.Tensor(input_sz,hidden_sz))
        self.Wd2=nn.Parameter(torch.Tensor(1,hidden_sz))
        self.bd2=nn.Parameter(torch.Tensor(hidden_sz))

        #o_t
        self.Wxo = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.Who = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.Wto=nn.Parameter(torch.Tensor(1,hidden_sz))
        self.Wdo=nn.Parameter(torch.Tensor(1,hidden_sz))
        self.bo = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()
    def init_weights(self):
        stdv=1.0/math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)
    def forward(self,lstm_input,init_state=None):
        """
        lstm_input:(bs,sq,embs),最后一维第一位是delta_t,第二位是delta_d
        """
        batch_size,seq_len=lstm_input.size(0),lstm_input.size(1)
        hidden_seq=[]
        if init_state is None:
            h_t,c_t=(
                torch.zeros(batch_size,self.hidden_sz).to(lstm_input.device),\
                torch.zeros(batch_size,self.hidden_sz).to(lstm_input.device)
                )
        else:
            h_t,c_t=init_state
        
        for t in range(seq_len):
            Tt=lstm_input[:,t,0].unsqueeze(1)
            Dt=lstm_input[:,t,1].unsqueeze(1)
            xt=lstm_input[:,t,2:]

            #更新门组件及内部候选状态
            it=torch.sigmoid(xt@self.Wxi+h_t@self.Whi+self.bi)
            ft=torch.sigmoid(xt@self.Wxf+h_t@self.Whf+self.bf)
            j=torch.tanh(xt@self.Wxc+h_t@self.Whc+self.bc)
            T1t=torch.sigmoid(xt@self.Wxt1+torch.sigmoid(Tt@self.Wt1)+self.bt1)
            T2t=torch.sigmoid(xt@self.Wxt2+torch.sigmoid(Tt@self.Wt2)+self.bt2)
            D1t=torch.sigmoid(xt@self.Wxd1+torch.sigmoid(Dt@self.Wd1)+self.bd1)
            D2t=torch.sigmoid(xt@self.Wxd2+torch.sigmoid(Dt@self.Wd2)+self.bd2)
            c_hat=ft*c_t+it*T1t*D1t*j
            c_t=ft*c_t+it*T2t*D2t*j
            ot=torch.sigmoid(xt@self.Wxo+h_t@self.Who+Tt@self.Wto+Dt@self.Wdo+self.bo)
            h_t=ot*torch.tanh(c_hat)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq=torch.cat(hidden_seq,dim=0)#(seq,bs,emb) tensor
        hidden_seq=hidden_seq.transpose(0,1).contiguous()#(bs,seq,emb) tensor
        return hidden_seq,(h_t,c_t)#hidden_seq是所有时间步的隐藏向量，h_t是最后一个隐藏向量，c_t是当前时间步的状态



class STGN_Model(nn.Module):
    def __init__(self,hidden_sz,num_x,sz_embs,num_rec,mlp_units):
        '''
        num_x:列表，用于embed,[num_ueser,num_poi,num_cat,num_hsh,num_rec,num_day,num_hours],全int
                           0                1             2              3                    4     
        sz_embs:列表[embed_size_user,embed_size_poi,embed_size_cat,embed_size_month/hours,embed_size_hsh5]
        '''
        super().__init__()
        self.num_rec=num_rec

        self.embed_user=nn.Embedding(num_x[0],sz_embs[0])
        self.embed_poi=nn.Embedding(num_x[1],sz_embs[1])
        self.embed_cat=nn.Embedding(num_x[2],sz_embs[2])
        self.embed_hour=nn.Embedding(num_x[3],sz_embs[3])
        self.embed_hsh5=nn.Embedding(num_x[4],sz_embs[4])
        #组合2
        self.stgn_cat=STGN(sz_embs[0]+sum(sz_embs[2:]),hidden_sz)#t,d,user,hsh5,hour,cat
        self.stgn_loc=STGN(sum(sz_embs[:2])+sum(sz_embs[3:]),hidden_sz)#t,d,user,poi,hour,hsh5
        #组合1
        # self.stgn_cat=STGN(sum(sz_embs[:-1]),hidden_sz)#无loc
        # self.stgn_loc=STGN(sum(sz_embs)-sz_embs[2],hidden_sz)#无cat
        self.mlp=STGN_MLP(mlp_units,hidden_sz)
    def forward(self,input):
        user=self.embed_user(input[0])#（u*t,emb)
        user=(torch.unsqueeze(user,dim=1)).repeat(1,self.num_rec,1)
        poi=self.embed_poi(input[1])#（u*t,rec,emb)
        cat=self.embed_cat(input[2])
        t=torch.unsqueeze(input[3],dim=-1)
        d=torch.unsqueeze(input[4],dim=-1)
        hour=self.embed_hour(input[5])
        hsh5=self.embed_hsh5(input[6])

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
        return h_cat_t,h_loc_t
        

    