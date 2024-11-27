import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

import pandas as pd
import numpy as np
import datetime
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
class RecPreferenceModel(nn.Module):
    def __init__(self,embed_size_user,embed_size_cat,embed_size_poi,hidden_size,
                size_user,size_cat,size_poi,
                num_layers,num_rec,dropout=0.0):
        super(RecPreferenceModel,self).__init__()
        
        self.num_rec=num_rec
        self.embed_user=nn.Embedding(size_user,embed_size_user)
        self.embed_cat=nn.Embedding(size_cat,embed_size_cat)
        self.embed_poi=nn.Embedding(size_poi,embed_size_poi)
        
        lstm_size=embed_size_user+embed_size_cat+embed_size_poi+2#delta_t，delta_d维度
        self.lstm=nn.LSTM(lstm_size,hidden_size,num_layers,batch_first=True,dropout=dropout)
        self.fc=nn.Linear(hidden_size,size_poi)
    
    def forward(self,inputs_user,inputs_cat,inputs_poi,inputs_delta_t,inputs_delta_d,rec_lens):
        '''
        inputs_cat、inputs_delta_t,inputs_delta_d是padding之后的
        rec_lens:每个时间步rec的真实长度，滑动窗口tensor
        '''
        inputs_user=self.embed_user(inputs_user).unsqueeze(1).repeat(1,inputs_cat.size(1),1)#(n_user,embs)->(n_user,seq_len,embs)
        inputs_user=inputs_user.unsqueeze(2).expand(-1,-1,self.num_rec,-1)#(bs,sq,rec,embs)

        inputs_cat=self.embed_cat(inputs_cat)#(n_user,seq_len,rec,embed_cat)
        inputs_poi=self.embed_poi(inputs_poi)#(n_user,seq_len,rec,embed_poi)
        inputs_delta_t=inputs_delta_t.unsqueeze(-1)#(n_users,seq_len,rec,1)
        inputs_delta_d=inputs_delta_d.unsqueeze(-1)#(n_users,seq_len,rec,1)
        inputs=torch.cat((inputs_user,inputs_cat,inputs_poi,inputs_delta_t,inputs_delta_d),-1) 
        valid_inputs=inputs.view(inputs.size()[0]*inputs.size()[1],inputs.size()[2],inputs.size()[3])#(bs*seq,rec,embs)
        rec_lens=rec_lens.cpu().to(torch.int64)
        valid_inputs=pack_padded_sequence(valid_inputs,rec_lens.view(rec_lens.size()[0]*rec_lens.size()[1]),batch_first=True,enforce_sorted=False)
        output,(h,c)=self.lstm(valid_inputs)#取hidden，(bs*seqlen,hidden)
        c=c.view(rec_lens.size()[0],rec_lens.size()[1],-1)#(bs,sq,hidden)
        # out=self.fc(c)#注掉之后相当于直接输出隐藏层,没注释掉之后就是还是tgt个输出,最后用second_importance在相应位置上做权重
        return c



def Process_train_rec(users,train_rec_x_cat_int,train_rec_x_poi_int,
                train_rec_x_delta_t,train_rec_x_delta_d,
                train_y_poi_int,train_rec_x_lens,train_x_second_importance,
                batch_size):
    '''
    users:(n_user,1)
    padded_reindex_rec_cat_id:(n_user,20,embed_cat_size)
    padded_rec_delta_t/d:(n_user,20),不需要embed，最后在模型里concat
    '''
    class RecDataset(Dataset):
        def __init__(self,users,train_rec_x_cat_int,train_rec_x_poi_int,
                train_rec_x_delta_t,train_rec_x_delta_d,
                train_y_poi_int,train_rec_x_lens,train_x_second_importance):
            self.users=users
            self.train_rec_x_poi_int=train_rec_x_poi_int
            self.train_rec_x_cat_int=train_rec_x_cat_int
            self.train_rec_x_delta_t=train_rec_x_delta_t
            self.train_rec_x_delta_d=train_rec_x_delta_d
            self.train_y_poi_int=train_y_poi_int
            self.train_rec_x_lens=train_rec_x_lens
            self.train_x_second_importance=train_x_second_importance
        def __len__(self):
            return len(self.users)#长度相同为所有用户个数
        def __getitem__(self,index):#得到每一个用户的数据，非每个时间步，即每个用户的打卡数据
            user=self.users[index]
            poi_id=self.train_rec_x_poi_int[index]
            cat_id=self.train_rec_x_cat_int[index]
            delta_t=self.train_rec_x_delta_t[index]
            delta_d=self.train_rec_x_delta_d[index]
            y_poi_int=self.train_y_poi_int[index]
            rec_x_lens=self.train_rec_x_lens[index]
            second_importance=self.train_x_second_importance[index]
            return user,poi_id,cat_id,delta_t,delta_d,y_poi_int,rec_x_lens,second_importance
    #创建数据集实例
    rec_dataset=RecDataset(users,train_rec_x_cat_int,train_rec_x_poi_int,
                train_rec_x_delta_t,train_rec_x_delta_d,
                train_y_poi_int,train_rec_x_lens,train_x_second_importance)
    #创建dataloader
    dataloader=DataLoader(rec_dataset,batch_size,shuffle=True)
    return dataloader