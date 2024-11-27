import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import pickle
import os
import datetime
from pandas import Timestamp

##1.1定义dataloader,已经padding之后再给dataloader，
##最后放入模型,模型里面embed
def Process_rec(users,padded_reindex_rec_cat_id,
                padded_rec_delta_t,padded_rec_delta_d,
                tgt,
                batch_size):
    '''
    users:(n_user,1)
    padded_reindex_rec_cat_id:(n_user,20,embed_cat_size)
    padded_rec_delta_t/d:(n_user,20),不需要embed，最后在模型里concat
    '''
    class RecDataset(Dataset):
        def __init__(self,users,padded_reindex_rec_cat_id,padded_rec_delta_t,padded_rec_delta_d,tgt):
            self.users=users
            self.padded_reindex_rec_cat_id=padded_reindex_rec_cat_id
            self.padded_rec_delta_t=padded_rec_delta_t
            self.padded_rec_delta_d=padded_rec_delta_d
            self.tgt=tgt
        def __len__(self):
            return len(self.users)#长度相同为所有用户个数
        def __getitem__(self,index):#得到每一个用户的数据，非每个时间步，即每个用户的打卡数据
            user=self.users[index]
            cat_id=self.padded_reindex_rec_cat_id[index]
            delta_t=self.padded_rec_delta_t[index]
            delta_d=self.padded_rec_delta_d[index]
            tgt_item=self.tgt[index]
            return user,cat_id,delta_t,delta_d,tgt_item
    #创建数据集实例
    rec_dataset=RecDataset(users,padded_reindex_rec_cat_id,padded_rec_delta_t,padded_rec_delta_d,tgt)
    #创建dataloader
    dataloader=DataLoader(rec_dataset,batch_size,shuffle=True)
    return dataloader