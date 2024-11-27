import pickle
import datetime
import numpy as np
import pandas as pd
import os
import warnings
import torch
from torch.utils.data import DataLoader,Dataset
warnings.filterwarnings('ignore')

def Process_model_data(users,rec_cat,rec_poi,rec_delta_t,rec_delta_d,y_poi,rec_x_lens,x_second_importance,
                  x_poi,x_cat,x_day,x_hour,x_hs5,x_days_subs,
                  negs_poi,negs_cat,
                  y_cat,y_day,y_hour,y_hs5,
                  train_x_lens_mask,batch_size):

    class Process_Dataset(Dataset):
        def __init__(self,users,rec_cat,rec_poi,rec_delta_t,rec_delta_d,y_poi,rec_x_lens,x_second_importance,
                  x_poi,x_cat,x_day,x_hour,x_hs5,x_days_subs,
                  negs_poi,negs_cat,
                  y_cat,y_day,y_hour,y_hs5,train_x_lens_mask):
            self.users=users
            self.rec_cat=rec_cat
            self.rec_poi=rec_poi
            self.rec_delta_t=rec_delta_t
            self.rec_delta_d=rec_delta_d
            self.y_poi=y_poi
            self.rec_x_lens=rec_x_lens
            self.x_second_importance=x_second_importance
            self.x_poi=x_poi
            self.x_cat=x_cat
            self.x_day=x_day
            self.x_hour=x_hour
            self.x_hs5=x_hs5
            self.x_days_subs=x_days_subs
            self.negs_poi=negs_poi
            self.negs_cat=negs_cat
            self.y_cat=y_cat
            self.y_day=y_day
            self.y_hour=y_hour
            self.y_hs5=y_hs5
            self.train_x_lens_mask=train_x_lens_mask
        def __len__(self):
            return len(self.users)#长度相同为所有用户个数
        def __getitem__(self,index):
            u=users[index]
            r_cat=rec_cat[index]
            r_poi=rec_poi[index]
            r_delta_t=rec_delta_t[index]
            r_delta_d=rec_delta_d[index]
            y_p=y_poi[index]
            r_x_lens=rec_x_lens[index]
            second_importance=x_second_importance[index]
            x_p=x_poi[index]
            x_c=x_cat[index]
            x_d=x_day[index]
            x_h=x_hour[index]
            x_hs=x_hs5[index]
            x_days_sub=x_days_subs[index]
            negs_p=negs_poi[index]
            negs_c=negs_cat[index]
            y_c=y_cat[index]
            y_d=y_day[index]
            y_h=y_hour[index]
            y_hs=y_hs5[index]
            len_mask=train_x_lens_mask[index]
            return u,r_cat,r_poi,r_delta_t,r_delta_d,y_p,r_x_lens,second_importance,x_p,x_c,x_d,x_h,x_hs,x_days_sub,negs_p,negs_c,y_c,y_d,y_h,y_hs,len_mask
    model_data=Process_Dataset(users,rec_cat,rec_poi,rec_delta_t,rec_delta_d,y_poi,rec_x_lens,x_second_importance,
                  x_poi,x_cat,x_day,x_hour,x_hs5,x_days_subs,
                  negs_poi,negs_cat,
                  y_cat,y_day,y_hour,y_hs5,train_x_lens_mask)
    dataloader=DataLoader(model_data,batch_size,shuffle=True)
    return dataloader