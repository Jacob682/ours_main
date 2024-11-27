import pickle
import torch
torch.cuda.empty_cache
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import warnings
warnings.filterwarnings('ignore')

import sys
from utils.utils import accuracy,MRR,EarlyStop,to_cuda
from both_model.preference_sequential_model import Preference_Stgn
# from load_data_psm import load_data_main_nyc,load_data_main_tky 

from torch.nn.parallel import DataParallel
import gc
gc.collect()
torch.cuda.empty_cache()

def exec_time(func):
    def new_func(*args,**args2):
        name=func.__name__
        start=datetime.now()
        print('--{%s}start:@%ss'%(name,start))
        back=func(*args,**args2)
        end=datetime.now()
        print('--{%s} start:@%ss'%(name,start))
        print('--{%s} end:  @%ss'%(name,end))
        total=(end-start).total_seconds()
        print('--{%s}total: @%.3fs=%.3fh'%(name,total,total/3600.0))
        return back
    return new_func

@exec_time
def Process_prefstgn_data(dir_inputs,batch_size):
    def pad_dict(dic_xy):
        for key,value in dic_xy.items():
            if not isinstance(value[0][0],list):#二维列表
                flat_list=[item for sublist in value for item in sublist]
                dic_xy[key]=torch.tensor(flat_list,dtype=torch.long)
            else:#三维列表
                flat_list=[torch.tensor(item) for sublist in value for item in sublist]
                dic_xy[key]=pad_sequence(flat_list,batch_first=True,padding_value=0)
        return dic_xy
    
    with open(dir_inputs,'rb') as f:
        dit_data=pickle.load(f)
    
    tra_dit_x,tra_dit_y,tes_dit_x,tes_dit_y,tra_dit_neg,tes_dit_neg=dit_data['tra_dit_x'],dit_data['tra_dit_y'],dit_data['tes_dit_x'],dit_data['tes_dit_y'],dit_data['tra_dit_neg'],dit_data['tes_dit_neg']
    tra_x_keys=['tra_xy_uids_extend_timesteps','tra_x_rec_poi_ints','tra_x_rec_cat_ints','tra_x_rec_delta_ts','tra_x_rec_delta_ds','tra_x_rec_hours','tra_x_rec_hsh5s','tra_x_rec_lens',
                'tra_x_poi_ints','tra_x_cat_ints','tra_x_days','tra_x_hours','tra_x_hsh5s','tra_x_sub_days','tra_x_sub_hours','tra_x_sub_hsh5s']#'tra_x_months',,'tra_x_sub_months','tra_x_sub_days','tra_x_sub_hours','tra_x_sub_hsh5s']#,'tra_x_second_importances']
    # tra_x_keys=['tra_xy_uids_extend_timesteps','tra_x_poi_ints','tra_x_cat_ints','tra_x_days','tra_x_hours','tra_x_hsh5s','tra_x_sub_days','tra_x_sub_hours','tra_x_sub_hsh5s']#'tra_x_months',,'tra_x_sub_months','tra_x_sub_days','tra_x_sub_hours','tra_x_sub_hsh5s']#,'tra_x_second_importances']
  
    tra_x_neg_keys=['tra_neg_pois','tra_neg_cats']#,'tra_neg_seconds']
    tra_y_keys=['tra_y_poi_ints','tra_y_cat_ints','tra_y_days','tra_y_hours','tra_y_hsh5s']#,'tra_y_second_ints','tra_y_months','tra_y_days','tra_y_hours','tra_y_hsh5s']
    tra_xy={key[4:]:tra_dit_x[key] for key in tra_x_keys}
    tra_xy.update({key[4:]:tra_dit_neg[key] for key in tra_x_neg_keys})
    tra_xy.update({key[4:]:tra_dit_y[key] for key in tra_y_keys})
    
    del dit_data
    tes_x_keys=['tes_xy_uids_extend_timesteps','tes_x_rec_poi_ints','tes_x_rec_cat_ints','tes_x_rec_delta_ts','tes_x_rec_delta_ds','tes_x_rec_hours','tes_x_rec_hsh5s','tes_x_rec_lens',
                'tes_x_poi_ints','tes_x_cat_ints','tes_x_days','tes_x_hours','tes_x_hsh5s','tes_x_sub_days','tes_x_sub_hours','tes_x_sub_hsh5s']#,'tes_x_months','tes_x_sub_months','tes_x_sub_days','tes_x_sub_hours','tes_x_sub_hsh5s','tes_x_second_importances']
    # tes_x_keys=['tes_xy_uids_extend_timesteps','tes_x_poi_ints','tes_x_cat_ints','tes_x_days','tes_x_hours','tes_x_hsh5s','tes_x_sub_days','tes_x_sub_hours','tes_x_sub_hsh5s']#,'tes_x_months','tes_x_sub_months','tes_x_sub_days','tes_x_sub_hours','tes_x_sub_hsh5s','tes_x_second_importances']
    tes_x_neg_keys=['tes_neg_pois','tes_neg_cats']#,'tes_neg_seconds']
    tes_y_keys=['tes_y_poi_ints','tes_y_cat_ints','tes_y_days','tes_y_hours','tes_y_hsh5s']#'tes_y_second_ints','tes_y_months','tes_y_days','tes_y_hours','tes_y_hsh5s']
    tes_xy={key[4:]:tes_dit_x[key] for key in tes_x_keys}
    tes_xy.update({key[4:]:tes_dit_neg[key] for key in tes_x_neg_keys})
    tes_xy.update({key[4:]:tes_dit_y[key] for key in tes_y_keys})
    
    tra_xy=pad_dict(tra_xy)
    tes_xy=pad_dict(tes_xy)
    class pref_stgn_Dataset(Dataset):
        def __init__(self,xy):
            #stgn
            self.users=xy['xy_uids_extend_timesteps']#(user)
            self.x_rec_poi_ints=xy['x_rec_poi_ints']#(user*timestep^,rec_num)
            self.x_rec_cat_ints=xy['x_rec_cat_ints']#(user*timestep^,rec_num)
            self.x_rec_delta_ts=xy['x_rec_delta_ts']#(user*timestep^,rec_num)
            self.x_rec_delta_ds=xy['x_rec_delta_ds']
            self.x_rec_hours=xy['x_rec_hours']
            self.x_rec_hsh5s=xy['x_rec_hsh5s']
            self.x_rec_lens=xy['x_rec_lens']
            self.y_poi_ints=xy['y_poi_ints']#(user*timestep)
            #pref
            self.xy=xy
            self.x_pois=xy['x_poi_ints']
            self.x_cats=xy['x_cat_ints']
            # self.x_months=xy['x_months']
            self.x_days=xy['x_days']
            self.x_hours=xy['x_hours']
            self.x_hsh5s=xy['x_hsh5s']
            self.x_sub_days=xy['x_sub_days']
            # self.x_sub_months=xy['x_sub_months']
            self.x_sub_hours=xy['x_sub_hours']
            self.x_sub_hsh5s=xy['x_sub_hsh5s']
            # self.x_second_importances=xy['x_second_importances']

            self.y_cat_ints=xy['y_cat_ints']#正样本
            # self.y_months=xy['y_months']
            self.y_days=xy['y_days']
            # self.y_second_ints=xy['y_second_ints']
            self.y_hours=xy['y_hours']
            self.y_hsh5s=xy['y_hsh5s']

            self.neg_pois=self.xy['neg_pois']
            self.neg_cats=self.xy['neg_cats']
            # self.neg_seconds=self.xy['neg_seconds']
        def __len__(self):
            return len(self.users)
        def __getitem__(self,index):
            u=self.users[index]
            x_rec_poi_int=self.x_rec_poi_ints[index]
            x_rec_cat_int=self.x_rec_cat_ints[index]
            x_rec_delta_t=self.x_rec_delta_ts[index]
            x_rec_delta_d=self.x_rec_delta_ds[index]
            x_rec_hours=self.x_rec_hours[index]
            x_rec_hsh5=self.x_rec_hsh5s[index]
            x_rec_len=self.x_rec_lens[index]#某时间步的rec长度，如第一个时间步则长度为1，第30个时间步，长度为20
            y_poi_int=self.y_poi_ints[index]

            x_poi=self.x_pois[index]
            x_cat=self.x_cats[index]
            # x_month=self.x_months[index]
            x_day=self.x_days[index]
            x_hour=self.x_hours[index]
            x_hsh5=self.x_hsh5s[index]
            x_sub_day=self.x_sub_days[index]
            # x_sub_month=self.x_sub_months[index]
            x_sub_hour=self.x_sub_hours[index]
            x_sub_hsh5=self.x_sub_hsh5s[index]
            # x_second_importance=self.x_second_importances[index]
            
            y_cat=self.y_cat_ints[index]
            # y_second=self.y_second_ints[index]
            # y_month=self.y_months[index]
            y_day=self.y_days[index]
            y_hour=self.y_hours[index]
            y_hsh5=self.y_hsh5s[index]

            x_neg_poi=self.neg_pois[index]
            x_neg_cat=self.neg_cats[index]

            print(x_poi.shape)
            #1     0       1           2               3              4          5           6        7
            #2          8    9     10     11     12     13        14          15        16        17
            #3        18     19      20      21
            #          22 
            # x_neg_second=self.neg_seconds[index]
            return u,x_rec_poi_int,x_rec_cat_int,x_rec_delta_t,x_rec_delta_d,x_rec_hours,x_rec_hsh5,x_rec_len,\
                    x_poi,x_cat,x_day,x_hour,x_hsh5,x_sub_day,x_sub_hour,x_sub_hsh5,x_neg_poi,x_neg_cat,\
                    y_cat,y_day,y_hour,y_hsh5,\
                    y_poi_int#,x_second_importance
        
            #1     0          1            2            3             4           5           6        7
            #2        8     9     10     11     12      13       14        15        16         17       18       19        20
            #3        21     22       23     24     25    26
            #4       27           28  
            # return u,x_rec_poi_int,x_rec_cat_int,x_rec_delta_t,x_rec_delta_d,x_rec_hours,x_rec_hsh5,x_rec_len,\
            #         x_poi,x_cat,x_month,x_day,x_hour,x_hsh5,x_sub_month,x_sub_day,x_sub_hour,x_sub_hsh5,x_neg_poi,x_neg_cat,x_neg_second,\
            #         y_cat,y_second,y_month,y_day,y_hour,y_hsh5,\
            #         y_poi_int#,x_second_importance

  

    tra_prefstgn_dataset=pref_stgn_Dataset(tra_xy)
    tes_prefstgn_dataset=pref_stgn_Dataset(tes_xy)
    tra_prefstgn_dataloader=DataLoader(tra_prefstgn_dataset,batch_size,shuffle=True)
    tes_prefstgn_dataloader=DataLoader(tes_prefstgn_dataset,batch_size,shuffle=True)
    # for i,e in enumerate(tes_prefstgn_dataloader):
    #     print(i)
    return tra_prefstgn_dataloader,tes_prefstgn_dataloader




def run_prefstgn(batch_size,patience,delta,num_layers,num_x,dropout,lr,weight_decay,
                 pref_embs,stgn_embs,mlp_units,dir_inputs,len_tra,len_tes,num_negs,head_num):
    '''
    dropout:无用
    '''
    '''tra_inputs=[pref_inputs,stgn_inputs,second_importance]
        tra:
        pref_inputs=[x_poi,x_cat,x_month,x_day,x_hour,x_hsh5,x_sub_month,x_sub_day,x_sub_hour,x_sub_hsh5,x_neg_poi,x_neg_cat,x_neg_second,\
                    y_cat,y_second,y_month,y_day,y_hour,y_hsh5,\
                    y_poi_int]
        stgn_inputs=[u,x_rec_poi_int,x_rec_cat_int,x_rec_delta_t,x_rec_delta_d,x_rec_months/x_rec_hours,x_rec_hsh5,x_rec_len]
        num_negs[tra_neg21,test_neg3906]
    '''
    tra_inputs,tes_inputs=Process_prefstgn_data(dir_inputs,batch_size)
    model=Preference_Stgn(num_x,pref_embs,stgn_embs,mlp_units,dropout,num_layers,head_num)
    # model=DataParallel(model)
    model=model.cuda()
    loss_function=nn.BCELoss(reduce='mean')
    optimizer=torch.optim.Adam(model.parameters(),lr,weight_decay=weight_decay)
    early_stop=EarlyStop(patience,delta)
    for epoch in range(patience):
    # if not early_stop.early_stop:
        torch.cuda.empty_cache()
        model.train()
        start=datetime.now()
        train_epoch_loss=0.0
        tra_acc_1,acc_5=0,0
        mrr=0
        for batch_step,inputs in enumerate(tra_inputs):
            inputs=to_cuda(inputs)
            model.zero_grad()
            # second_importance=inputs[-1]
            y=inputs[-1]
            # print("inputs",inputs)
            # print("y",y)
            # pref_inputs=inputs[7:-1]
            pref_inputs=inputs[8:]
            stgn_inputs=inputs[:8]
            model_inputs=[pref_inputs,stgn_inputs]#,second_importance]
            
            outputs,shuffle_indices=model(model_inputs,num_negs[0])#outputs(bs,neg_num+1）
            _,sorted_indice=torch.sort(outputs,dim=-1,descending=True)

            zero_position=(shuffle_indices==0)
            shuffle_indices_y=torch.zeros_like(shuffle_indices)#二元target
            shuffle_indices_y[zero_position]=1
            shuffle_indices_y[~zero_position]=0
            shuffle_indices_1d=torch.nonzero(shuffle_indices_y==1)[:,1]
            # print("shuffle_indices_y",shuffle_indices_y)
            # shuffle_indices_y=F.one_hot(shuffle_indices_1d)
            # bceloss
            b_avg_loss=loss_function(outputs,(shuffle_indices_y.to(torch.float32)).cuda())
            #pairwise loss
            # b_avg_loss=loss_function(outputs,torch.ones_like(outputs).to(torch.float32).cuda())
            
            b_avg_loss.backward()
            optimizer.step()
            train_epoch_loss+=b_avg_loss

            tra_acc_1+=accuracy(sorted_indice,shuffle_indices_1d,1)#多元，1d标签
            acc_5+=accuracy(sorted_indice,shuffle_indices_1d,5)
            mrr+=MRR(sorted_indice,shuffle_indices_1d)

        end=datetime.now()
        total=(end-start).total_seconds()
        print('-- total:@ %.3fs=%.2fh'%(total,total/3600))
        print('tra:',
              'epoch:[{}/{}]\t'.format(epoch,early_stop.counter),
              'loss:{:.4f}\t'.format(train_epoch_loss),
              'acc@1:{:.4f}\t'.format(tra_acc_1/len_tra),
              'acc@5:{:.4f}\t'.format(acc_5/len_tra),
              'mrr:{:.4f}\t'.format(mrr[0]/len_tra))#取[0]是因为MRR中where函数，返回了tensor。tensor不支持.format
        if epoch%1==0:
            torch.cuda.empty_cache()
            with torch.no_grad():
                model.eval()
                test_epoch_loss=0
                acc_1,acc_5,acc_10,acc_15=0,0,0,0
                mrr=0
                for batch_step,inputs in enumerate(tes_inputs):
                    inputs=to_cuda(inputs)
                    # second_importance=inputs[-1]
                    y=inputs[-1]
                    # pref_inputs=inputs[7:-1]
                    pref_inputs=inputs[8:]
                    stgn_inputs=inputs[:8]
                    model_inputs=[pref_inputs,stgn_inputs]#,second_importance]
                    outputs,shuffle_indices=model(model_inputs,num_negs[1])
                    _,sorted_indice=torch.sort(outputs,dim=-1,descending=True)

                    zero_position=(shuffle_indices==0)#因为正样本拼接在第一个，下标为0，找到0的位置则得到正样本位置（弃用poi_int,用位置来标识poi)
                    shuffle_indices_y=torch.zeros_like(shuffle_indices)#二元target
                    shuffle_indices_y[zero_position]=1#确认在y中，真值的位置，该位置取1
                    shuffle_indices_y[~zero_position]=0
                    shuffle_indices_1d=torch.nonzero(shuffle_indices_y==1)[:,1]#得到下标矩阵（user，poi_num）真实y的位置
                    b_avg_loss=loss_function(outputs,(shuffle_indices_y.to(torch.float32)).cuda())
                    #pairwise_loss
                    # b_avg_loss=loss_function(outputs,torch.ones_like(outputs).to(torch.float32).cuda())

                    test_epoch_loss+=b_avg_loss
                    acc_1+=accuracy(sorted_indice,shuffle_indices_1d,1)
                    acc_5+=accuracy(sorted_indice,shuffle_indices_1d,5)
                    acc_10+=accuracy(sorted_indice,shuffle_indices_1d,10)
                    acc_15+=accuracy(sorted_indice,shuffle_indices_1d,15)
                    mrr+=MRR(sorted_indice,shuffle_indices_1d)
                early_stop(test_epoch_loss,tra_acc_1)
            print(
                'val:',
                'epoch:[{}/{}]\t'.format(epoch,early_stop.counter),
                'tes_loss:{:4f}\t'.format(test_epoch_loss),
                'acc@1:{:.4f}\t'.format(acc_1/len_tes),
                'acc@5:{:.4f}\t'.format(acc_5/len_tes),
                'acc@10:{:.4f}\t'.format(acc_10/len_tes),
                'acc@15:{:.4f}\t'.format(acc_15/len_tes),
                'mrr:{:.4f}\t'.format(mrr[0]/len_tes)
            )
            if early_stop.early_stop:
                    print('Early Stop.')


@exec_time
def main_nyc():
    dir_inputs='/home/liuqiuyu/code/my_code/dataset/data_process.pkl'
    num_negs=[3906,3906]#neg_num跟随load_data变,nyc,neg_num取负采样+1
    len_tra,len_tes=80727,1078

    batch_size=10
    patience=500
    delta=1

    num_layers=1
    head_num=1
    dropout=[0.1,0.1,0.1]#(target-att dropout,hiera-attn dropout,mlp dropout
    lr=0.0001
    weight_decay=0
    pref_embs=[128,64,32,8,16,32]#(hidden,poi,cat,day,hour,hsh5)
    stgn_embs=[512,128,350,120,13,16]#(hidden,user,poi,cat,month/hour,hsh5)
    mlp_units=[1024,512,1]
                                        #   0        1       2     3      4       5      6             
    num_x=[1078,3906,284,95,20,8,25]#[num_ueser,num_poi,num_cat,num_hsh,num_rec,num_day,num_hours],hsh[0-95],共96个，但该集中娶不到95
    
    run_prefstgn(batch_size,patience,delta,num_layers,num_x,dropout,lr,weight_decay,
                              pref_embs,stgn_embs,mlp_units,dir_inputs,len_tra,len_tes,num_negs,head_num)
    pass



@exec_time
def main_sin():
    dir_inputs='code/my_code/dataset/data/SIN/data_process_sin.pkl'
    num_negs=[3959,3959]
    len_tra,len_tes=153445,1193 # 155831

    batch_size=20
    patience=500
    delta=1

    num_layers=1
    head_num=4
    dropout=[0.1,0.1,0.1]
    lr=0.0001
    weight_decay=0
    pref_embs=[128,64,32,8,16,32]#(hidden,poi,cat,month,day,hour,hsh5)
    stgn_embs=[512,128,350,120,13,16]#(hidden,user,poi,month,cat,hsh5)
    mlp_units=[1024,512,1]
    num_x=[3678,3959,257,21,20,8,25]#[num_user,num_poi, num_cat,num_hsh5,num_rec,num_months,num_day,num_hours]
                                     #114,170,73,21,num_rec,12,7,24
    run_prefstgn(batch_size,patience,delta,num_layers,num_x,dropout,lr,weight_decay,
                              pref_embs,stgn_embs,mlp_units,dir_inputs,len_tra,len_tes,num_negs,head_num)
    pass

                    
if __name__=='__main__':
    # load_data_main_nyc()
    main_nyc()