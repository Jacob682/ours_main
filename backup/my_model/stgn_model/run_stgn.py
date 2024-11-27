from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence
import pandas as pd
import numpy as np
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import warnings
warnings.filterwarnings('ignore')
import math
import sys
sys.path.append('./stgn')
from stgn import STGN_Model
sys.path.append('code/my_code/my_model/utils')
from utils import to_cuda,accuracy,EarlyStop
def exe_time(func):
    def new_func(*args, **args2):
        name = func.__name__
        start = datetime.now()
        print("-- {%s} start: @ %ss" % (name, start))
        back = func(*args, **args2)
        end = datetime.now()
        print("-- {%s} start: @ %ss" % (name, start))
        print("-- {%s} end:   @ %ss" % (name, end))
        total = (end - start).total_seconds()
        print("-- {%s} total: @ %.3fs = %.3fh" % (name, total, total / 3600.0))
        return back
    return new_func

def Process_stgn_data(dir_inputs,batch_size):
    def pad_dict(dic_xy):
        for key,value in dic_xy.items():
            # if not isinstance(value[0],list):#一维列表
            #     dic_xy[key]=torch.tensor(value,dtype=torch.long)
            if not isinstance(value[0][0],list):#二维列表
                flat_list=[item for sublist in value for item in sublist]
                dic_xy[key]=torch.tensor(flat_list,dtype=torch.long)
            else:#三维列表
                flat_list=[torch.tensor(item) for sublist in value for item in sublist]
                dic_xy[key]=pad_sequence(flat_list,batch_first=True,padding_value=0)
        return dic_xy
    
    with open(dir_inputs,'rb') as f:
        dit_data=pickle.load(f)
    tra_dit_x,tra_dit_y,tes_dit_x,tes_dit_y=dit_data['tra_dit_x'],dit_data['tra_dit_y'],dit_data['tes_dit_x'],dit_data['tes_dit_y']
    tra_x_keys=['tra_xy_uids_extend_timesteps','tra_x_rec_poi_ints','tra_x_rec_cat_ints','tra_x_rec_delta_ts','tra_x_rec_delta_ds','tra_x_rec_hsh5s','tra_x_rec_lens']#取用到的数据
    tra_y_keys=['tra_y_poi_ints']
    tes_x_keys=['tes_xy_uids_extend_timesteps','tes_x_rec_poi_ints','tes_x_rec_cat_ints','tes_x_rec_delta_ts','tes_x_rec_delta_ds','tes_x_rec_hsh5s','tes_x_rec_lens']
    tes_y_keys=['tes_y_poi_ints']
    tra_xy={key[4:]:tra_dit_x[key] for key in tra_x_keys}#不保留key中的'tra_',为后续用同一个dataset，dataloader
    tra_xy.update({key[4:]:tra_dit_y[key] for key in tra_y_keys})


    tes_xy={key[4:]:tes_dit_x[key] for key in tes_x_keys}
    tes_xy.update({key[4:]:tes_dit_y[key] for key in tes_y_keys})


    tra_xy=pad_dict(tra_xy)#(1078*timestep,rec_num),padded
    tes_xy=pad_dict(tes_xy)

    class STGN_Dataset(Dataset):#将xy一起传入，打乱
        def __init__(self,xy) -> None:
            self.users=xy['xy_uids_extend_timesteps']#(user*timestep)
            self.x_rec_poi_ints=xy['x_rec_poi_ints']#(user*timestep^,rec_num)
            self.x_rec_cat_ints=xy['x_rec_cat_ints']#(user*timestep^,rec_num)
            self.x_rec_delta_ts=xy['x_rec_delta_ts']#(user*timestep^,rec_num)
            self.x_rec_delta_ds=xy['x_rec_delta_ds']
            self.x_rec_hsh5s=xy['x_rec_hsh5s']
            self.x_rec_lens=xy['x_rec_lens']
            self.y_poi_ints=xy['y_poi_ints']#(user*timestep)
            pass
        def __len__(self):
            return len(self.users)
        def __getitem__(self, index):
            u=self.users[index]
            x_rec_poi_int=self.x_rec_poi_ints[index]
            x_rec_cat_int=self.x_rec_cat_ints[index]
            x_rec_delta_t=self.x_rec_delta_ts[index]
            x_rec_delta_d=self.x_rec_delta_ds[index]
            x_rec_hsh5=self.x_rec_hsh5s[index]
            x_rec_len=self.x_rec_lens[index]#某时间步的rec长度，如第一个时间步则长度为1，第30个时间步，长度为20
            y_poi_int=self.y_poi_ints[index]
            return u,x_rec_poi_int,x_rec_cat_int,x_rec_delta_t,x_rec_delta_d,x_rec_hsh5,x_rec_len,y_poi_int
    tra_stgn_dataset=STGN_Dataset(tra_xy)
    tes_stgn_dataset=STGN_Dataset(tes_xy)
    tra_stgn_dataloader=DataLoader(tra_stgn_dataset,batch_size,shuffle=True)
    tes_stgn_dataloader=DataLoader(tes_stgn_dataset,batch_size,shuffle=True)
    return tra_stgn_dataloader,tes_stgn_dataloader
     

def run_stgn(batch_size,epoch_size,num_rec,num_layers,num_x,dropout,lr,embd_size,hidden_size,mlp_units,
             dir_inputs,patience,delta):
    '''
    embd_size:list [embed_size_user,embed_size_poi,embed_size_cat,embed_size_hsh5]
    mlp_units:list
    '''
    tra_inputs,tes_inputs=Process_stgn_data(dir_inputs,batch_size)
    model=STGN_Model(hidden_size,num_x,embd_size,num_rec,mlp_units)
    model=model.cuda()
    loss_function=nn.CrossEntropyLoss(reduction='mean')
    optimizer=torch.optim.Adam(model.parameters(),lr)
    eary_stop=EarlyStop(patience,delta)
    for epoch in range(patience):
        model.train()
        start=datetime.now()
        train_epoch_loss=0.0
        acc_1,acc_5,mrr_1,mrr_5=0,0,0,0
        data_len=len(tra_inputs)
        len_tra=80727
        for batch_step,inputs in enumerate(tra_inputs):#inputs是包含所有特征（tensor)的列表
            model.zero_grad()
            inputs=to_cuda(inputs)
            stgn_inputs=inputs[:-1]
            y=inputs[-1]#tensor (b)
            outputs=model(stgn_inputs)#mlp_h_t（b,3906)

            _,sorted_indices=torch.sort(outputs,dim=-1,descending=True)
            b_avg_loss=loss_function(outputs,y)
            b_avg_loss.backward()
            optimizer.step()
            train_epoch_loss+=b_avg_loss#sum batch_avg

            acc_1+=accuracy(sorted_indices,y,1,0,0)
            acc_5+=accuracy(sorted_indices,y,5,0,0)
        

        end=datetime.now()
        total=(end-start).total_seconds()
        print('-- total:@ %.3fs=%.2fh'%(total,total/3600))
        print('tra:',
              'epoch:[{}/{}]\t'.format(epoch,eary_stop.counter),
              'loss:{:.4f}\t'.format(train_epoch_loss),
              'acc@1:{:.4f}\t'.format(acc_1/len_tra),
              'acc@5:{:.4f}\t'.format(acc_5/len_tra))

        
        if epoch%1==0:
            model=model.eval()
            test_epoch_loss=0.0
            len_tes=1078
            acc_1,acc_5=0,0
            for batch_step,inputs in enumerate(tes_inputs):
                inputs=to_cuda(inputs)
                stgn_inputs=inputs[:-1]
                y=inputs[-1]
                outputs=model(stgn_inputs)

                _,sorted_indices=torch.sort(outputs,dim=-1,descending=True)
                tes_b_avg_loss=loss_function(outputs,y)
                test_epoch_loss+=tes_b_avg_loss
                acc_1+=accuracy(sorted_indices,y,1,0,0)
                acc_5+=accuracy(sorted_indices,y,5,0,0)
            eary_stop(test_epoch_loss)
        print(
            'val:',
            'epoch:[{}/{}]\t'.format(epoch,eary_stop.counter),
            'tes_loss:{:4f}\t'.format(test_epoch_loss),
            'acc@1:{:.4f}\t'.format(acc_1/len_tes),
            'acc@5:{:.4f}\t'.format(acc_5/len_tes)
        )
        if eary_stop.early_stop:
            print('Early Stop.')

@exe_time
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
    return num_x

def main_cal():
    
    dir_inputs = 'code/my_code/dataset/data/CAL/data_process_cal.pkl'
    embed_size=[128,350,13,16]#(user,poi,cat,hsh5)
    num_x = [114,170,73,21]

    return dir_inputs,embed_size,num_x
@exe_time
def main():
    dir_inputs='/home/liuqiuyu/code/my_code/dataset/data_process.pkl'
    num_rec=20
    # dir_inputs='/home/liuqiuyu/code/my_code/dataset/data_process_70.pkl'
    # num_rec=70
    # dir_inputs='/home/liuqiuyu/code/my_code/dataset/data_process_30.pkl'
    # num_rec=30
    # dir_inputs='/home/liuqiuyu/code/my_code/dataset/data_process_10.pkl'
    # num_rec=10

    batch_size=50
    epoch_size=40
    
    num_layers=1
    dropout=0
    lr=0.0001
    embed_size=[50,350,120,16]#(user,poi,cat,hsh5)
    hidden_size=512
    mlp_units=[2048,1024,3906]
    num_x=[1078,3906,284,95]#个数

    patience=150
    delta=0

    dir_inputs,embed_size,num_x = main_cal()

    run_stgn(batch_size,epoch_size,num_rec,num_layers,num_x,dropout,
             lr,embed_size,hidden_size,mlp_units,dir_inputs,
             patience,delta)
    # tra_stgn_dataloader,tes_stgn_dataloader=Process_stgn_data(dir_inputs,batch_size)
    pass

if __name__=='__main__':
    main()