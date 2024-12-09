import pickle
import torch
import torch .nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch .nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from datetime import datetime
import pickle
import os
import logging
logging.basicConfig(filename='traning.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['CUDA_VISIBLE_DEVICES']='1'
import warnings
warnings.filterwarnings('ignore')
from utils.utils import accuracy, MRR, to_cuda, exe_time
from model.pref_austgn import Pref_Austgn


class pref_austgn_Dataset(Dataset):
        def __init__(self, xy):
            # austgn
            self.user = xy['x_unit']
            self.x_poi_rec = xy['x_poi_rec']
            self.x_cat_rec = xy['x_cat_rec']
            self.x_day_rec = xy['x_day_rec']
            self.x_hour_rec = xy['x_hour_rec']
            self.x_geo_rec = xy['x_geo_rec']
            self.x_delta_t_rec = xy['x_delta_t_rec']
            self.x_delta_d_rec = xy['x_delta_d_rec']
            self.x_poi_rec_q = xy['x_poi_rec_q']
            self.x_cat_rec_q = xy['x_cat_rec_q']
            self.x_day_rec_q = xy['x_day_rec_q']
            self.x_hour_rec_q = xy['x_hour_rec_q']
            self.x_geo_rec_q = xy['x_geo_rec_q']
            self.x_length_rec = xy['x_length_rec']

            #pref
            self.x_poi = xy['x_poi']
            self.x_cat = xy['x_cat']
            self.x_day = xy['x_day']
            self.x_hour = xy['x_hour']
            self.x_geo = xy['x_geo']
            self.x_sub_day = xy['x_sub_day']
            self.x_sub_hour = xy['x_sub_hour']
            self.x_sub_geo = xy['x_sub_geo']

            #y
            self.y_poi = xy['y_poi']
            self.y_cat = xy['y_cat']
            self.y_day = xy['y_day']
            self.y_hour = xy['y_hour']
            self.y_geo = xy['y_geo']

            #neg
            self.neg_poi = xy['neg_poi']
            self.neg_cat = xy['neg_cat']
        
        def __len__(self):
            return len(self.user)
        def __getitem__(self, index):
            #austgn [0: 14]
            u = self.user[index] #0
            x_poi_rec = self.x_poi_rec[index] #1
            x_cat_rec = self.x_cat_rec[index] #2
            x_day_rec = self.x_day_rec[index] #3
            x_hour_rec = self.x_hour_rec[index] #4
            x_geo_rec = self.x_geo_rec[index] #5
            x_delta_t_rec = self.x_delta_t_rec[index] #6
            x_delta_d_rec = self.x_delta_d_rec[index] #7
            x_poi_rec_q = self.x_poi_rec_q[index] #8
            x_cat_rec_q = self.x_cat_rec_q[index] #9
            x_day_rec_q = self.x_day_rec_q[index] #10
            x_hour_rec_q = self.x_hour_rec_q[index] #11
            x_geo_rec_q = self.x_geo_rec_q[index] #12
            x_length_rec = self.x_length_rec[index] # 13
            #pref [14: 22]
            x_poi = self.x_poi[index] #14 | 0
            x_cat = self.x_cat[index] #15 | 1
            x_day = self.x_day[index] #16 | 2
            x_hour = self.x_hour[index] #17 | 3
            x_geo = self.x_geo[index] #18 | 4
            x_sub_day = self.x_sub_day[index] #19 | 5
            x_sub_hour = self.x_sub_hour[index] #20 | 6
            x_sub_geo = self.x_sub_geo[index] #21 | 7
            #y [22: 27]
            y_poi = self.y_poi[index] #22 | 0
            y_cat = self.y_cat[index] #23 | 1
            y_day = self.y_day[index] #24 | 2
            y_hour = self.y_hour[index] #25 | 3
            y_geo = self.y_geo[index] #26 | 4
            #neg [27:29]
            neg_poi = self.neg_poi[index] #27 | 0
            neg_cat = self.neg_cat[index] #28 | 1
            
            return u, x_poi_rec, x_cat_rec, x_day_rec, x_hour_rec, x_geo_rec, x_delta_t_rec, x_delta_d_rec,\
            x_poi_rec_q, x_cat_rec_q, x_day_rec_q, x_hour_rec_q, x_geo_rec_q, x_length_rec,\
            x_poi, x_cat, x_day, x_hour, x_geo, x_sub_day, x_sub_hour, x_sub_geo,\
            y_poi, y_cat, y_day, y_hour, y_geo, \
            neg_poi, neg_cat
    

@exe_time
def Process_data(dir_data, batch_size):
    def fun_pad_dit(xy_dit):
        for key, value in xy_dit.items():
            if isinstance(value[0], list):
                if not isinstance(value[0][0],list):# 二维列表，比如y-->全部打平成一维
                    flat_list = [item for sublist in value for item in sublist]
                    xy_dit[key] = torch.tensor(flat_list, dtype=torch.long)
                else:#三维，四维列表，将第一第二维打平,并和所有用户中最长的时间步对齐
                    flat_list = [torch.tensor(item) for sublist in value for item in sublist]
                    xy_dit[key] = pad_sequence(flat_list, batch_first=True, padding_value=0)
            else:
                xy_dit[key] = torch.tensor(value, dtype=torch.int)
        return xy_dit
    
    with open(dir_data, 'rb') as f:
        data = pickle.load(f)
        '''
        [tra_x_feas_us, tra_y_feas_us, tra_neg_feas_us]
        '''
    x_keys = ['x_unit', 'x_poi', 'x_cat', 'x_day', 'x_hour', 'x_geo',\
              'x_sub_day', 'x_sub_hour', 'x_sub_geo',\
              'x_poi_rec', 'x_cat_rec', 'x_day_rec', 'x_hour_rec', 'x_geo_rec',\
              'x_delta_t_rec', 'x_delta_d_rec',\
              'x_poi_rec_q', 'x_cat_rec_q', 'x_day_rec_q', 'x_hour_rec_q', 'x_geo_rec_q', \
              'x_length_rec'] #(1078,时间步，每个包含之前的时间步,sub_num)
    
    y_keys = ['y_poi', 'y_cat', 'y_day', 'y_hour', 'y_geo'] #(1078,57)
    neg_keys = ['neg_poi', 'neg_cat'] #(1078,57)
    x_feas_us, y_feas_us, neg_feas_us = data #[num_feas, len_user, len_ts]
    
    xy_dit ={k:v for k, v in zip(x_keys, x_feas_us)}
    xy_dit.update({k:v for k, v in zip(y_keys, y_feas_us)})
    xy_dit.update({k:v for k,v in zip(neg_keys, neg_feas_us)})
    
    tra_tes_xy = fun_pad_dit(xy_dit)

    
    pref_austgn_dataset = pref_austgn_Dataset(tra_tes_xy)
    pref_austgn_dataloader = DataLoader(pref_austgn_dataset, batch_size, shuffle=True)
    return pref_austgn_dataloader

def fun_save_data(dir_data, batch_size, outputfile):
    if os.path.exists(outputfile[-1]):
        return
    else:
        for dir_data, dir_output in zip(dir_data, outputfile):
            tra_inputs = Process_data(dir_data, batch_size)
            with open(dir_output, 'wb') as f:
                pickle.dump(tra_inputs, f)
    return 


@exe_time
def run_pref_austgn(batch_size, num_epoch, delta, num_layers, num_x, lr, weight_decay, \
                    pref_embs, stgn_embs, mlp_units, dir_inputs_lists, dir_output_lists, dir_input_tst, dir_output_tst, len_tra, len_tes, num_neg, num_head, num_rec):
    
    model = Pref_Austgn(num_x, pref_embs, stgn_embs, mlp_units, num_layers, num_head, num_rec)
    model = model.cuda()
    loss_function = nn.BCEWithLogitsLoss(reduce = 'mean')
    # loss_function = nn.BCELoss(reduce='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    scaler = GradScaler()
    
    # 在训练之前加载数据，避免每个epoch都花费2min生成数据,只执行一次
    # 调用封装的prepare_data函数
    fun_save_data(dir_inputs_lists, batch_size, dir_output_lists) # 不需要返回，保存到文件
    
    for epoch in range(num_epoch):
        train_start = datetime.now()

        model.train()
        train_epoch_loss = 0.0
        tra_acc_1, tra_acc_5 = 0, 0
        mrr = 0
        for dir_data in dir_output_lists:
            # tra_inputs = Process_data(dir_data, batch_size)
            # 加载已保存的数据
            try:
                logging.info('load data from %s'%dir_data)
                tra_inputs = pickle.load(open(dir_data, 'rb'))
                logging.info('load data from %s success'%dir_data)
            except Exception as e:
                logging.error('load data from %s failed'%dir_data)
                logging.error(e)
                raise e


            for batch, batch_inputs in enumerate(tra_inputs):
                batch_inputs = to_cuda(batch_inputs)
                optimizer.zero_grad()
                austgn_inputs = batch_inputs[:14]
                pref_inputs = batch_inputs[14:22]
                y_inputs = batch_inputs[22:27]
                neg_inputs = batch_inputs[27:]
                model_inputs = [austgn_inputs, pref_inputs, y_inputs, neg_inputs]
                with autocast():
                    outputs, shuffle_indices = model(model_inputs, num_neg[0]) #(batch_size, neg_num, 1)
                    outputs = torch.squeeze(outputs, dim=-1) #(batch_size, neg_num)
                    
                    '''
                    outputs: (batch_size, neg_num)
                    shuffle_indices: (bs,neg_num),bs的list，neg_num的乱序index。index为具体数字
                    '''
                    # 计算整体的loss
                    ## 对输出结果排序
                    _, sorted_indice = torch.sort(outputs, dim=-1, descending=True)

                    #找到正样本的位置,生成bool掩码（true/false）
                    pos_position = (shuffle_indices == 0) # (batch_size, neg_num)将样本排序为0的位置设置为1，标识为正样本位置，因为正样本放在第一个位置，index=0
                    #用bool掩码得到正样本位置，并标识为1
                    y_shuffle = torch.zeros_like(shuffle_indices)
                    y_shuffle[pos_position] = 1 # 将正样本的位置设置为1
                    y_shuffle[~pos_position] = 0 # (batch_size, neg_num)
                    # 计算loss
                    b_avg_loss = loss_function(outputs, (y_shuffle.to(torch.float32)).cuda())
                    # 反向传播
                scaler.scale(b_avg_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_epoch_loss = train_epoch_loss + b_avg_loss

                
                
                # 计算评价指标
                y_shuffle_1d = torch.nonzero(y_shuffle==1)[:,1]
                tra_acc_1 = tra_acc_1 + accuracy(sorted_indice, y_shuffle_1d, 1)
                tra_acc_5 = tra_acc_5 + accuracy(sorted_indice, y_shuffle_1d, 5)
        train_end = datetime.now()
        total = (train_end - train_start).total_seconds()
        print('--total:@ %.3fs==%.3fmin'%(total, total/60))
        print('tra:',
              'epoch:[{}/{}]\t'.format(epoch, num_epoch),
              'loss:{:.4f}\t'.format(train_epoch_loss),
              'acc@1:{:.4f}\t'.format(tra_acc_1/len_tra),
              'acc@5:{:.4f}\t'.format(tra_acc_5/len_tra),)
        
        # 在test之前，存test数据
        fun_save_data(dir_input_tst, batch_size, dir_output_tst)
        tst_inputs = pickle.load(open(dir_output_tst[0], 'rb'))

        # if epoch % 1 == 0:
        #     with torch.no_grad():
        #         model.eval()
        #         test_epoch_loss = 0.0
        #         acc_1, acc_5, acc_10, acc_15 = 0, 0, 0, 0
        #         mrr = 0

        #         for batch_step, batch_inputs in enumerate(tst_inputs):
        #             batch_inputs = to_cuda(batch_inputs)
        #             austgn_inputs = batch_inputs[:14]
        #             pref_inputs = batch_inputs[14:22]
        #             y_inputs = batch_inputs[22:27]
        #             neg_inputs = batch_inputs[27:]
        #             model_inputs = [austgn_inputs, pref_inputs, y_inputs, neg_inputs]
        #             with autocast():
        #                 outputs, shuffle_indices = model(model_inputs, num_neg[1])
        #                 outputs = torch.squeeze(outputs, dim=-1)
        #                 _, sorted_indice = torch.sort(outputs, dim=-1, descending=True)
        #                 pos_position = (shuffle_indices == 0)
        #                 y_shuffle = torch.zeros_like(shuffle_indices)
        #                 y_shuffle[pos_position] = 1
        #                 y_shuffle[~pos_position] = 0
        #                 b_avg_loss = loss_function(outputs, (y_shuffle.to(torch.float32)).cuda())
        #             test_epoch_loss = test_epoch_loss + b_avg_loss
        #             y_shuffle_1d = torch.nonzero(y_shuffle==1)[:,1]
        #             acc_1 = acc_1 + accuracy(sorted_indice, y_shuffle_1d, 1)
        #             acc_5 = acc_5 + accuracy(sorted_indice, y_shuffle_1d, 5)
        #             acc_10 = acc_10 + accuracy(sorted_indice, y_shuffle_1d, 10)
        #             acc_15 = acc_15 + accuracy(sorted_indice, y_shuffle_1d, 15)

        #         print('tst:',
        #           'epoch:[{}/{}]\t'.format(epoch, num_epoch),
        #           'tst_loss:{:.4f}\t'.format(test_epoch_loss),
        #           'acc@1:{:.4f}\t'.format(acc_1/len_tes),
        #           'acc@5:{:.4f}\t'.format(acc_5/len_tes),
        #           'acc@10:{:.4f}\t'.format(acc_10/len_tes)
        #         )
            
@exe_time
def main_nyc():
    dir_input_lists = ['/data/liuqiuyu/POI_OURS_DATA/data/model_use/tra0.pkl',\
                       '/data/liuqiuyu/POI_OURS_DATA/data/model_use/tra1.pkl']
    dir_input_tst = ['/data/liuqiuyu/POI_OURS_DATA/data/model_use/tes.pkl'] # 做成列表为了共用fun_save_data

    dir_output_lists = ['/data/liuqiuyu/POI_OURS_DATA/data/model_use/tra1_40_prepared.pkl',\
                        '/data/liuqiuyu/POI_OURS_DATA/data/model_use/tra0_40_prepared.pkl']
    dir_output_tst = ['/data/liuqiuyu/POI_OURS_DATA/data/model_use/tes_40_prepared.pkl']
    num_negs = [3905, 3906] #一个是tra的neg(需要+1，补正样本），一个是tes的neg
    len_tra, len_tes = 82883, 1078
    batch_size, num_epoch = 40, 100
    delta = 1
    num_layers = 1
    num_head = 1
    num_rec = 20
    lr = 0.00001
    weight_decay = 0
    pref_embs = [256, 64, 32, 8, 16, 32]
    stgn_embs = [256, 128, 150, 120, 12, 16, 32] # (hidden,user,poi,cat,month/hour,hsh5)
    pref_mlp_units = [512, 128, 256] # 此处pref_mlp_units[-1]和stgn.hidden_size相同，为了inner_attn维度对齐
    mlp_units = (pref_mlp_units, [1024, 512, 1])
    num_x = [1079, 3906, 285, 96, 8, 25, 20] #hsh[0-95]共96个

    run_pref_austgn(batch_size, num_epoch, delta, num_layers, num_x, lr, weight_decay, \
                    pref_embs, stgn_embs, mlp_units, dir_input_lists, dir_output_lists, dir_input_tst, dir_output_tst, len_tra, len_tes, num_negs, num_head, num_rec)
if __name__ =='__main__':
     main_nyc()