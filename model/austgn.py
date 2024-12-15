import torch
import torch.nn as  nn
from utils.utils import MLP_LN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
class AUSTGN(nn.Module):
    def __init__(self, ipt_sz, hidden_sz, q_sz):
        super().__init__()
        self.ipt_sz = ipt_sz
        self.hidden_sz = hidden_sz
        self.q_sz = q_sz
        
        self.Wxi=nn.Parameter(torch.Tensor(ipt_sz,hidden_sz))
        self.Whi=nn.Parameter(torch.Tensor(hidden_sz,hidden_sz))
        self.bi=nn.Parameter(torch.Tensor(hidden_sz))

        #f_t
        self.Wxf=nn.Parameter(torch.Tensor(ipt_sz,hidden_sz))
        self.Whf=nn.Parameter(torch.Tensor(hidden_sz,hidden_sz))
        self.bf=nn.Parameter(torch.Tensor(hidden_sz))

        #c_t
        self.Wxc = nn.Parameter(torch.Tensor(ipt_sz, hidden_sz))
        self.Whc = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.bc = nn.Parameter(torch.Tensor(hidden_sz))

        #T1t
        self.Wxt1=nn.Parameter(torch.Tensor(ipt_sz,hidden_sz))
        self.Wt1=nn.Parameter(torch.Tensor(1,hidden_sz))
        self.bt1=nn.Parameter(torch.Tensor(hidden_sz))

        #T2t
        self.Wxt2=nn.Parameter(torch.Tensor(ipt_sz,hidden_sz))
        self.Wt2=nn.Parameter(torch.Tensor(1,hidden_sz))
        self.bt2=nn.Parameter(torch.Tensor(hidden_sz))

        #D1t
        self.Wxd1=nn.Parameter(torch.Tensor(ipt_sz,hidden_sz))
        self.Wd1=nn.Parameter(torch.Tensor(1,hidden_sz))
        self.bd1=nn.Parameter(torch.Tensor(hidden_sz))

        #D2t
        self.Wxd2=nn.Parameter(torch.Tensor(ipt_sz,hidden_sz))
        self.Wd2=nn.Parameter(torch.Tensor(1,hidden_sz))
        self.bd2=nn.Parameter(torch.Tensor(hidden_sz))

        #o_t
        self.Wxo = nn.Parameter(torch.Tensor(ipt_sz, hidden_sz))
        self.Who = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.Wto=nn.Parameter(torch.Tensor(1,hidden_sz))
        self.Wdo=nn.Parameter(torch.Tensor(1,hidden_sz))
        self.bo = nn.Parameter(torch.Tensor(hidden_sz))

        # attn
        self.linear_a = nn.Linear(q_sz, hidden_sz)
        self.attn = nn.MultiheadAttention(hidden_sz, num_heads=1, batch_first=True) # (q, k, v, ...)
        
        self.init_weights()

    def init_weights(self):
        stdv=1.0/math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)
        
    def forward(self, input, x_q, keys_length, init_state=None):
        '''
        input : (bs, 统一sq, embs)，最后一维第一位是delta_t,第二位是delta_d
        x_q: (bs, 统一sq，embs)
        keys_lengths: (bs, 1)做packed用
        init_state = (h_t, c_t) 不需要有at，因为at可以用0初始化的ht，ct生成

        统一sq为rec_num
        # 将序列取消padded，得到batch内真实的rec长度,避免虚假依赖
        # 计算batch每个rec内时间步的attn分数
        # 将attn分数传递到STGNCell中，参与最后输出的计算
        '''

        # 将张量做成PackedSequence类，放入模型中。模型中得到每个batch的时间步
        # keys_length = keys_length.squeeze(-1).to(torch.int64).cpu()
        # input_packed = pack_padded_sequence(input, keys_length, batch_first=True, enforce_sorted=False)
        # q_packed= pack_padded_sequence(x_q, keys_length, batch_first=True, enforce_sorted=False)

        # input, lengths = pad_packed_sequence(input_packed, batch_first=True) # 填充成长度相等的三维张量|lenght(batch),真实的长度
        # q_unpacked, _ = pad_packed_sequence(q_packed, batch_first=True) 
        batch_size, seq_len = input.size(0), input.size(1)
        hidden_seq = [] # 所有隐藏状态
         
        if init_state is None: # 初始化最初状态
            h_t, c_t= (
                torch.zeros(batch_size, self.hidden_sz, device=input.device), # h_t只包含当前时间步的
                torch.zeros(batch_size, self.hidden_sz, device=input.device)
            )
            at = 1 # 第一个时间步的attn分数
        else:
            h_t, c_t = init_state
            at = 1 # 暂时设置分数为1

        for t in range(seq_len): # 迭代每个时间步
            
            Tt=input[:,t,0].unsqueeze(1)
            Dt=input[:,t,1].unsqueeze(1)
            xt=input[:,t,2:]

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

            


class Sequential_Model(nn.Module):
    def __init__(self, stgn_embs, num_x, mlp_units, num_rec):
        '''
        stgn_embs = [256, 128, 150, 120, 12, 16, 32] | [(0hidden,1user,2poi,3cat,4day,5hour,6geo)]
        num_x:被embedding的个数
                    0user;1poi;2cat;3geo;4day;5hour;6num_rec
            num_x = [1079, 3906, 285, 96, 8, 25, 20]   
        '''
        super().__init__()
        self.num_rec = num_rec
        self.embed_user = nn.Embedding(num_x[0], stgn_embs[1])
        self.embed_poi = nn.Embedding(num_x[1], stgn_embs[2])
        self.embed_cat = nn.Embedding(num_x[2], stgn_embs[3])
        self.embed_geo = nn.Embedding(num_x[3], stgn_embs[6])
        self.embed_day = nn.Embedding(num_x[4], stgn_embs[4])
        self.embed_hour = nn.Embedding(num_x[5], stgn_embs[5])
        
        ipt_poi_sz = sum(stgn_embs[1:3]) + sum(stgn_embs[4:]) # 跳过cat，user+其他
        ipt_cat_sz = stgn_embs[1] + sum(stgn_embs[3:]) # 跳过poi，user+其他

        self.austgn_poi = AUSTGN(ipt_poi_sz, stgn_embs[0], ipt_poi_sz)
        self.austgn_cat = AUSTGN(ipt_cat_sz, stgn_embs[0], ipt_cat_sz)
        
        self.seq_linear_poi = nn.Linear(stgn_embs[0], stgn_embs[0])
        self.seq_linear_cat = nn.Linear(stgn_embs[0], stgn_embs[0])

    def forward(self, input):
        '''
        input = [0u, 1x_poi_rec, 2x_cat_rec, 3x_day_rec, 4x_hour_rec, 5x_geo_rec, 6x_delta_t_rec, 7x_delta_d_rec,\
            8x_poi_rec_q, 9x_cat_rec_q, 10x_day_rec_q, 11x_hour_rec_q, 12x_geo_rec_q, 13x_length_rec]
        '''

        user = self.embed_user(input[0]) #(batch_size, emb)
        user = (torch.unsqueeze(user, dim = 1).repeat(1, self.num_rec, 1))
        poi = self.embed_poi(input[1]) #(batch_size, rec_num) -> (batch_size, rec_num, embs)
        cat = self.embed_cat(input[2])
        day = self.embed_day(input[3])
        hour = self.embed_hour(input[4])
        geo = self.embed_geo(input[5])
        
        poi_q = self.embed_poi(input[8])
        cat_q = self.embed_cat(input[9])
        day_q = self.embed_day(input[10])
        hour_q = self.embed_hour(input[11])
        geo_q = self.embed_geo(input[12])

        t = torch.unsqueeze(input[6], dim = -1)
        d = torch.unsqueeze(input[7], dim = -1)
        
        keys_length = input[-1] # (batch_size, 1) 每个batch_size中样本实际的长度
        
        austgn_loc_input = torch.cat((t, d, user, day, hour, geo, poi), dim = -1)
        austgn_loc_q = torch.cat((user, poi_q, day_q, hour_q, geo_q), dim = -1) # (batch_size, unit_sq, embs)
        
        austgn_cat_input = torch.cat((t, d, user, day, hour, geo, cat), dim = -1) #(batch_size, unit_sq, embs+2)
        austgn_cat_q = torch.cat((user, cat_q, day_q, hour_q, geo_q), dim = -1) #(batch_size, unit_sq, embs)
        
        

        _, (h_poi_t, _) = self.austgn_poi(austgn_loc_input, austgn_loc_q, keys_length)
        _, (h_cat_t, _) = self.austgn_cat(austgn_cat_input, austgn_cat_q, keys_length)


        h_poi_out = self.seq_linear_poi(h_poi_t) #(batch_size, embs)
        h_cat_out = self.seq_linear_cat(h_cat_t)

        return (h_poi_out, h_cat_out)
