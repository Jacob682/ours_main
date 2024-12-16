import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import time
from git import Repo
SAVE_FOLDER_PATH = None

def divide_no_nan(x,y,nan_value=0.0):
            mask=y==0
            y=torch.where(mask,torch.ones_like(y),y)
            result=x/y
            result=torch.where(mask,torch.tensor(nan_value),result)
            return result

def to_cuda(variables):
    cuda_list=[]
    for v in variables:
        cuda_list.append(torch.tensor(v).cuda())
    return cuda_list


def compute_Log_Loss(loss_fun,y_pred,y_true,mask):
    '''
    loss_fun:损失函数,交叉熵损失(bce)
    y_pred:正负样本的概率，(bs,sq,21)
    y_true:tgt,(bs,seq,21)#表示打乱后正样本的index
    mask:每个用户时间步的真实长度,(bs,sq)
    '''
    loss_batch=loss_fun(y_pred,y_true)#(bs,sq,21)
    loss_batch=torch.sum(loss_batch,dim=-1)
    loss_batch*=mask
    loss_batch=torch.mean(loss_batch)
    return loss_batch

def label_Generate(outputs_shape):
    labels=torch.zeros(outputs_shape)
    labels[:,:,0:1]=1
    return labels

def top_K_Precision(prob,y,k):
    '''
    prob:概率(batch,seq,21)
    y:int,下标(bs,sq,21)onehot
    return:平均每个user每个时间步的评价指标
    '''
    _,topk_indice=prob.topk(k,dim=-1)#取前k个最大值，prob不需要排序
    y_index=torch.argmax(y,dim=-1)#(bs,sq)
    y_expand=y_index.unsqueeze(-1).expand_as(topk_indice)#(bs,sq,k)
    correct_topk=topk_indice.eq(y_expand)#(bs,sq,k),当有预测正确的值时，某位置取1
    topk_acc=correct_topk.sum().float()/y_index.numel()
    return topk_acc

#mrr
def mrr(prob,y,k):
    '''
    prob:概率(batch,seq,21)
    y:int,下标(bs,sq,21)onehot
    return:平均每个user每个时间步的评价指标
    '''
    _,topk_indice=prob.topk(k,dim=-1)#(bs,sq,k)
    y_index=torch.argmax(y,dim=-1)#(bs,sq)
    y_expand=y_index.unsqueeze(-1).expand_as(topk_indice)#(bs,sq,k)
    correct_topk=topk_indice.eq(y_expand)#(bs,sq,k),当有预测正确的值时，该位置取1
    correct_index=correct_topk.nonzero()#(num,3)
    ranks=correct_index[:,-1]+1#(num),正确预测个数，取值正确预测的下标
    rranks=torch.reciprocal(ranks)
    mrrk=torch.sum(rranks)/y_index.numel()
    return mrrk

def accuracy(indices, batch_y, k, count=0, delta_dist=0):
    '''
    计算一个batch内一个窗口的accuracy
    indices:(batch_size,poi_size), 已排序
    batch_y:(batch_size),对indices的ground truth
    count:训练/测试集中用户数量
    delta_dist:无用
    return:一个batch中命中的总量
    '''
    hit=0
    for i in range(indices.size(0)):# 一个用户
        sort=indices[i]
        if batch_y[i].long() in sort[:k]:
            hit = hit + 1
    return hit

def MRR(indices,batch_y):
    m=0
    for i in range(indices.size(0)):
        sort=indices[i]
        if batch_y[i].long() in sort:
            m = m + 1/(torch.where(sort==batch_y[i])[0]+1)
    return m


def exe_time(func):
    def new_func(*args, **args2):
        name = func.__name__
        start = datetime.now()
        back = func(*args, **args2)
        end = datetime.now()
        print("-- {%s} start: @ %ss" % (name, start))
        print("-- {%s} end:   @ %ss" % (name, end))
        total = (end - start).total_seconds()
        print("-- {%s} total: @ %.3fs = %.3fh = %.3fmin" % (name, total, total / 3600.0, total / 60.0))
        return back
    return new_func

class MLP_BN(nn.Module):
    def __init__(self, mlp_units, num_fea, ipt_size, rate=0.1, acti=torch.relu):
        '''
        num_fea: 特征数目，如7/24/96
        '''
        super(MLP_BN, self).__init__()
        self.dropout = nn.Dropout(rate) 
        self.bn1 = nn.BatchNorm1d(num_fea, eps=1e-6)
        self.bn2 = nn.BatchNorm1d(num_fea, eps=1e-6)

        self.dense1 = nn.Linear(ipt_size, mlp_units[0])
        self.dense2 = nn.Linear(mlp_units[0], mlp_units[1])
        self.acti = acti
    def forward(self, x):
        '''
        x:[bs, num_fea, emb]
        要在num_fea上做bn
        '''
        x = self.bn1(x) #(bs, num_fea, ipt_size)
        x = self.dropout(x)
        x = self.dense1(x) #(bs, num_fea, mlp_unit[0])
        x = self.acti(x)
        
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.acti(x)

        return x
    
class MLP_BN4d(nn.Module):
    def __init__(self, mlp_units, num_fea, ipt_size, rate=0.1, acti=torch.relu):
        '''
        mlp_size:输入的维度
        '''
        super(MLP_BN4d, self).__init__()
        self.num_fea = num_fea
        self.ipt_size = ipt_size
        self.dropout = nn.Dropout(rate) 
        self.bn1 = nn.BatchNorm1d(num_fea, eps=1e-6)
        self.bn2 = nn.BatchNorm1d(num_fea, eps=1e-6)
        self.dense1 = nn.Linear(ipt_size, mlp_units[0])
        self.dense2 = nn.Linear(mlp_units[0], mlp_units[1])
        self.acti = acti
    def forward(self, x):
        '''
        x:(bs, neg_num, key_num, emb)
        '''
        batch_size = x.size(0)
        neg_num = x.size(1)
        key_num = x.size(2)

        x = x.view(batch_size * neg_num, key_num, -1)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.acti(x)
        
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.acti(x)

        x = x.view(batch_size, neg_num, key_num, -1)

        return x


class MLP_LN(nn.Module):
    def __init__(self,mlp_units, mlp_size, acti=torch.relu, rate=0.1):
        super().__init__()
        '''
        mlp_units:list,由x映射到到mlp_units[0]->mlp_units[1]->mlp_units[2]
        mlp_size:int,输入维度
        '''
        self.dropout=nn.Dropout(rate)
        
        # self.layernorm0 = nn.LayerNorm(mlp_size, eps=1e-6)
        self.layernorm0 = nn.LayerNorm(mlp_units[0], eps=1e-6)
        self.layernorm1 = nn.LayerNorm(mlp_units[1], eps=1e-6)

        self.dense1 = nn.Linear(mlp_size,mlp_units[0])
        self.dense2 = nn.Linear(mlp_units[0],mlp_units[1])
        self.dense_score = nn.Linear(mlp_units[1],mlp_units[2])
        self.acti0 = acti
        self.acti1 = F.sigmoid
    def forward(self,x):
        '''
        x:[bs,sq,hidden_size+embs_size]#x是attn concat queries结果，attn之前对keys做了dense
        '''
        x = self.dropout(x)
        x = self.acti0(self.dense1(x))
        x = self.layernorm0(x)
        
        x = self.dropout(x)
        x = self.acti0(self.dense2(x))
        x = self.layernorm1(x)
        
        y = self.dense_score(x)
        y = self.acti1(y)#(bs,sq,1)
        
        return y
    
class MLP_LN_SIGMOID(nn.Module):
    def __init__(self,mlp_units, mlp_size, acti=torch.relu, rate=0.1):
        super().__init__()
        '''
        mlp_units:list,由x映射到到mlp_units[0]->mlp_units[1]->mlp_units[2]
        mlp_size:int,输入维度
        '''
        self.dropout=nn.Dropout(rate)
        
        self.layernorm0 = nn.LayerNorm(mlp_size, eps=1e-6)
        self.layernorm1 = nn.LayerNorm(mlp_units[0], eps=1e-6)
        self.layernorm2 = nn.LayerNorm(mlp_units[1], eps=1e-6)

        self.dense1 = nn.Linear(mlp_size,mlp_units[0])
        self.dense2 = nn.Linear(mlp_units[0],mlp_units[1])        
        self.dense_score = nn.Linear(mlp_units[1],mlp_units[2])
        self.acti0 = acti
        self.acti1 = torch.sigmoid
    def forward(self,x):
        '''
        x:[bs,sq,hidden_size+embs_size]#x是attn concat queries结果，attn之前对keys做了dense
        '''
        x = self.layernorm0(x)
        x = self.dropout(x)
        x = self.acti0(self.dense1(x))
        
        x = self.layernorm1(x)
        x = self.dropout(x)
        x = self.acti0(self.dense2(x))
        
        x = self.layernorm2(x)
        y = self.dense_score(x)
        y = self.acti1(y)#(bs,sq,1)
        return y

def save_checkpoint(model, optimizer, epoch, save_dir):
    global SAVE_FOLDER_PATH
    if SAVE_FOLDER_PATH is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        SAVE_FOLDER_PATH = os.path.join(save_dir, timestamp)
        os.makedirs(SAVE_FOLDER_PATH, exist_ok=True)

    checkpoint_path = os.path.join(SAVE_FOLDER_PATH, f'epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {checkpoint_path}")
    return model, optimizer, epoch

import os
from git import Repo

def get_current_branch(repo_path='/home/liuqiuyu/POI_OURS'):
    try:
        # 检查 .git 是文件还是文件夹
        git_path = os.path.join(repo_path, '.git')
        if os.path.isfile(git_path):
            # 如果 .git 是文件，读取实际的 gitdir 路径
            with open(git_path, 'r') as f:
                line = f.readline().strip()
                if line.startswith("gitdir:"):
                    git_dir = line.split(": ")[1]
                else:
                    raise ValueError(f"Invalid .git file format: {line}")
        elif os.path.isdir(git_path):
            git_dir = git_path
        else:
            raise FileNotFoundError(f"No .git found in {repo_path}")

        # 使用解析后的路径初始化 Repo
        repo = Repo(git_dir)
        return repo.active_branch.name
    except Exception as e:
        print('Error while getting branch:', e)
        return None
           