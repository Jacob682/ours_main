import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
from datetime import datetime

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

def accuracy(indices,batch_y,k,count=0,delta_dist=0):
    '''
    计算一个batch内一个窗口的accuracy
    indices:(batch_size,poi_size)
    batch_y:(batch_size),对indices的ground truth
    count:训练/测试集中用户数量
    delta_dist:无用
    return:一个batch中命中的总量
    '''
    hit=0
    for i in range(indices.size(0)):#一个用户
        sort=indices[i]
        print("batch y a :",batch_y)
        print("sort[k] :",sort[:k])
        if batch_y[i].long() in sort[:k]:
            hit+=1
    return hit

def MRR(indices,batch_y):
    m=0
    for i in range(indices.size(0)):
        sort=indices[i]
        if batch_y[i].long() in sort:
            m+=1/(torch.where(sort==batch_y[i])[0]+1)
    return m


class EarlyStop:
    def __init__(self,patience=150,delta=1):
        self.patience=patience
        self.delta=delta
        self.counter=0
        self.best_score=None
        self.early_stop=False
    def __call__(self,val_loss,tra_acc_1):
        if self.best_score is None:
            self.best_score=val_loss
        elif val_loss>self.best_score-self.delta:
            self.counter+=1
            if self.counter>=self.patience:
                self.early_stop=True
        elif tra_acc_1==1:
            self.early_stop=True
        else:
            self.best_score=val_loss
            self.counter+=1



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