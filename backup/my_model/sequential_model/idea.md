1. 不将delta_t,delta_d embedding(√)
2. 使用stgn替换lstm
3. 使用poi输入代替cat输入


note:
1. 都用所有时间步做loss（√）
    也许想只用最后一个时间步做loss，但往后推测试集合的时候，也隔了好几个时间步
2. 使用stgn
3. 每个epoch训练测试集合
    1. 划分测试数据


# readme
数据流程

    1. cat_id,poi reindex
    2. padding，与此同时toTensor
    3. 放入dataloader，分batch（要求tensor）
    4. 送入模型(要求：tensor)，模型处理维度对齐等问题