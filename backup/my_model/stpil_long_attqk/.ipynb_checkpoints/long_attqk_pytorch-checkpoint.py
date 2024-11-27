import torch
import torch.nn as nn

class STPIL_Long_attqk(STPIL_Basic):
    def __init__(self, usr_feas, train, test, long_short, num_inps, dim_feas, num_negs, num_topk,
                 mlp_units, transformer_params, alpha_lambda, dropout_rate):
        super(STPIL_Long_attqk, self).__init__(
            usr_feas, train, test, long_short, num_inps, dim_feas, num_negs, num_topk,
            mlp_units, transformer_params, alpha_lambda, dropout_rate)
        # trainable_variables from outer classes
        self.model_mlp = MLP(self.mlp_units, self.acti, self.rate_mlp)
        self.dense = nn.Linear(self.d_model, self.d_model)  # 调控长期兴趣的维度。
        self.model_att_qk_sft = BahdanauAttentionQK_softmax(self.d_model, self.rate_long)  # 多兴趣聚合

    def forward(self, tra_elements):
        loss = self.batch_model_tra(tra_elements)
        return loss
        
    def batch_model_tra(self, tra_elements):  # 变量都是batch形式,tra_elements
        # input
        # ===============================================================
        # input: idx
        [
            # (sz, 1), (sz, len+1), (sz, len+1, 20), (sz, len+1, 7/9), (sz, len+1, 5/5/5/5)
            u_idxs,
            tra_pois_idxs, tra_cats_idxs, tra_1sts_idxs, tra_days_idxs, tra_slts_idxs, tra_hs5s_idxs, tra_msks,
            tra_pois_negs_idxs, tra_cats_negs_idxs, tra_1sts_negs_idxs,
            tra_subs_days_idxs, tra_subs_1sts_idxs,
            tra_rect_timh_idxs, tra_rect_dist_idxs, tra_rect_timv_idxs, tra_rect_ctxt_idxs,
            tra_long, tra_short  # (sz, len+1, 7/9), (sz, len+1, 5)
        ] = tra_elements
        # input: emb    # 这些用于算loss
        # u_embs = self.emb_layer_usr(u_idxs)                   # (sz, 1, 128)
        tra_pois_embs = self.emb_layer_poi(tra_pois_idxs)  # (sz, len+1, 128)
        tra_cats_embs = self.emb_layer_cat(tra_cats_idxs)  # (sz, len+1, 64)
        # tra_1sts_embs = self.emb_layer_1st(tra_1sts_idxs)     # (sz, len+1, 16)
        tra_days_embs = self.emb_layer_day(tra_days_idxs)  # (sz, len+1, 16)
        tra_slts_embs = self.emb_layer_slt(tra_slts_idxs)  # (sz, len+1, 32)
        tra_hs5s_embs = self.emb_layer_hs5(tra_hs5s_idxs)  # (sz, len+1, 64)
        tra_pois_negs_embs = self.emb_layer_poi(tra_pois_negs_idxs)  # (sz, len+1, 10, 128)
        tra_cats_negs_embs = self.emb_layer_cat(tra_cats_negs_idxs)
        # tra_1sts_negs_embs = self.emb_layer_1st(tra_1sts_negs_idxs)
        # input: concatenate
        # ===============================================================
        batch_sz = tra_msks.size(0)
        tra_len_plus = tra_msks.size(1)
        # 用户的：shape=(sz, len+1, 21\len_q, 128)
        # tra_user_embs = u_embs.repeat(1, tra_len_plus, 1)   # (sz, len+1, 128)
        # tra_user_embs = tra_user_embs.unsqueeze(2).repeat(1, 1, 1+self.num_negs, 1)
        # 正样本：shape=(sz, len+1, 208)
        tra_posi_embs = torch.cat([tra_pois_embs, tra_cats_embs], dim=-1)  # , tra_1sts_embs
        # 负样本：shape=(sz, len+1, 10, 208)
        tra_nega_embs = torch.cat([tra_pois_negs_embs, tra_cats_negs_embs], dim=-1)  # , tra_1sts_negs_embs
        # 上下文：shape=(sz, len+1, 112)
        tra_ctxt_embs = torch.cat([tra_days_embs, tra_slts_embs, tra_hs5s_embs], dim=-1)
        # 长兴趣：shape=(sz, len+1, 7/9)
        # tra_long_idxs = tra_long.type(tra_pois_embs.dtype)
        # mask：shape=(sz, len+1)
        tra_msks = tra_msks.type(tra_pois_embs.dtype)
        # label：shape=(sz, len+1, 21)
        labels = torch.tensor([[0]])  # 第0列是1表示正样本。
        n_classes = torch.tensor(self.num_negs + 1)
        labels_one_hot = torch.nn.functional.one_hot(labels, n_classes)  # (1, 1, 21),
        labels_one_hot = labels_one_hot.repeat(batch_sz, tra_len_plus, 1)
        labels_one_hot = labels_one_hot.type(tra_pois_embs.dtype)  # (batch_sz, len+1, 21)

        '''
        预测过程：基于各种特征预测在当前[t+1]时刻poi，训练时基于[t+1]时的正负样本。
            input_seq是[1, len-1]时刻的真值。
            target_seq是[2, len]时刻的真值, mask也是在这些时刻
            top_idx_sorted预测的是[2, len]时刻的真值
            后边各种计算要根据这个[1, len-1] → [2, len]来调整输入的维度。= [a,b,c] → [b,c,d]
            
            用[0,a,b,c,d]预测[b,c,d], 因为最左侧补了零列用于短期兴趣从seq中取数据。
        tra:
            len_q = 1+num_neg
            len_k = 7/9, days/1sts
            q.shape = (batch_sz, len+1, len_q, d_model) = (sz, len+1, 21, 128)
            k.shape = (batch_sz, len+1, len_k, d_model) = (sz, len+1, 7/9, 128)
            v.shape = (batch_sz, len+1, len_k, d_model) = (sz, len+1, 7/9, 128)
            mask.shape = (batch_sz, len+1, 1, 7/9) = (sz, len+1, 1, 7/9)
        '''
        tra_long = tra_long.unsqueeze(2)  # [1, len-1, 7/9] → [1, len-1, 1, 7/9]
        long_feas = self.dense(tra_long)  # (sz, len+1, 1, 7/9) → (sz, len+1, 1, 128)
        # key, query, value
        tra_q = torch.cat([tra_posi_embs, long_feas], dim=-2)  # (sz, len+1, len_q, 128)
        tra_k = tra_ctxt_embs.unsqueeze(2).repeat(1, 1, self.num_negs + 1, 1)  # (sz, len+1, len_k, 128)
        tra_v = tra_ctxt_embs.unsqueeze(2).repeat(1, 1, self.num_negs + 1, 1)  # (sz, len+1, len_k, 128)
        # mask: 从[c, 1, d]变为[c, d]，变换前的mask是在输入上有效的，可以使用。
        mask = tra_msks.unsqueeze(2).repeat(1, 1, self.num_negs + 1)
        mask = mask.unsqueeze(3)  # (sz, len+1, 1, 7/9)
        mask = mask.type(tra_pois_embs.dtype)

        '''
        长期兴趣特征整合。
        '''
        # Attention权重
        att_wts = self.model_att_qk_sft(tra_q, tra_k, tra_v, mask)  # (sz, len+1, len_q, d_model)
        # (sz, len+1, len_q, d_model) → (sz, len+1, d_model)
        tra_ctxt_2nd = torch.einsum('ijkl,ijlm->ijkm', att_wts, tra_v)  # (sz, len+1, 1, d_model)
        tra_ctxt_2nd = tra_ctxt_2nd.squeeze(2)  # (sz, len+1, d_model)

        '''
        模型结构。
        '''
        # mlp
        # context_2nd = torch.cat([ctxt_embs, ctxt_2nd], dim=-1)
        out_ctxt_2nd = self.model_mlp(tra_ctxt_2nd)  # (sz, len+1, 256)
        out_ctxt_2nd = out_ctxt_2nd.unsqueeze(2).repeat(1, 1, self.num_negs + 1, 1)  # (sz, len+1, 21, 256)
        out_posi_embs = tra_posi_embs.unsqueeze(2).repeat(1, 1, self.num_negs + 1, 1)  # (sz, len+1, 21, 208)
        # concatenate
        out_ctxt_2nd = torch.cat([out_ctxt_2nd, out_posi_embs], dim=-1)  # (sz, len+1, 21, 464)
        # predict
        logits = self.model_dense(out_ctxt_2nd)  # (sz, len+1, 21, 1)
        logits = logits.squeeze(-1)  # (sz, len+1, 21)
        logits = logits * labels_one_hot  # (sz, len+1, 21)
        loss = -torch.log(torch.sigmoid(logits)).sum()  # 注意是求和，而不是均值。loss由正样本+负样本构成。

        return loss
