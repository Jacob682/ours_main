class STPIL_Long_attqk(STPIL_Basic):
    def __init__(self, usr_feas, train, test, long_short, num_inps, dim_feas, num_negs, num_topk,
                 mlp_units, transformer_params, alpha_lambda, dropout_rate):
        super(STPIL_Long_attqk, self).__init__(
            usr_feas, train, test, long_short, num_inps, dim_feas, num_negs, num_topk,
            mlp_units, transformer_params, alpha_lambda, dropout_rate)
        # trainable_variables from outer classes
        self.model_mlp = MLP(self.mlp_units, self.acti, self.rate_mlp)
        self.dense = tf.keras.layers.Dense(self.d_model)     # 调控长期兴趣的维度。
        # self.model_att_k = BahdanauAttentionK(self.d_model, self.rate_long)          # 兴趣用k自己获得。
        self.model_att_qk_sft = BahdanauAttentionQK_softmax(self.d_model, self.rate_long)    # 多兴趣聚合

    def batch_model_tra(self, tra_elements):     # 变量都是batch形式,tra_elements
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
            tra_long, tra_short         # (sz, len+1, 7/9), (sz, len+1, 5)
        ] = tra_elements
        # input: emb    # 这些用于算loss
        # u_embs = self.emb_layer_usr(u_idxs)                   # (sz, 1, 128)
        tra_pois_embs = self.emb_layer_poi(tra_pois_idxs)     # (sz, len+1, 128)
        tra_cats_embs = self.emb_layer_cat(tra_cats_idxs)     # (sz, len+1, 64)
        # tra_1sts_embs = self.emb_layer_1st(tra_1sts_idxs)     # (sz, len+1, 16)
        tra_days_embs = self.emb_layer_day(tra_days_idxs)     # (sz, len+1, 16)
        tra_slts_embs = self.emb_layer_slt(tra_slts_idxs)     # (sz, len+1, 32)
        tra_hs5s_embs = self.emb_layer_hs5(tra_hs5s_idxs)     # (sz, len+1, 64)
        tra_pois_negs_embs = self.emb_layer_poi(tra_pois_negs_idxs)   # (sz, len+1, 10, 128)
        tra_cats_negs_embs = self.emb_layer_cat(tra_cats_negs_idxs)
        # tra_1sts_negs_embs = self.emb_layer_1st(tra_1sts_negs_idxs)
        # input: concatenate
        # ===============================================================
        batch_sz = tf.shape(tra_msks)[0]
        tra_len_plus = tf.shape(tra_msks)[1]
        # 用户的：shape=(sz, len+1, 21\len_q, 128)
        # tra_user_embs = tf.tile(u_embs, [1, tra_len_plus, 1])   # (sz, len+1, 128)
        # tra_user_embs = tf.tile(tf.expand_dims(tra_user_embs, axis=2), [1, 1, 1+self.num_negs, 1])
        # 正样本：shape=(sz, len+1, 208)
        tra_posi_embs = tf.concat([tra_pois_embs, tra_cats_embs], axis=-1)  # , tra_1sts_embs
        # 负样本：shape=(sz, len+1, 10, 208)
        tra_nega_embs = tf.concat([tra_pois_negs_embs, tra_cats_negs_embs], axis=-1)    # , tra_1sts_negs_embs
        # 上下文：shape=(sz, len+1, 112)
        tra_ctxt_embs = tf.concat([tra_days_embs, tra_slts_embs, tra_hs5s_embs], axis=-1)
        # 长兴趣：shape=(sz, len+1, 7/9)
        # tra_long_idxs = tf.cast(tra_long, dtype=tra_pois_embs.dtype)
        # mask：shape=(sz, len+1)
        tra_msks = tf.cast(tra_msks, dtype=tra_pois_embs.dtype)
        # label：shape=(sz, len+1, 21)
        labels = tf.constant([[0]])                     # 第0列是1表示正样本。
        n_classes = tf.constant(self.num_negs + 1)
        labels_one_hot = tf.one_hot(labels, n_classes)  # (1, 1, 21),
        labels_one_hot = tf.tile(labels_one_hot, [batch_sz, tra_len_plus, 1])
        labels_one_hot = tf.cast(labels_one_hot, dtype=tra_pois_embs.dtype)    # (batch_sz, len+1, 21)

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
            q.shape = (batch_sz, len+1, len_q, emb_dim)[:, 2:]
            k.shape = (batch_sz, len+1, len_k, emb_dim)[:, 1:-1]
            m.shape = (batch_sz, len+1, 1\num_heads, 1\len_q, len_k)[:, 1:-1]
            o.shape = (batch_sz, len-1, len_q, d_model)
            模型输出.shape = (batch_sz, len-1, len_q)
            样本真值.shape = (batch_sz, len-1, len_q)
        '''
        # model: 长期兴趣avg
        # ===============================================================
        # 输入：正样本、上下文、长期兴趣切分序列
        # 输出：[k = 切分出的长期兴趣，与k维度对应的mask]
        [
            cum_subs_avg,   # (batch_sz, len+1, 7/9, emb_dim)= (batch_sz, len+1, len_k, emb_dim)
            cum_subs_msk    # (batch_sz, len+1, 7/9, 1)      = (batch_sz, len+1, len_k, 1)
        ] = self.fun_long_interests_att(tra_posi_embs, tra_ctxt_embs, tra_subs_days_idxs)

        # model: transformer
        # ===============================================================
        # 在每一时刻：q=候选item+context, k=多种兴趣, mask是兴趣位上是否有兴趣，pos_encoding在长期兴趣中不需要。
        # target item当做query去参与att计算。
        # 输入：q = [2, len]，k = [1, len-1]，msk = [1, len-1]
        # # 1. 得到：正负候选item拼接作为q
        ipt_q = tf.concat([tf.expand_dims(tra_posi_embs, axis=2), tra_nega_embs], axis=2)
        # # (batch_sz, len+1, len_q, emb_dim), len_q=1+20
        ipt_q = tf.concat([ipt_q,
                           tf.tile(tf.expand_dims(tra_ctxt_embs, axis=2), [1, 1, 1+self.num_negs, 1])], axis=-1)
        # # 2. 得到：根据候选item得到的长期兴趣加权和，有几个候选，就有几个加权和输出。=[2, len]
        cum_subs_avg = self.dense(cum_subs_avg)
        # # att-k(多个兴趣)
        # [
        #     enc_out,    # (batch_sz, len-1, 64)     = (batch_sz, len-1, d_model)
        #     attw        # (batch_sz, len-1, 7/9, 1) = (batch_sz, len-1, len_k, 1)
        # ] = self.model_att_k(k=cum_subs_avg[:, 1:-1],
        #                      mask=cum_subs_msk[:, 1:-1], training=self.training)
        # enc_out_lenq = tf.tile(tf.expand_dims(enc_out, -2), [1, 1, 1+self.num_negs, 1])   # (batch_sz, len-1, len_q, d_model)
        # # att-qk(多个兴趣)
        k = tf.tile(tf.expand_dims(cum_subs_avg[:, 1:-1], axis=2), [1, 1, 1+self.num_negs, 1, 1])
        m = tf.tile(tf.expand_dims(cum_subs_msk[:, 1:-1], axis=2), [1, 1, 1+self.num_negs, 1, 1])
        [
            att_out,        # (batch_sz, len-1, len_q, d_model)
            attw            # (batch_sz, len-1, len_q, 7, 1)
        ] = self.model_att_qk_sft(q=ipt_q[:, 2:],   # q.shape=(batch_sz, len-1, len_q, emb_dim)
                                  k=k,              # k.shape=(batch_sz, len-1, len_q, 7, d_model)
                                  mask=m,           # m.shape=(batch_sz, len-1, len_q, 7, 1)
                                  training=self.training)

        # model: mlp + score, 均是[2, len]时刻,
        # ===============================================================
        # 输入：用户特征、长期兴趣加权和、候选item完整信息，
        # 输出：几个候选的得分概率
        concat = tf.concat([
            # tra_user_embs[:, 2:],
            # enc_out_lenq,     # att-k
            att_out,            # att-qk
            ipt_q[:, 2:]
        ], axis=-1)                                         # (batch_sz, len-1, len_q, emb_dim)
        mlp_out = self.model_mlp(concat, self.training)     # (batch_sz, len-1, len_q)
        seq_loss = self.fun_compute_log_loss(self.loss_object_tra, labels_one_hot[:, 2:], mlp_out, tra_msks[:, 2:])
        loss_model = seq_loss

        # L2 loss
        # ===============================================================
        # 这里emb单独计算L2 = lambda / 2 * X^2 = lambda * nn.l2_loss() = model.losses / 2
        emb_for_l2_1 = [# u_embs,
                        tra_pois_embs, tra_cats_embs,                           # tra_1sts_embs,
                        tra_days_embs, tra_slts_embs, tra_hs5s_embs]    # 5大特征
        emb_for_l2_2 = [tra_pois_negs_embs, tra_cats_negs_embs] # 负样本         # , tra_1sts_negs_embs
        emb_for_l2_3 = [self.model_mlp, self.dense, # self.model_att_k
                        self.model_att_qk_sft]     # 子网络
        l2_los_emb = self.lm / 2. * (       # 只累加mask=1的位置。
            tf.reduce_sum([
                tf.reduce_sum(tf.reduce_sum(emb_1 * emb_1, axis=2) * tra_msks) for emb_1 in emb_for_l2_1]) +
            tf.reduce_sum([
                tf.reduce_sum(tf.reduce_sum(emb_2 * emb_2, axis=[2, 3]) * tra_msks / self.num_negs) for emb_2 in emb_for_l2_2]))
        l2_los_weight = self.lm * (
            tf.reduce_sum([
                tf.reduce_sum([tf.nn.l2_loss(v) for v in model_sub.trainable_variables]) for model_sub in emb_for_l2_3]))

        loss_total = loss_model + l2_los_emb + l2_los_weight    # batch内loss加和
        return loss_model, loss_total

    def batch_model_tes(self, tra_elements, tes_elements):   # 变量都是batch形式
        # input
        # ===============================================================
        # input: idx
        [
            # (sz, 1), (sz, len+1), (sz, len+1, 7/9)
            # test时: 用于生成transformer的key，每个用户的表达固定。不需要使用负样本。
            u_idxs,
            tra_pois_idxs, tra_cats_idxs, tra_1sts_idxs, tra_days_idxs, tra_slts_idxs, tra_hs5s_idxs, tra_msks,
            tra_subs_days_idxs, tra_subs_1sts_idxs,
            tra_rect_timh_idxs, tra_rect_dist_idxs, tra_rect_timv_idxs, tra_rect_ctxt_idxs,
            tra_long, tra_short         # (sz, len+1, 7/9), (sz, len+1, 5)
        ] = tra_elements
        [
            # (sz, len_tes), (sz, len_tes), (sz, len_tes, num_poi+1)    # len_tes=1
            # test时: 用于生成transformer的query，即全部item特征拼接上下文真值后，分别与用户的各个key做一遍查询。会有些耗时。
            tes_pois_idxs,
            tes_days_idxs, tes_slts_idxs, tes_hs5s_idxs, tes_msks,  # 上下文真值
            tes_hash_mask
        ] = tes_elements

        # input: emb
        # u_embs = self.emb_layer_usr(u_idxs)                   # (sz, 1, 128)
        tra_pois_embs = self.emb_layer_poi(tra_pois_idxs)     # (sz, len+1, 128)
        tra_cats_embs = self.emb_layer_cat(tra_cats_idxs)     # (sz, len+1, 64)
        # tra_1sts_embs = self.emb_layer_1st(tra_1sts_idxs)     # (sz, len+1, 16)
        tra_days_embs = self.emb_layer_day(tra_days_idxs)     # (sz, len+1, 16)
        tra_slts_embs = self.emb_layer_slt(tra_slts_idxs)     # (sz, len+1, 32)
        tra_hs5s_embs = self.emb_layer_hs5(tra_hs5s_idxs)     # (sz, len+1, 64)
        tes_all_pois_embs = self.emb_layer_poi(self.tes_all_pois_idxs) # (num_poi+1, 128)
        tes_all_cats_embs = self.emb_layer_cat(self.tes_all_cats_idxs) # (num_poi+1, 64)
        # tes_all_1sts_embs = self.emb_layer_1st(self.tes_all_1sts_idxs) # (num_poi+1, 16)
        tes_days_embs = self.emb_layer_day(tes_days_idxs)     # (sz, 1, 16)
        tes_slts_embs = self.emb_layer_slt(tes_slts_idxs)     # (sz, 1, 32)
        tes_hs5s_idxs = self.emb_layer_hs5(tes_hs5s_idxs)     # (sz, 1, 64)
        # input: concatenate
        # ===============================================================
        batch_sz = tf.shape(tra_msks)[0]
        # tra_len_plus = tf.shape(tra_msks)[1]
        # tes_len_orig = tf.shape(tes_msks)[1]

        # tra：生成transformer key
        # 正样本：shape=(sz, len+1, 272)
        tra_posi_embs = tf.concat([tra_pois_embs, tra_cats_embs], axis=-1)  # , tra_1sts_embs
        # 上下文：shape=(sz, len+1, 112)
        tra_ctxt_embs = tf.concat([tra_days_embs, tra_slts_embs, tra_hs5s_embs], axis=-1)
        # 长兴趣：shape=(sz, len+1, 7/9)
        # tra_long_idxs = tf.cast(tra_long, dtype=tra_pois_embs.dtype)
        # mask：shape=(sz, len+1)
        # tra_msks = tf.cast(tra_msks, dtype=u_embs.dtype)  # tes时需用int32

        # tes: 生成transformer query
        # 用户的：shape=(sz, 1, num_poi+1, 128)
        # tra_user_embs = tf.tile(tra_pois_embs, [1, self.num_pois+1, 1]) # (sz, num_poi+1, 128)
        # tra_user_embs = tf.expand_dims(tra_user_embs, 1)
        # 候选的样本：shape=(sz, num_poi+1, 208), sz中每个用户，都要对所有候选poi算分。
        tes_all_cand_embs = tf.concat([tes_all_pois_embs, tes_all_cats_embs], axis=-1)  # , tes_all_1sts_embs
        tes_all_cand_embs = tf.expand_dims(tes_all_cand_embs, 0)            # (1, num_poi+1, 272)
        tes_all_cand_embs = tf.tile(tes_all_cand_embs, [batch_sz, 1, 1])    # (sz, num_poi+1, 272)
        # 候选样本的上下文真值GT：shape=(sz, num_poi+1, 112)
        tes_ctxt_embs = tf.concat([tes_days_embs, tes_slts_embs, tes_hs5s_idxs], axis=-1)
        tes_ctxt_embs = tf.tile(tes_ctxt_embs, [1, self.num_pois+1, 1])  # ipt_q中，每个候选item都拼接上同样的ctxt
        # mask: shape=(sz, 1)
        tes_msks = tf.cast(tes_msks, dtype=tra_pois_embs.dtype)
        tes_hash_mask = tf.cast(tes_hash_mask, dtype=tra_pois_embs.dtype)

        '''
        预测过程：
            用seq=[0,a,b,c,d]预测[e], 因为最左侧补了零列用于短期兴趣从seq中取数据。
        tes:
            len_q = 1+num_poi
            len_k = 7/9, days/1sts
            q.shape = (batch_sz, 1, len_q, emb_dim)[:, 2:]
            k.shape = (batch_sz, 1, len_k, emb_dim)[:, 1:-1]
            m.shape = (batch_sz, 1, 1\num_heads, 1\len_q, len_k)[:, 1:-1]
            o.shape = (batch_sz, 1, len_q, d_model)
            模型输出.shape = (batch_sz, 1, len_q)
            样本真值.shape = (batch_sz, 1)
        '''
        # model: 长期兴趣avg
        # ===============================================================
        # 输入：正样本、上下文、长期兴趣切分序列
        # 输出：[k = 切分出的长期兴趣，与k维度对应的mask]
        [
            cum_subs_avg,   # (batch_sz, len+1, 7/9, emb_dim)= (batch_sz, len+1, len_k, emb_dim)
            cum_subs_msk    # (batch_sz, len+1, 7/9, 1)      = (batch_sz, len+1, len_k, 1)
        ] = self.fun_long_interests_att(tra_posi_embs, tra_ctxt_embs, tra_subs_days_idxs)

        # 获得各个用户在tra_last_one上的长期兴趣表达
        # seq=[0,a,b,c,d], mask=[0,1,1,1,1], sum(mask)=4, seq[4]='d'=tra_last_one
        cum_subs_avg = tf.gather(cum_subs_avg, indices=tf.reduce_sum(tra_msks, axis=1), batch_dims=1)
        cum_subs_msk = tf.gather(cum_subs_msk, indices=tf.reduce_sum(tra_msks, axis=1), batch_dims=1)
        cum_subs_avg = tf.expand_dims(cum_subs_avg, 1)  # (batch_sz, 1, 7/9, emb_dim)
        cum_subs_msk = tf.expand_dims(cum_subs_msk, 1)  # (batch_sz, 1, 7/9, 1)

        # model: transformer
        # ===============================================================
        # 在每一时刻：q=所有候选item+真值context, k=多种兴趣(tra_last_one), mask是兴趣位上是否有兴趣，pos_encoding在长期兴趣中不需要。
        # target item当做query去参与att计算。
        # 输入：q = [len+1]，k = [len]，msk = [len], 其中len == tra_last_one
        # # 1. 得到：正负候选item拼接作为q
        ipt_q = tf.concat([tes_all_cand_embs, tes_ctxt_embs], axis=-1)
        # # (batch_sz, 1, num_poi+1, emb_dim)
        ipt_q = tf.expand_dims(ipt_q, 1)
        # # 2. 得到：根据候选item得到的长期兴趣加权和，有几个候选，就有几个加权和输出。=[2, len]
        cum_subs_avg = self.dense(cum_subs_avg)
        # # att-k(多个兴趣)
        # [
        #     enc_out,    # (batch_sz, 1, 64)     = (batch_sz, 1, d_model)
        #     attw        # (batch_sz, 1, 7/9, 1) = (batch_sz, 1, len_k, 1)
        # ] = self.model_att_k(k=cum_subs_avg,
        #                      mask=cum_subs_msk, training=self.training)
        # enc_out_lenq = tf.tile(tf.expand_dims(enc_out, -2), [1, 1, 1+self.num_pois, 1])   # (batch_sz, 1, num_poi+1, d_model)
        # # att-qk(多个兴趣)
        k = tf.tile(tf.expand_dims(cum_subs_avg, axis=2), [1, 1, 1+self.num_pois, 1, 1])
        m = tf.tile(tf.expand_dims(cum_subs_msk, axis=2), [1, 1, 1+self.num_pois, 1, 1])
        [
            att_out,        # (batch_sz, 1, num_poi+1, d_model)
            attw            # (batch_sz, 1, num_poi+1, 7, 1)
        ] = self.model_att_qk_sft(q=ipt_q,   # q.shape=(batch_sz, 1, num_poi+1, emb_dim)
                                  k=k,       # k.shape=(batch_sz, 1, num_poi+1, 7, d_model)
                                  mask=m,    # m.shape=(batch_sz, 1, num_poi+1, 7, 1)
                                  training=self.training)

        # model: mlp + score, 均是[2, len]时刻,
        # ===============================================================
        # 输入：用户特征、长期兴趣加权和、候选item完整信息，
        # 输出：几个候选的得分概率
        concat = tf.concat([
            # tra_user_embs,
            # enc_out_lenq,     # att-k
            att_out,        # att-qk
            ipt_q
        ], axis=-1)                                         # (batch_sz, 1, num_poi+1, emb_dim)
        mlp_out = self.model_mlp(concat, self.training)     # (batch_sz, 1, num_poi+1), len_tes=1, len_q=num_poi+1
        # 得到对test集中各个真值的预测
        # mlp_out *= tes_hash_mask
        top_val, top_idx_sorted = tf.math.top_k(
            mlp_out, k=self.num_topk[-1], sorted=True)  # 俩返回值shape = (batch_sz, len_tes=1, top_num)

        seq_loss = self.fun_compute_log_loss(self.loss_object_tes, tes_pois_idxs, mlp_out, tes_msks)

        loss_model = seq_loss
        return loss_model, top_idx_sorted   # shape = (batch_sz, len_tes, top_num)
