# -*- coding: utf-8 -*-
# @Time    : 2021/12/20 12:08
# @Author  : Chen Xu
# @Email   : xc_chen@ruc.edu.cn

r"""
@inproceedings{10.1145/3511808.3557299,
author = {Xu, Chen and Xu, Jun and Chen, Xu and Dong, Zhenghua and Wen, Ji-Rong},
title = {Dually Enhanced Propensity Score Estimation in Sequential Recommendation},
year = {2022},
isbn = {9781450392365},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3511808.3557299},
doi = {10.1145/3511808.3557299},
abstract = {Sequential recommender systems train their models based on a large amount of implicit user feedback data and may be subject to biases when users are systematically under/over-exposed to certain items. Unbiased learning based on inverse propensity scores (IPS), which estimate the probability of observing a user-item pair given the historical information, has been proposed to address the issue. In these methods, propensity score estimation is usually limited to the view of item, that is, treating the feedback data as sequences of items that interacted with the users. However, the feedback data can also be treated from the view of user, as the sequences of users that interact with the items. Moreover, the two views can jointly enhance the propensity score estimation. Inspired by the observation, we propose to estimate the propensity scores from the views of user and item, called Dually Enhanced Propensity Score Estimation (DEPS). Specifically, given a target user-item pair and the corresponding item and user interaction sequences, DEPS first constructs a time-aware causal graph to represent the user-item observational probability. According to the graph, two complementary propensity scores are estimated from the views of item and user, respectively, based on the same set of user feedback data. Finally, two transformers are designed to make use of the two propensity scores and make the final preference prediction. Theoretical analysis showed the unbiasedness and variance of DEPS. Experimental results on three publicly available benchmarks and a proprietary industrial dataset demonstrated that DEPS can significantly outperform the state-of-the-art baselines.},
booktitle = {Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
pages = {2260–2269},
numpages = {10},
keywords = {propensity score estimation, sequential recommendation},
location = {Atlanta, GA, USA},
series = {CIKM '22}
}

"""

import random

import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder,MLPLayers, SequenceAttLayer, ContextSeqEmbLayer
from recbole.utils import InputType
from torch.nn.init import xavier_normal_, constant_
from recbole.model.loss import IPSDualBCELoss
import numpy as np

import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence




class Timeware_Propensity_Estimation(nn.Module):
    def __init__(self, hidden_size, dropout_prob,config,dataset,max_seq_length,num_layer = 2):
        super(Timeware_Propensity_Estimation, self).__init__()
        #Since the transformer already encode the sequence information, we only need to calculate the item probability through two layer MLP
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.device = config['device']
        self.gru = config['gru_type']
        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.max_seq_length = max_seq_length
        self.gru_layers = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layer,
            bias=False,
            batch_first=True,
        )
        self.item_feat = dataset.get_item_feature()
        num_item_feature = len(self.item_feat.interaction)

        item_feat_dim =self.hidden_size
        mask_mat = torch.arange(self.max_seq_length).to(self.device).view(1, -1)  # init mask
        # init sizes of used layers
        self.att_list = [4  * self.hidden_size] + self.mlp_hidden_size
        self.interest_evolution = InterestEvolvingLayer(
            mask_mat, item_feat_dim, item_feat_dim, self.att_list, gru=self.gru
        )
        self.dense1 = nn.Linear(self.hidden_size, self.hidden_size)

        self.loss = nn.CrossEntropyLoss()

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self,item_seq_emb, target_item, item_emb, seq_len,target_item_feat_emb,tr_output):
        ##IPS not need to backward to the underlying model
        item_emb = item_emb
        #（掩码 + 位置）
        item_seq_emb = item_seq_emb
        target_item_feat_emb=target_item_feat_emb
        gru_output, _ = self.gru_layers(item_seq_emb)
        gru_output = self.dense1(gru_output)
        seq_output = self.gather_indexes(gru_output, torch.max(torch.zeros_like(seq_len),seq_len - 1))
        #乘embedding的权重参数item_emb
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))

        IPS_score = torch.softmax(logits,dim=-1)

        IPS_score = IPS_score.gather(dim=-1,index=target_item.unsqueeze(-1))
        aux_loss = self.loss(logits, target_item)
        # print(f"item_emb:{item_emb.shape},item_seq_emb:{item_seq_emb.shape},gru_output:{gru_output.shape},\
        # seq_output:{seq_output.shape},logits:{logits.shape},target_item:{target_item.shape},IPS_score:{IPS_score.shape},loss:{loss}")
        #return IPS_score, loss, logits
        evolution = self.interest_evolution(target_item_feat_emb, gru_output, seq_len,tr_output)

        # dien_in = torch.cat([evolution, target_item_feat_emb, user_feat_list], dim=-1)
        # # input the DNN to get the prediction score
        # dien_out = self.dnn_mlp_layers(dien_in)
        # preds = self.dnn_predict_layer(dien_out)
        # preds = self.sigmoid(preds)

        #return preds.squeeze(1), aux_loss
        if torch.isnan(aux_loss):
            aux_loss=0
            print("aux_loss为nan")
        return aux_loss,evolution

class DIENTR(SequentialRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DIENTR, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']


        self.embedding_size = config['embedding_size']
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.mask_ratio = config['max_mask_ratio']

        self.min_mask_ratio = config['min_mask_ratio']

        self.warmup_rate = config['warmup_rate']

        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']

        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']

        self.loss_type = config['loss_type']

        self.mlm_factor = config['mlm_factor']
        self.IPS_factor = config['IPS_factor']

        self.data_type = config['dataset']

        # load dataset info
        self.item_mask_token = self.n_items
        self.user_mask_token = self.n_users
        #self.cls_token = 1
        self.dataset = dataset

        self.ItemIPS_Estimation_Net = Timeware_Propensity_Estimation(hidden_size=self.hidden_size,dropout_prob=self.dropout_prob,config =config,dataset=dataset,max_seq_length=self.max_seq_length)
        self.UserIPS_Estimation_Net = Timeware_Propensity_Estimation(hidden_size=self.hidden_size,dropout_prob=self.dropout_prob,config =config,dataset=dataset,max_seq_length=self.max_user_seq_length )


        self.types = ['user', 'item']
        self.user_feat = dataset.get_user_feature()
        self.item_feat = dataset.get_item_feature()

        self.DualLoss = nn.MSELoss()

        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # mask token add 1
        self.user_embedding = nn.Embedding(self.n_users + 1, self.hidden_size, padding_idx=0)  # mask token add 1

        self.pooling_mode = config['pooling_mode']
        self.device = config['device']

        self.item_position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.user_position_embedding = nn.Embedding(self.max_user_seq_length, self.hidden_size)

        self.item_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.user_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.mlp_hidden_size = [6 * self.hidden_size] + self.mlp_hidden_size

        self.dnn_mlp_layers = MLPLayers(self.mlp_hidden_size, activation='Dice', dropout=self.dropout_prob, bn=True)
        self.dnn_predict_layers = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()


        ##Recbole load sequences is different from ours. To avoid un-fair comparision, we load the item and user sequences in this function
        self.Item2UserSeq,self.Item2TimeSeq = self.LoadUserSeq()

        self.total_step = int(config['epochs'] * (self.interaction_num/config['train_batch_size']))
        self.batch_step = int(self.interaction_num/config['train_batch_size']) * 2

        self.IPS_training_rate = int(config['IPS_training_rate'])

        self.current_step = 0

        self.loss = IPSDualBCELoss(clip=config['IPS_clip'],max_clip=config['IPS_max_clip'],
                                   total_steps=self.total_step, warmup_rate=self.warmup_rate,alpha=config['alpha'])

        self.apply(self._init_weights)


    def GetUserSeq(self,item_id,time):
        items = item_id.detach().cpu().numpy()
        times = time.detach().cpu().numpy()
        user_seq = []
        user_seq_length = []

        for i in range(len(items)):
            #init_time = 0
            index = 0
            item = items[i]
            if item in self.Item2TimeSeq:
                for t in self.Item2TimeSeq[item]:
                    if int(times[i]) <= t:
                        index = index + 1
                    else:
                        break
                if index == 0:
                    user_seq_length.append(0)
                    user_seq.append([0] * self.max_user_seq_length)
                else:
                    min_pos = max(0,index-self.max_user_seq_length)
                    length = index-min_pos
                    user_seq.append(self.Item2UserSeq[item][min_pos:index] + [0] * (self.max_user_seq_length-length))
                    user_seq_length.append(length)
            else:
                user_seq_length.append(0)
                user_seq.append([0] * self.max_user_seq_length)

        return torch.LongTensor(user_seq).to(self.device),torch.LongTensor(user_seq_length).to(self.device)


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.1)
            #xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)

    def LoadUserSeq(self):
        import os
        train,valid,test = pd.read_csv(os.path.join("dataset",self.data_type,self.data_type+".train.inter"),delimiter='\t'),\
        pd.read_csv(os.path.join("dataset",self.data_type,self.data_type+".valid.inter"),delimiter='\t'),\
        pd.read_csv(os.path.join("dataset",self.data_type,self.data_type+".test.inter"),delimiter='\t')

        all = pd.concat((train,valid,test),axis=0)

        self.interaction_num = len(train)

        iid_field,uid_field,label_field,iid_list_field,uid_list_field,time_field = all.columns
        all.sort_values(by=[time_field], ascending=True,inplace=True)
        Item2UserSeq = {}
        Item2TimeSeq = {}
        for i in range(len(all)):
            item_id,user_id,uid_list,time = all.iloc[i,0],all.iloc[i,1],all.iloc[i,4],all.iloc[i,5]
            item_id = self.dataset.token2id(self.dataset.iid_field,str(item_id))
            user_id = self.dataset.token2id(self.dataset.uid_field,str(user_id))
            if item_id not in Item2UserSeq.keys():
                Item2UserSeq[item_id] = []
                Item2TimeSeq[item_id] = []
            Item2UserSeq[item_id].append(user_id)
            Item2TimeSeq[item_id].append(int(time))


        return Item2UserSeq,Item2TimeSeq

    def get_attention_mask(self, seq, seq_len):
        """Generate bidirectional attention mask for multi-head attention."""

        attention_mask = (seq > 0).long() #[B,L]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


    def _neg_sample(self, item_set):
        item = random.randint(1, self.n_items - 1)
        while item in item_set:
            item = random.randint(1, self.n_items - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def reconstruct_train_data(self, seq, mask_length, pad_item = True):
        """
        Mask item sequence for MLM training in the transformers.
        """
        device = seq.device
        batch_size = seq.size(0)

        sequence_instances = seq.cpu().numpy().tolist()

        masked_sequences = []
        pos_values = []
        #neg_items = []
        masked_index = []
        for m, instance in enumerate(sequence_instances):
            # WE MUST USE 'copy()' HERE!
            #valid_length = seq_len[m].item()
            masked_sequence = instance.copy()
            pos = []
            # neg_item = []
            index_ids = []
            t = 0
            for index_id, j in enumerate(instance):
                # padding is 0, the sequence is end
                t = index_id
                if j == 0:
                    break
                prob = random.random()
                if prob < self.mask_ratio:
                    pos.append(j)
                    if pad_item:
                        masked_sequence[index_id] = self.item_mask_token
                    else:
                        masked_sequence[index_id] = self.user_mask_token
                    index_ids.append(index_id) ####skip the cls vector

            masked_sequences.append(masked_sequence)
            pos_values.append(self._padding_sequence(pos, mask_length))
            #neg_items.append(self._padding_sequence(neg_item, self.mask_item_length))
            masked_index.append(self._padding_sequence(index_ids, mask_length))

        # [B Len]

        masked_sequences = torch.tensor(masked_sequences, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        pos_values = torch.tensor(pos_values, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]

        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_sequences, pos_values, masked_index


    def forward(self, user, item_seq, next_items, user_seq, item_seq_len, user_seq_len):

        target_user_feat_emb = self.user_embedding(user)
        user_feat_list = self.user_embedding(user_seq)

        item_feat_list = self.item_embedding(item_seq)
        target_item_feat_emb = self.item_embedding(next_items)

        position_ids = torch.arange(self.max_seq_length, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.item_position_embedding(position_ids)
        #label_embedding = self.item_label_embedding(item_label_seq)

        input_emb_item = item_feat_list + position_embedding
        input_emb_item = self.LayerNorm(input_emb_item)
        input_emb_item = self.dropout(input_emb_item)
        extended_attention_mask = self.get_attention_mask(item_seq,item_seq_len)

        #print(extended_attention_mask[0:2])
        item_output = self.item_encoder(input_emb_item, extended_attention_mask, output_all_encoded_layers=True)
        item_output = item_output[-1]

        position_ids = torch.arange(self.max_user_seq_length, dtype=torch.long, device=user_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(user_seq)
        position_embedding = self.user_position_embedding(position_ids)
        #label_embedding = self.user_label_embedding(user_label_seq)

        input_emb_user = user_feat_list + position_embedding
        input_emb_user = self.LayerNorm(input_emb_user)
        input_emb_user = self.dropout(input_emb_user)
        extended_attention_mask = self.get_attention_mask(user_seq, user_seq_len)
        user_output = self.user_encoder(input_emb_user, extended_attention_mask, output_all_encoded_layers=True)
        user_output = user_output[-1]

        #exit(0)
        return item_output,target_item_feat_emb,user_output,target_user_feat_emb,input_emb_item,input_emb_user  # [B H]

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        times = interaction[self.TIME]
        user = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        #这个为什么是数字，位置吗
        next_items = interaction[self.ITEM_ID]
        #user_seq为不同item下的用户序列，user_seq_len为每个iten下的用户序列长度
        user_seq,user_seq_len = self.GetUserSeq(next_items,times)

        label = interaction[self.LABEL_FIELD]

        self.mask_user_length = int(self.mask_ratio * self.max_user_seq_length)
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        masked_item_seq, pos_items, masked_item_index = self.reconstruct_train_data(item_seq,self.mask_item_length, pad_item=True)
        masked_user_seq, pos_users, masked_user_index = self.reconstruct_train_data(user_seq,self.mask_user_length ,pad_item=False)

        item_output,target_item_feat_emb,user_output,target_user_feat_emb,item_seq_emb,user_seq_emb = self.forward(user=user, item_seq=masked_item_seq, next_items=next_items, user_seq=masked_user_seq,
                                                                                                                   item_seq_len=item_seq_len, user_seq_len=user_seq_len)

        pred_item_index_map = self.multi_hot_embed(masked_item_index, masked_item_seq.size(-1))  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_item_index_map = pred_item_index_map.view(masked_item_index.size(0), masked_item_index.size(1), -1)  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_item_index_map, item_output)  # [B mask_len H]

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        #test_user_emb,test_item_emb = self.embedding_layer.GetAllFeature(self.n_users,self.n_items)
        test_user_emb = self.user_embedding.weight[:self.n_users]
        test_item_emb = self.item_embedding.weight[:self.n_items]

        item_logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B mask_len item_num]
        item_targets = (masked_item_index > 0).float().view(-1)  # [B*mask_len]

        pred_user_index_map = self.multi_hot_embed(masked_user_index, masked_user_seq.size(-1))  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_user_index_map = pred_user_index_map.view(masked_user_index.size(0), masked_user_index.size(1), -1)  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_user_index_map, user_output)  # [B mask_len H]

        user_logits = torch.matmul(seq_output, test_user_emb.transpose(0, 1))  # [B mask_len item_num]
        user_targets = (masked_user_index > 0).float().view(-1)  # [B*mask_len]


        item_social_emb = torch.mean(user_output*((masked_user_seq>0).float().view(masked_user_seq.size(0),-1,1)),dim=1,keepdim=False)

        user_history_emb = torch.mean(item_output*((masked_item_seq>0).float().view(masked_item_seq.size(0),-1,1)),dim=1,keepdim=False)


        item_emb = torch.cat((item_social_emb,target_item_feat_emb),dim=-1)
        user_emb = torch.cat((user_history_emb,target_user_feat_emb),dim=-1)
        item_IPS_loss, item_gruout = self.ItemIPS_Estimation_Net(item_seq_emb, next_items, test_item_emb,item_seq_len,target_item_feat_emb,item_output)
        user_IPS_loss, user_gruout = self.UserIPS_Estimation_Net(user_seq_emb, user, test_user_emb,user_seq_len,target_user_feat_emb,user_output)
        in_r = torch.cat([user_emb, item_emb,item_gruout,user_gruout], dim=-1)
        out_r = self.dnn_mlp_layers(in_r)


        preds = self.dnn_predict_layers(out_r)
        preds = self.sigmoid(preds).squeeze(1)

        item_mlm_loss = torch.sum(loss_fct(item_logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * item_targets) \
                        / torch.sum(item_targets)

        user_mlm_loss = torch.sum(loss_fct(user_logits.view(-1, test_user_emb.size(0)), pos_users.view(-1)) * user_targets) \
                        / torch.sum(user_targets)

        #pred = preds[preds.isnan()] = 0
        dual_loss = self.DualLoss(item_IPS_loss,user_IPS_loss)
        print(f"predsshape:{preds},labelsshape:{label}")
        loss = F.binary_cross_entropy(preds, label, reduction='mean')
        # if torch.isnan(loss):
        #     loss=0
        #     print("loss为nan")
        #loss = self.loss(preds,label,p_item_score,p_user_score)
        # print(f"item_IPS_loss:{item_IPS_loss}，\
        # ,user_IPS_loss:{user_IPS_loss},user_mlm_loss:{user_mlm_loss},loss:{loss}")
        if self.training:
            self.current_step = self.current_step + 1
            self.loss.update()
            if self.current_step > self.warmup_rate * self.total_step:
                #unbiased learning phrase
                self.mask_ratio = self.min_mask_ratio
                if self.current_step % (self.batch_step*(1+self.IPS_training_rate)) <= self.batch_step:
                    return loss
                else:
                    return item_IPS_loss+user_IPS_loss+dual_loss
            else:
                return item_mlm_loss+user_mlm_loss+item_IPS_loss+user_IPS_loss+dual_loss

        return loss


    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        test_item = interaction[self.ITEM_ID]
        user = interaction[self.USER_ID]
        times = interaction[self.TIME]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_seq,user_seq_len = self.GetUserSeq(test_item,times)
        test_item_emb = self.item_embedding.weight[:self.n_items]
        test_user_emb = self.user_embedding.weight[:self.n_users]
        item_output,target_item_feat_emb,user_output,target_user_feat_emb,item_seq_emb,user_seq_emb = self.forward(user=user, item_seq=item_seq, next_items=test_item, user_seq=user_seq,
                                                                                                                   item_seq_len=item_seq_len,user_seq_len=user_seq_len)
        item_emb = torch.cat((torch.mean(user_output*((user_seq >0).float().view(user_seq.size(0),-1,1)),dim=1,keepdim=False),target_item_feat_emb),dim=-1)
        user_emb = torch.cat((torch.mean(item_output*((item_seq >0).float().view(item_seq.size(0),-1,1)),dim=1,keepdim=False),target_user_feat_emb),dim=-1)
        item_IPS_loss, item_gruout = self.ItemIPS_Estimation_Net(item_seq_emb, test_item, test_item_emb, item_seq_len,
                                                                 target_item_feat_emb,item_output)
        user_IPS_loss, user_gruout = self.UserIPS_Estimation_Net(user_seq_emb, user, test_user_emb, user_seq_len,
                                                                 target_user_feat_emb,user_output)
        in_r = torch.cat([user_emb, item_emb,item_gruout,user_gruout], dim=-1)
        out_r = self.dnn_mlp_layers(in_r)
        preds = self.dnn_predict_layers(out_r)
        scores = self.sigmoid(preds).squeeze(1)

        return scores


class InterestEvolvingLayer(nn.Module):
    """As the joint influence from external environment and internal cognition, different kinds of user interests are
    evolving over time. Interest Evolving Layer can capture interest evolving process that is relative to the target
    item.
    """

    def __init__(
        self,
        mask_mat,
        input_size,
        rnn_hidden_size,
        att_hidden_size=(80, 40),
        activation='sigmoid',
        softmax_stag=False,
        gru='GRU'
    ):
        super(InterestEvolvingLayer, self).__init__()

        self.mask_mat = mask_mat
        self.gru = gru

        self.attention_layer = SequenceAttLayer(mask_mat, att_hidden_size, activation, softmax_stag, True)
        self.dynamic_rnn = DynamicRNN(input_size=input_size, hidden_size=rnn_hidden_size, gru=gru)

    def final_output(self, outputs, keys_length):
        """get the last effective value in the interest evolution sequence
        Args:
            outputs (torch.Tensor): the output of `DynamicRNN` after `pad_packed_sequence`
            keys_length (torch.Tensor): the true length of the user history sequence

        Returns:
            torch.Tensor: The user's CTR for the next item
        """
        batch_size, hist_len, _ = outputs.shape  # [B, T, H]
        keys_length=keys_length.view(-1, 1) - 1
        mask = (
            torch.arange(hist_len, device=keys_length.device).repeat(batch_size, 1) == torch.max(torch.zeros_like(keys_length),keys_length)
        )

        return outputs[mask]

    def forward(self, queries, keys, keys_length,tr_output):
        hist_len = keys.shape[1]  # T
        keys_length_cpu = keys_length.cpu()
        if self.gru == 'GRU':
            packed_keys = pack_padded_sequence(
                input=keys, lengths=keys_length_cpu, batch_first=True, enforce_sorted=False
            )
            packed_rnn_outputs, _ = self.dynamic_rnn(packed_keys)
            rnn_outputs, _ = pad_packed_sequence(
                packed_rnn_outputs, batch_first=True, padding_value=0.0, total_length=hist_len
            )
            att_outputs = self.attention_layer(queries, rnn_outputs, keys_length)
            outputs = att_outputs.squeeze(1)

        # AIGRU
        elif self.gru == 'AIGRU':
            att_outputs = self.attention_layer(queries, keys, keys_length)
            interest = keys * att_outputs.transpose(1, 2)
            packed_rnn_outputs = pack_padded_sequence(
                interest, lengths=keys_length_cpu, batch_first=True, enforce_sorted=False
            )
            _, outputs = self.dynamic_rnn(packed_rnn_outputs)
            outputs = outputs.squeeze(0)

        elif self.gru == 'AGRU' or self.gru == 'AUGRU':
            att_outputs = self.attention_layer(queries, keys, keys_length).squeeze(1)  # [B, T]
            # packed_rnn_outputs = pack_padded_sequence(
            #     keys, lengths=keys_length_cpu, batch_first=True, enforce_sorted=False
            # )
            # packed_att_outputs = pack_padded_sequence(
            #     att_outputs, lengths=keys_length_cpu, batch_first=True, enforce_sorted=False
            # )
            outputs = self.dynamic_rnn(keys, tr_output)
            #outputs, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0, total_length=hist_len)
            outputs = self.final_output(outputs, keys_length)  # [B, H]

        return outputs

class DynamicRNN(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, gru='AGRU'):
        super(DynamicRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = AUGRUCell(input_size, hidden_size, bias)

    def forward(self, input, att_scores=None, hidden_output=None):
        # if not isinstance(input, PackedSequence) or not isinstance(att_scores, PackedSequence):
        #     raise NotImplementedError("DynamicRNN only supports packed input and att_scores")

        #input, batch_sizes, sorted_indices, unsorted_indices = input
        # att_scores = att_scores.data
        #
        # max_batch_size = int(batch_sizes[0])
        # if hidden_output is None:
        #     hidden_output = torch.zeros(max_batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        #
        # outputs = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        #
        # begin = 0
        # for batch in batch_sizes:
        #     new_hx = self.rnn(input[begin:begin + batch], hidden_output[0:batch], att_scores[begin:begin + batch])
        #     outputs[begin:begin + batch] = new_hx
        #     hidden_output = new_hx
        #     begin += batch

        #return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)
        if hidden_output is None:
            hidden_output = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)

        outputs = torch.zeros_like(input, dtype=input.dtype, device=input.device)

        begin = 0
        for i in range(input.size(1)):
            new_hx = self.rnn(input[:,i,:], hidden_output, att_scores[:,i,:])
            outputs[:,i,:] = new_hx
            hidden_output = new_hx

        return outputs
class AUGRUCell(nn.Module):
    """ Effect of GRU with attentional update gate (AUGRU). AUGRU combines attention mechanism and GRU seamlessly.

    Formally:
        ..math: \tilde{{u}}_{t}^{\prime}=a_{t} * {u}_{t}^{\prime} \\
                {h}_{t}^{\prime}=\left(1-\tilde{{u}}_{t}^{\prime}\right) \circ {h}_{t-1}^{\prime}+\tilde{{u}}_{t}^{\prime} \circ \tilde{{h}}_{t}^{\prime}

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iu|W_ih)
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        # (W_hr|W_hu|W_hh)
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        if bias:
            # (b_ir|b_iu|b_ih)
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
            # (b_hr|b_hu|b_hh)
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, input, hidden_output, att_score):
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hidden_output, self.weight_hh, self.bias_hh)
        i_r, i_u, i_h = gi.chunk(3, 1)
        h_r, h_u, h_h = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_u + h_u)
        new_state = torch.tanh(i_h + reset_gate * h_h)

        #att_score = att_score.view(-1, 1)
        att_score = att_score.squeeze(1)
        update_gate = att_score * update_gate
        hy = (1 - update_gate) * hidden_output + update_gate * new_state

        return hy