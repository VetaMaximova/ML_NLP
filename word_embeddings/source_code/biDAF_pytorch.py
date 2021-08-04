import os, copy
import json
import nltk
import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import time
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
import utils as utils


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x
    
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=False, dropout=0.2):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.reset_params()
        #During training, randomly zeroes some of the elements of the input tensor
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}'))
            nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}'), val=0)
            getattr(self.rnn, f'bias_hh_l{i}').chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}_reverse'))
                nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}_reverse'), val=0)
                getattr(self.rnn, f'bias_hh_l{i}_reverse').chunk(4)[1].fill_(1)

    def forward(self, x):
        x, x_len = x
        x = self.dropout(x)

        x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = x.index_select(dim=0, index=x_idx)
        _, x_ori_idx = torch.sort(x_idx)

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
        x_packed, (h, c) = self.rnn(x_packed)

        x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)[0]
        x = x.index_select(dim=0, index=x_ori_idx)
        h = h.permute(1, 0, 2).contiguous().view(-1, h.size(0) * h.size(2)).squeeze()
        h = h.index_select(dim=0, index=x_ori_idx)

        return x, h
    
class BiDAF(nn.Module):
    def __init__(self, pretrained, word_dim, dropout = 0.2):
        super(BiDAF, self).__init__()
        
        # 1. Word Embedding Layer
        # initialize word embedding with GloVe
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=False)  #does not get updated (?)
        
        self.hidden_size = word_dim
         # 2. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=word_dim,
                                 hidden_size=self.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=dropout)
        # 3. Attention Flow Layer
        self.att_weight_c = Linear(self.hidden_size * 2, 1)
        self.att_weight_q = Linear(self.hidden_size * 2, 1)
        self.att_weight_cq = Linear(self.hidden_size * 2, 1)

        # 4. Modeling Layer
        self.modeling_LSTM = LSTM(input_size=self.hidden_size * 8,
                                   hidden_size=self.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=dropout)

        # 5. Output Layer
        self.p1_weight_g = Linear(self.hidden_size * 8, 1, dropout=dropout)
        self.p1_weight_m = Linear(self.hidden_size * 2, 1, dropout=dropout)
        self.p2_weight_g = Linear(self.hidden_size * 8, 1, dropout=dropout)
        self.p2_weight_m = Linear(self.hidden_size * 2, 1, dropout=dropout)

        self.output_LSTM = LSTM(input_size=self.hidden_size * 2,
                                hidden_size=self.hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)
        
    def att_flow_layer(self, c, q):
        """
        :param c: (batch, c_len, hidden_size * 2)
        :param q: (batch, q_len, hidden_size * 2)
        :return: (batch, c_len, q_len)
        """
        c_len = c.size(1)
        q_len = q.size(1)

        cq = []
        for i in range(q_len):
            #(batch, 1, hidden_size * 2)
            qi = q.select(1, i).unsqueeze(1)
            #(batch, c_len, 1)
            ci = self.att_weight_cq(c * qi).squeeze()
            cq.append(ci)
        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)

        # (batch, c_len, q_len)
        s = self.att_weight_c(c).expand(-1, -1, q_len) + \
            self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq

        # (batch, c_len, q_len)
        a = F.softmax(s, dim=2)
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
        c2q_att = torch.bmm(a, q)
        # (batch, 1, c_len)
        b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
        # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
        q2c_att = torch.bmm(b, c).squeeze()
        # (batch, c_len, hidden_size * 2) (tiled)
        q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
        # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

        # (batch, c_len, hidden_size * 8)
        x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
        return x

    def output_layer(self, g, m, l):
        """
        :param g: (batch, c_len, hidden_size * 8)
        :param m: (batch, c_len ,hidden_size * 2)
        :return: p1: (batch, c_len), p2: (batch, c_len)
        """
        # (batch, c_len)
        p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
        # (batch, c_len, hidden_size * 2)
        m2 = self.output_LSTM((m, l))[0]
        # (batch, c_len)
        p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()

        return p1, p2
        
    def forward(self, batch):
        # 1. Word Embedding Layer
        c_word = self.word_emb(batch.c_word[0])
        q_word = self.word_emb(batch.q_word[0])
        c_lens = batch.c_word[1]
        q_lens = batch.q_word[1]

        # 2. Contextual Embedding Layer
        c = self.context_LSTM((c_word, c_lens))[0]
        q = self.context_LSTM((q_word, q_lens))[0]
        # 3. Attention Flow Layer
        g = self.att_flow_layer(c, q)
        # 4. Modeling Layer
        m = self.modeling_LSTM((g, c_lens))[0]
        # 5. Output Layer
        p1, p2 = self.output_layer(g, m, c_lens)

        # (batch, c_len), (batch, c_len)
        return p1, p2
    
    
    
class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
        
        
