import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import numpy as np

# 传统的预测点击率模型
class LR(nn.Module):
    def __init__(self,
                 feature_nums,
                 output_dim = 1):
        super(LR, self).__init__()
        self.linear = nn.Embedding(feature_nums, output_dim)

        self.bias = nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        out = self.bias + torch.sum(self.linear(x), dim=1)
        pctrs = torch.sigmoid(out)

        return pctrs


class RNN(nn.Module):
    def __init__(self,
                 feature_nums,
                 hidden_dims,
                 bi_lstm,
                 out_dims=1):
        super(RNN, self).__init__()
        self.feature_nums = feature_nums # 输入数据特征维度
        self.hidden_dims = hidden_dims # 隐藏层维度
        self.bi_lism = bi_lstm # LSTM串联数量

        self.lstm = nn.LSTM(self.feature_nums, self.hidden_dims, self.bi_lism)
        self.out = nn.Linear(self.hidden_dims, out_dims)

    def forward(self,x):
        x1, _ = self.lstm(x)
        a, b, c = x1.shape
        out = self.out(x1.view(-1, c))
        out1 = out.view(a, b, -1)

        return out1