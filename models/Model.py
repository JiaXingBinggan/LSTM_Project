import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import numpy as np

def weight_init(layers):
    for layer in layers:
        if isinstance(layer, nn.BatchNorm1d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif isinstance(layer, nn.Linear):
            n = layer.in_features
            y = 1.0 / np.sqrt(n)
            layer.weight.data.uniform_(-y, y)
            layer.bias.data.fill_(0)
            # nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')

# 传统的预测点击率模型
class LR(nn.Module):
    def __init__(self,
                 feature_nums,
                 output_dim = 1):
        super(LR, self).__init__()
        self.linear = nn.Linear(feature_nums, output_dim)

        self.bias = nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        out = self.bias + torch.sum(self.linear(x), dim=1)

        return out.unsqueeze(1)


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

class MLP(nn.Module):
    def __init__(self,
                 feature_nums,
                 neuron_nums,
<<<<<<< HEAD
                 dropout_rate,
=======
>>>>>>> 78018e8c3cfd0b3d1c83041e1e43ffc855a7f79c
                 output_dim=1):
        super(MLP, self).__init__()
        self.feature_nums = feature_nums
        self.neuron_nums = neuron_nums
<<<<<<< HEAD
        self.dropout_rate = dropout_rate
=======
>>>>>>> 78018e8c3cfd0b3d1c83041e1e43ffc855a7f79c

        deep_input_dims = self.feature_nums

        layers = list()

        neuron_nums = self.neuron_nums
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            # layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_num

        weight_init(layers)

<<<<<<< HEAD
        layers.append(nn.Linear(deep_input_dims, output_dim))
=======
        layers.append(nn.Linear(deep_input_dims, 1))
>>>>>>> 78018e8c3cfd0b3d1c83041e1e43ffc855a7f79c

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
        :return: pctrs
        """
        out = self.mlp(x)

        return out