import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import argparse
import random
import models.Model as Model
import models.create_data as Data
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(model_name, feature_nums, hidden_dims, bi_lstm):
    if model_name == 'LR':
        return Model.LR(feature_nums)
    elif model_name == 'RNN':
        return Model.RNN(feature_nums, hidden_dims, bi_lstm)

def get_dataset(data_path, dataset_name, city_name, look_back):
    '''
    :param data_path: 数据根目录
    :param dataset_name: 数据名称
    :param city_name: 城市名称
    :param look_back: 为几行数据作为为特征维度数量
    :return:
    '''
    # AEP,AP,ATSI,DAY,DEOK,DOM,DUQ,EKPC,MIDATL,NI
    data_path = data_path + dataset_name
    datas = pd.read_csv(data_path)[[city_name]].values # 取出对应city的数据

    # 归一化
    max_value = np.max(datas)
    min_value = np.min(datas)
    scalar = max_value - min_value
    datas = list(map(lambda x: x / scalar, datas))

    data_x = [] # 特征
    data_y = [] # 标签
    for i in range(len(datas) - look_back):
        data_x.append(datas[i:i + look_back])
        data_y.append(datas[i + look_back])

    return np.asarray(data_x).reshape(-1, 1, look_back), np.asarray(data_y).reshape(-1, 1, 1)


def train(model, optimizer, data_loader, loss, device):
    model.train()  # 转换为训练模式
    total_loss = 0
    log_intervals = 0
    for features, labels in data_loader:
        features, labels = features.float().to(device), labels.to(device)
        y = model(features)
        train_loss = loss(y, labels.float())

        model.zero_grad()
        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.item()  # 取张量tensor里的标量值，如果直接返回train_loss很可能会造成GPU out of memory

        log_intervals += 1

    return total_loss / log_intervals


def test(model, data_loader, loss, device):
    model.eval()
    targets, predicts = list(), list()
    intervals = 0
    total_test_loss = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.float().to(device), labels.to(device)
            y = model(features)

            test_loss = loss(y, labels.float())
            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())
            intervals += 1
            total_test_loss += test_loss.item()

    return total_test_loss / intervals

def sub(model, data_loader, loss, device):
    model.eval()
    predicts = list()
    intervals = 0
    total_test_loss = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.float().to(device), labels.to(device)
            y = model(features)

            test_loss = loss(y, labels.float())
            predicts.extend(y.tolist())
            intervals += 1
            total_test_loss += test_loss.item()

    return total_test_loss / intervals, predicts

def main(args):
    device = torch.device(args.device)  # 指定运行设备

    data_x, data_y = get_dataset(args.data_path, args.dataset_name, args.city_name, args.look_back)

    train_dataset = Data.generate_dataset(data_x, data_y)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    feature_nums = args.look_back
    model = get_model(args.model_name, feature_nums, args.hidden_dims, args.bi_lstm).to(device)

    loss = nn.MSELoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_time = datetime.datetime.now()
    for epoch_i in range(args.epochs):
        torch.cuda.empty_cache()  # 清理无用的cuda中间变量缓存

        train_average_loss = train(model, optimizer, train_data_loader, loss, device)

        train_end_time = datetime.datetime.now()

        if epoch_i % args.print_interval == 0:
            print('epoch:', epoch_i, 'training average loss:', train_average_loss, '[{}s]'.format((train_end_time - start_time).seconds))

    valid_dataset = Data.generate_dataset(data_x, data_y)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # 验证集submission
    valid_loss, preds = sub(model, valid_data_loader, loss, device)
    final_preds = np.asarray(preds)

    plt.plot(final_preds.flatten(), 'r', label='prediction')
    plt.plot(data_y.flatten(), 'b', label='real')
    plt.legend(loc='best')
    plt.show()

# 用于预训练传统预测点击率模型
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/')
    parser.add_argument('--dataset_name', default='ele_loads.csv', help='dataset')
    parser.add_argument('--model_name', default='RNN', help='LR, RNN')
    parser.add_argument('--num_workers', default=4, help='4, 8, 16, 32')
    parser.add_argument('--feature_nums', default=2)
    parser.add_argument('--hidden_dims', default=8)
    parser.add_argument('--bi_lstm', default=2, help='1, 2')
    parser.add_argument('--look_back', default=2, help='以几行数据为特征维度数量')
    parser.add_argument('--city_name', default='AEP', help='AEP,AP,ATSI,DAY,DEOK,DOM,DUQ,EKPC,MIDATL,NI')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--print_interval', type=int, default=10)
    parser.add_argument('--device', default='cpu:0')

    args = parser.parse_args()

    # 设置随机数种子
    setup_seed(1)

    main(args)