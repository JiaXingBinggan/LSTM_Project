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

import logging
import sys

from config import Config

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(model_name, feature_nums, hidden_dims, bi_lstm, neuron_nums, dropout_rate):
    '''
        获取模型
        :param model_name: 模型名
        :param feature_nums: 特征数量，其等于look_back
        :param hidden_dims:  LSTM隐藏层数量
        :param bi_lstm: 是否使用双向LSTM
        :param neuron_nums: MLP的神经网络结构
        :return:
    '''
    if model_name == 'LR':
        return Model.LR(args.feature_nums)
    if model_name == 'MLP':
        return Model.MLP(feature_nums, neuron_nums, dropout_rate)
    elif model_name == 'RNN':
        return Model.RNN(feature_nums, hidden_dims, bi_lstm)

def get_dataset(data_path, dataset_name, city_name, look_back):
    '''
        读取数据
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

    return np.asarray(data_x).reshape(-1, 1, look_back), np.asarray(data_y).reshape(-1, 1, 1), scalar


def train(model, optimizer, data_loader, loss, device):
    '''
        训练函数
        :param model: pytorch模型类
        :param optimizer: 优化器
        :param data_loader: pytorch数据加载器
        :param loss: 损失
        :param device: 采用cpu还是gpu
        :return:
    '''
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


def sub(model, data_loader, loss, device):
    '''
        评估函数
        :param model: pytorch模型类
        :param data_loader: pytorch数据加载器
        :param loss: 损失
        :param device: 采用cpu还是gpu
        :return:
    '''
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

    return total_test_loss / intervals, np.asarray(predicts).flatten()

def main(args, logger):
    '''
        主函数
        :param args: 超参定义器
        :param logger: 日志句柄
        :return:
    '''
    device = torch.device(args.device)  # 指定运行设备

    data_x, data_y, scalar = get_dataset(args.data_path, args.dataset_name, args.city_name, args.look_back)

    '''
    写个划分训练集和测试集的方法，但我不建议这么做，因为会影响时序。最好还是利用新的测试集
    '''
    # train_slice = int(data_x.shape[0] * 0.6) # 6 4划分训练集
    # train_dataset = Data.generate_dataset(data_x[:train_slice, :], data_y[:train_slice, :])
    # train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    # test_dataset = Data.generate_dataset(data_x[train_slice:, :], data_y[train_slice:, :])
    # test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
    #                                                 num_workers=args.num_workers)

    train_dataset = Data.generate_dataset(data_x, data_y)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    feature_nums = args.look_back
    model = get_model(args.model_name, feature_nums, args.hidden_dims, args.bi_lstm, args.neuron_nums, args.dropout_rate).to(device)
    logger.info(model)

    if args.loss_type == 'mse':
        loss = nn.MSELoss()
    elif args.loss_type == 'smoothl1loss':
        loss = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_time = datetime.datetime.now()
    train_epoch_loss = []
    for epoch_i in range(args.epochs):
        torch.cuda.empty_cache()  # 清理无用的cuda中间变量缓存

        train_average_loss = train(model, optimizer, train_data_loader, loss, device) # 训练
        train_epoch_loss.append(train_average_loss)

        train_end_time = datetime.datetime.now()

        if epoch_i % args.print_interval == 0: # 打印结果
            logger.info('City {}, model {}, epoch {}, train_{}_loss {}, '
                        '[{}s]'.format(args.city_name, args.model_name, epoch_i,
                                         args.loss_type, train_average_loss, (train_end_time - start_time).seconds))

    torch.save(model.state_dict(), os.path.join(args.save_param_dir,
                                                args.model_name + '_best_' + args.loss_type + '.pth')) # 存储参数

    valid_dataset = Data.generate_dataset(data_x, data_y)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # submission
    valid_loss, preds = sub(model, valid_data_loader, loss, device)

    '''
        如果从训练集中划分了测试集，这里可以改为如下形式，test_data_loader为上面第146行定义的变量
    '''
    # valid_loss, preds = sub(model, test_data_loader, loss, device)

    output_dict = {}
    output_dict.setdefault(args.loss_type + '_loss', train_epoch_loss)
    output_dict.setdefault(args.loss_type + '_preds', (preds * scalar).tolist())

    return output_dict


# 用于预训练传统预测点击率模型
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=Config.data_path)
    parser.add_argument('--dataset_name', default='ele_loads.csv', help='dataset')
    parser.add_argument('--model_name', default=Config.model_name, help='LR, RNN, MLP')
    parser.add_argument('--neuron_nums', type=list, default=Config.neuron_nums)
    parser.add_argument('--dropout_rate', type=float, default=Config.dropout_rate)
    parser.add_argument('--num_workers', default=Config.num_workers, help='4, 8, 16, 32')
    parser.add_argument('--hidden_dims', default=Config.hidden_dims)
    parser.add_argument('--bi_lstm', default=Config.bi_lstm, help='1, 2')
    parser.add_argument('--look_back', default=Config.look_back, help='以几行数据为特征维度数量')
    parser.add_argument('--city_name', default='AEP', help='AEP,AP,ATSI,DAY,DEOK,DOM,DUQ,EKPC,MIDATL,NI')
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--lr', type=float, default=Config.lr)
    parser.add_argument('--weight_decay', type=float, default=Config.weight_decay)
    parser.add_argument('--batch_size', type=int, default=Config.batch_size)
    parser.add_argument('--print_interval', type=int, default=Config.print_interval)
    parser.add_argument('--device', default=Config.device)

    parser.add_argument('--loss_type', type=str, default='mse', help='smoothl1loss')
    parser.add_argument('--save_log_dir', default=Config.save_log_dir)
    parser.add_argument('--save_res_dir', default=Config.save_res_dir)
    parser.add_argument('--save_param_dir', default=Config.save_param_dir)

    args = parser.parse_args()

    # 设置随机数种子
    setup_seed(Config.seed)

    if not os.path.exists(args.save_log_dir):
        os.mkdir(args.save_log_dir)

    if not os.path.exists(args.save_res_dir):
        os.mkdir(args.save_res_dir)

    if not os.path.exists(args.save_param_dir):
        os.mkdir(args.save_param_dir)

    logging.basicConfig(level=logging.DEBUG,
                        filename=args.save_log_dir + args.model_name + '_output.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.info('===> start training!  ')
    # 将各个州的数据保存到字典中
    current_city_output_dicts = dict.fromkeys(tuple('AEP,AP,ATSI,DAY,DEOK,DOM,DUQ,EKPC,MIDATL,NI'.split(',')))
    for city_name in current_city_output_dicts.keys():
        logger.info('===> now excuate the city {}  '.format(city_name))
        args.city_name = city_name
        current_city_output_dict = {}
        for loss_type in ['mse', 'smoothl1loss']:
            args.loss_type = loss_type
            output_dict = main(args, logger) # 主运行函数
            for key in output_dict.keys():
                current_city_output_dict.setdefault(key, output_dict[key])
        current_city_output_dicts[city_name] = current_city_output_dict

    # 存储loss和preds
    for loss_type in ['mse', 'smoothl1loss']:
        city_preds = {}
        city_losses = {}
        for city_name in current_city_output_dicts.keys():
            current_city_output_loss = current_city_output_dicts[city_name][loss_type + '_loss']
            current_city_output_pred = current_city_output_dicts[city_name][loss_type + '_preds']

            city_losses.setdefault(city_name, current_city_output_loss)
            city_preds.setdefault(city_name, current_city_output_pred)

        city_loss_df = pd.DataFrame(data=city_losses)
        city_loss_df.to_csv(args.save_res_dir + '/ele_' + loss_type + '_losses_' + args.model_name + '.csv', index=None)

        city_pred_df = pd.DataFrame(data=city_preds)
        city_pred_df.to_csv(args.save_res_dir + '/ele_' + loss_type + '_preds_' + args.model_name + '.csv', index=None)


