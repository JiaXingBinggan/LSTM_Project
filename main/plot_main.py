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

    return np.asarray(data_x).reshape(-1, 1, look_back), np.asarray(data_y).reshape(-1, 1, 1), scalar


def mk_dir(dir_name, model_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    dir_path = dir_name + model_name + '/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

# 用于预训练传统预测点击率模型
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=Config.data_path)
    parser.add_argument('--dataset_name', default='ele_loads.csv', help='dataset')
    parser.add_argument('--model_name', default=Config.model_name, help='LR, RNN, MLP')
    parser.add_argument('--look_back', default=Config.look_back, help='以几行数据为特征维度数量')

    parser.add_argument('--plot_gap', type=int, default=Config.plot_gap)
    parser.add_argument('--loss_type', type=str, default='mse', help='smoothl1loss')
    parser.add_argument('--save_log_dir', default=Config.save_log_dir)
    parser.add_argument('--save_plot_dir', default=Config.save_plot_dir)
    parser.add_argument('--save_record_dir', default=Config.save_record_dir)
    parser.add_argument('--save_res_dir', default=Config.save_res_dir)

    args = parser.parse_args()

    # 设置随机数种子
    setup_seed(Config.seed)

    # 创建文件夹
    mk_dir(args.save_plot_dir, args.model_name)
    save_plot_dir = args.save_plot_dir + args.model_name + '/'

    mk_dir(args.save_record_dir, args.model_name)
    save_record_dir = args.save_record_dir + args.model_name + '/'

    logging.basicConfig(level=logging.DEBUG,
                        filename=args.save_log_dir + args.model_name + '_plot.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.info('===> current model is {}!'.format(args.model_name))
    logger.info('===> start ploting!  ')
    current_city_output_dicts = dict.fromkeys(tuple('AEP,AP,ATSI,DAY,DEOK,DOM,DUQ,EKPC,MIDATL,NI'.split(',')))

    for city_name in current_city_output_dicts.keys():
        logger.info('===> now plot the figures for the city {}  '.format(city_name))
        args.city_name = city_name

        data_x, data_y, scalar = get_dataset(args.data_path, args.dataset_name, args.city_name, args.look_back) # 获取当前州的数据

        for loss_type in ['mse', 'smoothl1loss']:
            args.loss_type = loss_type
            preds = pd.read_csv(args.save_res_dir + 'ele_' + args.loss_type + '_preds_' + args.model_name + '.csv')

            # 绘图
            plt.plot(preds[[args.city_name]].values.flatten()[::args.plot_gap], 'r', label='prediction')
            plt.plot(np.multiply(data_y.flatten()[::args.plot_gap], scalar), 'b', label='real')
            plt.legend(loc='best')
            plt.savefig(save_plot_dir + args.city_name + '_' + args.loss_type + '.jpg')
            # plt.show()
            plt.close()

    # 取各个州训练的最后一轮loss
    logger.info('===> now record the lowest loss for all cities')
    for loss_type in ['mse', 'smoothl1loss']:
        args.loss_type = loss_type
        losses = pd.read_csv(args.save_res_dir + 'ele_' + args.loss_type + '_losses_' + args.model_name + '.csv')
        losses.iloc[-1, :].to_csv(save_record_dir + args.loss_type + '_' + args.model_name + '.csv')