class Config:
    data_path = '../data/' # 数据读取路径
    save_log_dir = 'log/' # 存放日志
    save_res_dir = 'result/' # 存放预测结果及loss变化
    save_param_dir = 'params/' # 存放参数
    save_plot_dir = 'plot/' # 存放绘制的图片
    save_record_dir = 'record/' # 存放每个州训练过程结束后的loss，包括mse和smoothl1loss
    device = 'cpu:0' # 没有gpu可以选择cpu:0

    seed = 1

    model_name = 'MLP' # LR,RNN,MLP

    # for LSTM(RNN)
    look_back = 2 # 以多少个小时作为特征，例如1,2,3,4，如果look_back为2，则1，2为特征，3为标签；下一次则是2，3为特征，4为标签，以此类推
    hidden_dims = 8 # LSTM隐藏层
    bi_lstm = 2 # 2个LSTM串联，第二个LSTM接收第一个的计算结果

    neuron_nums = [32, 16] # MLP的神经网络结构
    dropout_rate = 0.2 # MLP dropout的比例

    epochs = 500 # 训练轮次
    batch_size = 128 # 批次大小
    print_interval = 10 # 每print_interval打印
    lr = 3e-4 # 学习率
    weight_decay = 1e-5 # 正则项
    num_workers = 8 # 如果cpu只有4核则改为4

    plot_gap = 50 # 图片打印间隔，因为原始数据又8000多条，因此每隔plot_gap取一个点作为画图的数据