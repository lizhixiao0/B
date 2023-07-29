import argparse

import numpy as np
import torch
from torch import nn, optim

from BNet import BNet
from dataset import WeatherDataPreprocessor
from model import WeatherPredict
from run import train_model, validate_model, test_model, he_init

parser = argparse.ArgumentParser(description=' All params')
# ----dataset
parser.add_argument('--file_path', type=str,
                    default=r'E:\lin\DeepLearning\02\附件2数据.csv',
                    help='raw data file')  # required=True,

parser.add_argument('--horizon', type=int, default=1, metavar='N', help='horizon')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size (default: 32)')
parser.add_argument('--window', type=int, default=4 * 24, metavar='N', help='window_step')
parser.add_argument('--window_step', type=int, default=1, metavar='N', help='window_step')

para = parser.parse_known_args()[0]

dataset = WeatherDataPreprocessor(para)
device = 'cuda'
# model = WeatherPredict(input_dim=50, output_dim=5, max_seq_len=7 * 24, nhead=2, num_encoder_layers=4,
# #                        num_decoder_layers=2)
# model = BNet()
# model.apply(he_init)
model = torch.load('2_b_entire_model.pth')
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

model = model.to(device)
criterion = criterion.to(device)

# 训练模型
num_epochs = 30
batch_size = para.batch_size
# train_model(model, dataset, criterion, optimizer, num_epochs, batch_size, device=device, validate_interval=5)
# torch.save(model, '2_b_entire_model.pth')

# 在验证集上进行预测
# validate_model(model, dataset, criterion, batch_size, device)

# 在测试集上进行预测并绘制图形
# test_predict_data, test_real_data = test_model(model, dataset, criterion, batch_size, device)

date_feature = torch.from_numpy(dataset.data.iloc[0:4 * 24 * 10, 5:9].values).cuda().to(torch.float32)

test_data = dataset.data.values[-4 * 24:, :]
test_data = torch.from_numpy(test_data)
test_data = test_data.unsqueeze(0)
test_data = test_data.cuda()
test_data = test_data.to(torch.float32)

# 滚动预测
predictions = []

# 设置滚动窗口的大小（假设窗口大小为3）
window_size = 1

for i in range(4 * 24 * 3):  # 一共进行4 * 24次预测
    # 预测当前时间步的结果
    predict = model(test_data)

    # 将预测结果与新的日期特征拼接

    new_data = torch.cat([predict, date_feature[i:i + 1, :]], dim=1)

    # 更新测试数据，去掉滚动窗口的第一个时间步，添加新的数据
    test_data = torch.cat([test_data[:, 1:, :], new_data.unsqueeze(0)], dim=1)

    # 将当前时间步的预测结果添加到预测列表中
    predictions.append(predict)

# 将所有预测结果拼接成一个张量
predictions = torch.cat(predictions, dim=0)

# 转为预测值
predictions = dataset.denormalize_data(predictions)
# 将预测结果转换回 numpy 数组
# predictions = predictions.cpu().detach().numpy()

print(predictions)
