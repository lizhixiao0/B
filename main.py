import argparse

import torch
from torch import nn, optim

from dataset import WeatherDataPreprocessor
from model import WeatherPredict
from run import train_model, validate_model, test_model

parser = argparse.ArgumentParser(description=' All params')
# ----dataset
parser.add_argument('--file_path', type=str,
                    default=r'E:\lin\DeepLearning\02\附件2.csv',
                    help='raw data file')  # required=True,

parser.add_argument('--horizon', type=int, default=1, metavar='N', help='horizon')
parser.add_argument('--batch_size', type=int, default=2, metavar='N', help='batch size (default: 32)')
parser.add_argument('--window', type=int, default=7 , metavar='N', help='window_step')
parser.add_argument('--window_step', type=int, default=1, metavar='N', help='window_step')

para = parser.parse_known_args()[0]

dataset = WeatherDataPreprocessor(para)
device='cuda'
model = WeatherPredict( input_dim=32, output_dim=6, max_seq_len=7*24,nhead=2, num_encoder_layers=2, num_decoder_layers=2)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

model = model.cuda()
criterion = criterion.cuda()

# 训练模型
num_epochs = 2
batch_size = para.batch_size
train_model(model, dataset, criterion, optimizer, num_epochs, batch_size, device)

# 在验证集上进行预测
validate_model(model, dataset, criterion, batch_size, device)

# 在测试集上进行预测并绘制图形
test_model(model, dataset, criterion, batch_size, device)
