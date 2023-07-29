import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from utils import evaluate_forecast


def he_init(module):
    """
    He权重初始化
    Args:
        module: 网络模型
    Returns:
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# 定义每5个epoch进行一次验证的函数
def validate_every_n_epoch(model, dataset, criterion, batch_size, device, epoch, validate_interval=5):
    if (epoch + 1) % validate_interval == 0:
        validate_model(model, dataset, criterion, batch_size, device)


def train_model(model, dataset, criterion, optimizer, num_epochs, batch_size, validate_interval, device):
    model.train()
    print('---start train---')
    for epoch in range(num_epochs):
        start_time = time.time()
        for X, Y in tqdm(
                dataset.get_batches(dataset.window_data[0], dataset.window_data[1], batch_size=batch_size, shuffle=True,
                                    device=device
                                    ), total=len(dataset.window_data[0]) // batch_size):
            optimizer.zero_grad()
            Y = Y[:, 0:5]
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

        # 在每5个epoch后进行一次验证
        validate_every_n_epoch(model, dataset, criterion, batch_size, 'cuda', epoch, validate_interval)

        end_time = time.time()
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f},time={end_time - start_time}s")


def validate_model(model, dataset, criterion, batch_size, device):
    print('---valid---')
    model.eval()
    val_outputs_list = []  # 存储每次的val_outputs值
    val_real_list = []  # 存储每次的val_outputs值
    with torch.no_grad():
        for X, Y in dataset.get_batches(dataset.window_data[0], dataset.window_data[1], batch_size=batch_size,
                                        shuffle=True, device=device):
            Y = Y[:, 0:5]
            val_outputs = model(X)
            val_loss = criterion(val_outputs, Y)
            val_outputs_list.append(val_outputs)  # 将val_outputs添加到列表中
            val_real_list.append(Y)
        print(f"Validation Loss: {val_loss.item():.4f}")

    # 将列表转换为张量
    val_outputs_tensor = torch.cat(val_outputs_list, dim=0)
    val_real_tensor = torch.cat(val_real_list, dim=0)

    val_real_data = dataset.denormalize_data(val_real_tensor)
    val_predict_data = dataset.denormalize_data(val_outputs_tensor)
    # 获取评估结果
    mae, mse, rmse = evaluate_forecast(val_real_data, val_predict_data)
    # 打印 mae、mse 和 rmse 数组中的每个元素
    for mae_val, mse_val, rmse_val in zip(mae, mse, rmse):
        print(f'mae={mae_val:.4f}, mse={mse_val:.4f}, rmse={rmse_val:.4f}')


def test_model(model, dataset, criterion, batch_size, device):
    print('---test---')
    model.eval()
    test_predict_list = []
    test_real_list = []
    with torch.no_grad():
        for X, Y in dataset.get_batches(dataset.window_data[0], dataset.window_data[1], batch_size=batch_size,
                                        shuffle=False, device=device):
            test_outputs = model(X)
            Y = Y[:, 0:5]
            test_loss = criterion(test_outputs, Y)
            test_predict_list.append(test_outputs)
            test_real_list.append(Y)
        print(f"Test Loss: {test_loss.item():.4f}")

    test_outputs_tensor = torch.cat(test_predict_list)
    test_real_tensor = torch.cat(test_real_list)

    test_predict_data = dataset.denormalize_data(test_outputs_tensor)
    test_real_data = dataset.denormalize_data(test_real_tensor)

    # 获取评估结果
    mae, mse, rmse = evaluate_forecast(test_predict_data, test_real_data)
    # 打印 mae、mse 和 rmse 数组中的每个元素
    for mae_val, mse_val, rmse_val in zip(mae, mse, rmse):
        print(f'mae={mae_val:.4f}, mse={mse_val:.4f}, rmse={rmse_val:.4f}')
    # 创建图形对象
    fig = plt.figure()
    # 绘制数据
    plt.plot(test_predict_data[-4 * 24:, 1], 'b:', label='Predicted Data')
    plt.plot(test_real_data[-4 * 24:, 1], label='True Real Data')
    # 添加图例
    plt.legend()
    # 添加标题
    plt.title('Comparison of Real and Predicted Data')
    # 显示图形
    plt.show()
    return test_predict_data, test_real_data
