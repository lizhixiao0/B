import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm


def evaluate_predictions(predictions, targets):
    """
    计算预测结果与真实值之间的评估指标：RMSE、MSE、MAE

    Args:
        predictions: 预测结果张量
        targets: 真实值张量

    Returns:
        rmse: 均方根误差
        mse: 均方误差
        mae: 平均绝对误差
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()

    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    return rmse, mse, mae


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


def train_model(model, dataset, criterion, optimizer, num_epochs, batch_size, validate_interval):
    model.apply(he_init)

    print('---start train---')
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        for X, Y in tqdm(
                dataset.get_batches(dataset.window_data[0], dataset.window_data[1], batch_size=batch_size, shuffle=True,
                                    ), total=len(dataset.window_data[0]) // batch_size):
            optimizer.zero_grad()
            Y = Y[:, 0:6]
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
        for X, Y in dataset.get_batches(dataset.val_data, dataset.val_labels, batch_size=batch_size, shuffle=False,
                                        device=device):
            val_outputs = model(X)
            val_loss = criterion(val_outputs, Y)
            val_outputs_list.append(val_outputs)  # 将val_outputs添加到列表中
            val_real_list.append(Y)
        print(f"Validation Loss: {val_loss.item():.4f}")

    # 将列表转换为张量
    val_outputs_tensor = torch.cat(val_outputs_list, dim=0)
    val_real_tensor = torch.cat(val_real_list, dim=0)
    val_real_data = dataset.inverse_normalize_data(val_real_tensor)
    val_predict_data = dataset.inverse_normalize_data(val_outputs_tensor)
    print('Valid---rmse=%.4f, mse=%.4f, mae=%.4f' % evaluate_predictions(val_predict_data, val_real_data))


def test_model(model, dataset, criterion, batch_size, device):
    print('---test---')
    model.eval()
    test_predict_list = []
    test_real_list = []
    with torch.no_grad():
        for X, Y in dataset.get_batches(dataset.test_data, dataset.test_labels, batch_size=batch_size, shuffle=False,
                                        device=device):
            test_outputs = model(X)
            test_loss = criterion(test_outputs, Y)
            test_predict_list.append(test_outputs)
            test_real_list.append(Y)
        print(f"Test Loss: {test_loss.item():.4f}")

    test_outputs_tensor = torch.cat(test_predict_list)
    test_real_tensor = torch.cat(test_real_list)

    test_predict_data = dataset.inverse_normalize_data(test_outputs_tensor)
    test_real_data = dataset.inverse_normalize_data(test_real_tensor)
    print('rmse=%.4f, mse=%.4f, mae=%.4f' % evaluate_predictions(test_predict_data, test_real_data))

    real_data = dataset.raw_test_y[:test_predict_data.size]
    # 创建图形对象
    fig = plt.figure()
    # 绘制数据
    plt.plot(test_real_data[-4 * 24:], 'r-', label='Real Data')
    plt.plot(test_predict_data[-4 * 24:], 'b:', label='Predicted Data')
    plt.plot(real_data[-4 * 24:], label='True Real Data')
    # 添加图例
    plt.legend()
    # 添加标题
    plt.title('Comparison of Real and Predicted Data')
    # 显示图形
    plt.show()
