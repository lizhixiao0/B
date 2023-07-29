import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn


class WeatherDataPreprocessor(object):
    def __init__(self, para):
        super().__init__()
        self.file_path = para.file_path
        self.data = None
        self.load_data()
        self.scaler = MinMaxScaler()
        # self.scaler = StandardScaler() # 均值归一或者标准化归一

        self.data, self.raw_data, self.mean, self.max_min = self.preprocess_data()
        self.window_data, self.y_index = self._split_window(self.data, 4 * 24, 1, 1)  # 获得窗口化的数据

    def load_data(self):
        """
        加载数据集
        Returns: 加载到self.data
        """
        # 读取CSV文件
        encodings = ['gbk', 'utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                self.data = pd.read_csv(self.file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        # self.data = pd.read_csv(self.file_path,encoding='utf-8')

    def preprocess_data(self):
        """
        数据预处理
            - 线性插值
            - 日期转换
            - 特征添加
            - 特征选择
            - 归一化
        Returns:
            归一化后的数据  原始数据  每列的均值

        """
        # 使用线性插值填充异常值
        self.data.interpolate(method='linear', inplace=True)

        # 将日期列转换为Datetime类型
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        # 添加新的变量 年份，月份，日，小时
        self.data['year'] = self.data['timestamp'].dt.year - 2020
        self.data['month'] = self.data['timestamp'].dt.month
        self.data['day'] = self.data['timestamp'].dt.day
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['minute'] = self.data['timestamp'].dt.minute
        # 提取特征和目标变量

        features = self.data[
            ['风速（m/15min）', '温度（℃）', '辐照度（W/m2）', '风向', '气压', 'year', 'month', 'day', 'hour']]

        target = self.data[['风速（m/15min）', '温度（℃）', '辐照度（W/m2）', '风向', '气压']]

        # 归一化特征
        feature_columns = ['风速（m/15min）', '温度（℃）', '辐照度（W/m2）', '风向', '气压']
        features, means, max_min = self.mean_normalize(features, feature_columns)

        # 日期特征嵌入
        # d = self.data[['year', 'month', 'day', 'hour']]
        # date_feature = self.Embedded(self.data[['year', 'month', 'day', 'hour']])
        #
        # features['year', 'month', 'day', 'hour'] = date_feature

        # 返回处理后的特征和目标变量作为模型输入
        return features, np.asarray(target), means, max_min

    @staticmethod
    def _split_window(data, window, horizon, window_step):
        """
        滑动时间窗口划分
        Args:
            data: 原始数据
            window: 窗口大小
            horizon: 预测时长
            window_step: 时间跨度，默认1

        Returns: 划分好的数据集
        """
        data = np.array(data)
        row, col = data.shape
        start_idx = np.arange(0, row - window - horizon + 1, window_step)
        end_idx = start_idx[:, np.newaxis] + np.arange(window)

        X = data[end_idx]
        Y = data[start_idx + window + horizon - 1]  # Fix the indexing for Y
        y_index = start_idx + window

        return [X, Y], y_index

    @staticmethod
    def mean_normalize(data, feature_columns):
        """
        均值归一化函数  x-mean  / max-min
        Args:
            data (pd.DataFrame): 需要进行均值归一化的数据集。
            feature_columns: 要进行归一化的特征名称
        Returns:
            data: 均值归一化后的数据集。
            mean: 包含每个特征的均值，以便用于后续的还原。
        """

        means = data[feature_columns].mean()  # 计算前6个特征的均值
        imax = data[feature_columns].max()
        imin = data[feature_columns].min()
        max_min = imax - imin
        normalized_data = (data[feature_columns] - means) / (
                data[feature_columns].max() - data[feature_columns].min())  # 均值归一化
        data[feature_columns] = normalized_data  # 更新原始数据集中的前6个特征

        return data, means, max_min

    def denormalize_data(self, normalized_data):
        """
        均值还原函数
        Args:
            normalized_data : 均值归一化后的数据集。
        Returns:
            denormalized_data: 还原后的数据集。
        """
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.cpu().detach().numpy()
        means = self.mean.values
        max_min = self.max_min.values
        raw_data = normalized_data * max_min + means
        # _, length = normalized_data.shape
        # raw_data = np.ones((_, length))
        # for i in range(length):
        #     imax = np.max(normalized_data[:, i])
        #     imin = np.min(normalized_data[:, i])
        #     raw_data[:, i] = normalized_data[:, i] * (imax - imin) + means[i]

        return raw_data

    @staticmethod
    def get_batches(inputs, targets, batch_size, shuffle=False, device='cpu'):
        """
        获取batch数据的函数，类似DataLoader里封装的函数，这里不过是自己手写的。
        :param inputs: X 数据集的输入
        :param targets: Y 数据集的labels
        :param batch_size: batch_size 大小
        :param shuffle: True用于打乱数据集，每次都会以不同的顺序返回
        :param device: 使用设备
        :return: 返回batch块大小的数据
        """
        inputs = torch.from_numpy(inputs).to(torch.float32)
        targets = torch.from_numpy(targets).to(torch.float32)
        length = len(inputs)  # 输入集的长度
        if shuffle:
            index = torch.randperm(length)  # torch.randperm(n)：将0~n-1（包括0和n-1）随机打乱后获得的数字序列，函数名是random permutation缩写
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length - batch_size:
            end_idx = min(length, start_idx + batch_size)  # 取一个batch块的数据，到最后不够了就取到最后一个值
            excerpt = index[start_idx:end_idx]  # 一个batch大小的数据
            X = inputs[excerpt]
            Y = targets[excerpt]
            # if cuda:
            X = X.to(device)
            Y = Y.to(device)
            yield X, Y
            start_idx += batch_size
