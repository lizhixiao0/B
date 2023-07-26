import pandas as pd
import torch
import torch.nn as nn


class WeatherPredict(nn.Module):
    def __init__(self):
        super().__init__()

        def forward(self):
            pass


class WeatherDateEmbedded(nn.Module):
    def __init__(self):
        super().__init__()
        # 日期特征编码
        self.year_embedding = nn.Embedding(2, 2)
        self.month_embedding = nn.Embedding(12, 6)
        self.day_embedding = nn.Embedding(31, 12)
        self.hour_embedding = nn.Embedding(24, 10)

    def forward(self, data):
        data = torch.tensor(data.value, dtype=torch.float)
        embedded_year = self.year_embedding(data[:, 0])
        embedded_month = self.month_embedding(data[:, 1])
        embedded_day = self.day_embedding(data[:, 2])
        embedded_hour = self.hour_embedding(data[:, 3])
        encoded_features = torch.cat([embedded_year, embedded_month, embedded_day, embedded_hour], dim=1)
        # Convert the output Tensor back to a NumPy array
        encoded_features_np = encoded_features.detach().numpy()

        return encoded_features
