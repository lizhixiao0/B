import math

import pandas as pd
import torch
import torch.nn as nn


class WeatherPredict(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, nhead=8, num_encoder_layers=4, num_decoder_layers=2):
        super(WeatherPredict, self).__init__()
        self.date_embedded = WeatherDateEmbedded().cuda()
        self.position_encoder = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True) for _ in
             range(num_encoder_layers)])

        self.decoder = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True) for _ in
             range(num_decoder_layers)])

        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        src = self.date_embedded(src)
        tgt = src  # 因为输入是一样的
        src = self.position_encoder(src)
        tgt = src

        memory = self.encoder(src)  # Transpose to (seq_len, batch_size, d_model)
        output = self.decoder(tgt, memory)  # Transpose to (seq_len, batch_size, d_model)

        output = self.fc(output)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()


class WeatherDateEmbedded(nn.Module):
    def __init__(self):
        super().__init__()
        # 日期特征编码
        self.year_embedding = nn.Embedding(2, 1)
        self.month_embedding = nn.Embedding(13, 6)
        self.day_embedding = nn.Embedding(31, 12)
        self.hour_embedding = nn.Embedding(24, 10)

    def forward(self, data):
        raw_data = data
        data = torch.tensor(data[:, :, -4:], dtype=torch.long)

        year_data = data[:, :, 0]
        month_data = data[:, :, 1]
        day_data = data[:, :, 2]
        hour_data = data[:, :, 3]

        # Pass each component to its corresponding embedding layer
        embedded_year = self.year_embedding(year_data)
        embedded_month = self.month_embedding(month_data)
        embedded_day = self.day_embedding(day_data)
        embedded_hour = self.hour_embedding(hour_data)
        encoded_features = torch.cat([embedded_year, embedded_month, embedded_day, embedded_hour], dim=1)
        # Convert the output Tensor back to a NumPy array
        encoded_features_np = torch.cat([raw_data, encoded_features.detach()], dim=2)
        return encoded_features_np
