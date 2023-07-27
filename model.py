import math

import torch
import torch.nn as nn


class WeatherPredict(nn.Module):
    def __init__(self, input_dim, output_dim=6, max_seq_len=7 * 24, nhead=2, num_encoder_layers=4,
                 num_decoder_layers=2):
        super(WeatherPredict, self).__init__()

        self.year_embedding = nn.Embedding(3, 1)
        self.month_embedding = nn.Embedding(13, 5)
        self.day_embedding = nn.Embedding(31, 10)
        self.hour_embedding = nn.Embedding(24, 10)

        self.position_encoder = PositionalEncoding(input_dim, max_seq_len)
        self.encoder = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=32, nhead=nhead, dim_feedforward=128, batch_first=True) for _ in
             range(num_encoder_layers)])

        self.decoder = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=32, nhead=nhead, dim_feedforward=input_dim, batch_first=True) for _ in
             range(num_decoder_layers)])

        self.downsample = nn.Conv1d(kernel_size=1, in_channels=7, out_channels=1)
        self.fc = nn.ModuleList([nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32)])
        self.output = nn.Linear(32, out_features=output_dim)

    def forward(self, src):

        # data(2,7,10)
        data = src[:, :, -4:].clone().detach().long()

        year_data = data[:, :, 0]
        month_data = data[:, :, 1]
        day_data = data[:, :, 2]
        hour_data = data[:, :, 3]

        # Pass each component to its corresponding embedding layer
        embedded_year = self.year_embedding(year_data)
        embedded_month = self.month_embedding(month_data)
        embedded_day = self.day_embedding(day_data)
        embedded_hour = self.hour_embedding(hour_data)
        encoded_features = torch.cat([embedded_year, embedded_month, embedded_day, embedded_hour], dim=2)

        encoded_features_np = torch.cat([src[:, :, 0:6], encoded_features], dim=2)

        src = encoded_features_np
        # tgt = src  # 因为输入是一样的
        src = self.position_encoder(src)
        tgt = src

        for layer in self.encoder:
            try:
                src = layer(src) + src
            except:
                print(src.shape)
                print(src)
        for layer in self.decoder:
            tgt = layer(tgt, src) + tgt

        output = self.downsample(tgt).squeeze(1)
        for layer in self.fc:
            output = layer(output)

        output = self.output(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        # Calculate the positional encodings for each position up to max_seq_len
        position_encoding = torch.zeros(max_seq_len, input_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * -(math.log(10000.0) / input_dim))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_encoding', position_encoding)

    def forward(self, input_tensor):
        # Get the positional encodings for the input sequence
        batch_size, seq_len, input_dim = input_tensor.size()
        position_encoding = self.position_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)

        # Add the positional encodings to the input tensor
        output = input_tensor + position_encoding.to(input_tensor.device)

        return output


class WeatherDateEmbedded(nn.Module):
    def __init__(self):
        super().__init__()
        # 日期特征编码
        self.year_embedding = nn.Embedding(3, 1)
        self.month_embedding = nn.Embedding(13, 5)
        self.day_embedding = nn.Embedding(31, 10)
        self.hour_embedding = nn.Embedding(24, 10)

    def forward(self, data):
        self.train()
        raw_data = data
        # data(2,7,10)
        data = data[:, :, -4:].clone().detach().long()

        year_data = data[:, :, 0]
        month_data = data[:, :, 1]
        day_data = data[:, :, 2]
        hour_data = data[:, :, 3]

        # Pass each component to its corresponding embedding layer
        embedded_year = self.year_embedding(year_data)
        embedded_month = self.month_embedding(month_data)
        embedded_day = self.day_embedding(day_data)
        embedded_hour = self.hour_embedding(hour_data)
        encoded_features = torch.cat([embedded_year, embedded_month, embedded_day, embedded_hour], dim=2)

        encoded_features_np = torch.cat([raw_data[:, :, 0:6], encoded_features], dim=2)
        return encoded_features_np
