import math

import torch
import torch.nn as nn


class WeatherPredict(nn.Module):
    def __init__(self, input_dim, output_dim=5, max_seq_len=4 * 24, nhead=2, num_encoder_layers=4,
                 num_decoder_layers=2):
        super(WeatherPredict, self).__init__()

        self.year_embedding = nn.Embedding(3, 1)
        self.month_embedding = nn.Embedding(13, 5)
        self.day_embedding = nn.Embedding(32, 11)
        self.hour_embedding = nn.Embedding(25, 10)
        self.min_embedding = nn.Embedding(61, 18)

        self.position_encoder = PositionalEncoding(input_dim, max_seq_len)
        self.embedded = nn.Linear(in_features=50, out_features=input_dim)
        self.encoder = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=128, batch_first=True) for _ in
             range(num_encoder_layers)])

        self.decoder = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=input_dim, batch_first=True) for
             _ in
             range(num_decoder_layers)])

        self.downsample = nn.Conv1d(kernel_size=1, in_channels=4 * 24, out_channels=1)
        self.fc = nn.ModuleList([nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, input_dim)])
        self.output = nn.Linear(input_dim, out_features=output_dim)

    def forward(self, src):
        # 日期特征嵌入层
        # data(2,7,10)
        data = src[:, :, -5:].clone().detach().long()
        year_data = data[:, :, 0]
        month_data = data[:, :, 1]
        day_data = data[:, :, 2]
        hour_data = data[:, :, 3]
        minute_data = data[:, :, 4]
        # Pass each component to its corresponding embedding layer
        embedded_year = self.year_embedding(year_data)
        embedded_month = self.month_embedding(month_data)
        embedded_day = self.day_embedding(day_data)
        embedded_hour = self.min_embedding(hour_data)
        embedded_minute = self.hour_embedding(minute_data)
        encoded_features = torch.cat([embedded_year, embedded_month, embedded_day, embedded_hour, embedded_minute],
                                     dim=2)
        encoded_features_np = torch.cat([src[:, :, 0:5], encoded_features], dim=2)
        src = self.embedded(encoded_features_np.to(torch.float32))
        # tgt = src  # 因为输入是一样的
        src = self.position_encoder(src)
        tgt = src

        # Transformer层
        for layer in self.encoder:
            src = layer(src) + src
        for layer in self.decoder:
            tgt = layer(tgt, src) + tgt

        output = self.downsample(tgt).squeeze(1)
        for layer in self.fc:
            output = layer(output)

        output = self.output(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_seq_len=4 * 24):
        super(PositionalEncoding, self).__init__()
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




class ARModel(nn.Module):
    def __init__(self, input_dim, lag):
        super(ARModel, self).__init__()
        self.input_dim = input_dim
        self.lag = lag
        self.layers = nn.ModuleList([nn.Linear(lag, 1) for _ in range(input_dim)])

    def forward(self, x):
        x=x.to(torch.float32)
        x = x[:, :, 0:5]
        batch_size, time_steps, _ = x.size()
        ar_out = []
        for i in range(self.input_dim):
            ar_i = self.layers[i](x[:, -self.lag:, i])
            ar_out.append(ar_i)
        output = torch.cat(ar_out, dim=1)

        return output
