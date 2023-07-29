import torch
import torch.nn as nn

from model import WeatherPredict, ARModel


class BNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleDict({
            'TransformerLayer': WeatherPredict(input_dim=128, output_dim=5, max_seq_len=4 * 24, nhead=2,
                                               num_encoder_layers=2, num_decoder_layers=2),
            'ARLayer': ARModel(5, 4 * 24),
            'OutLayer': nn.Linear(10, 5)
        })

    def forward(self, x):
        transform_out = self.net['TransformerLayer'](x)

        ar_out = self.net['ARLayer'](x)
        out = torch.cat([transform_out, ar_out], dim=1)
        out = self.net['OutLayer'](out)
        return out
