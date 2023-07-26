import argparse

from dataset import WeatherDataPreprocessor

parser = argparse.ArgumentParser(description=' All params')
# ----dataset
parser.add_argument('--file_path', type=str,
                    default=r'E:\lin\DeepLearning\02\附件2.csv',
                    help='raw data file')  # required=True,

parser.add_argument('--horizon', type=int, default=1, metavar='N', help='horizon')
parser.add_argument('--batch_size', type=int, default=8, metavar='N', help='batch size (default: 32)')
parser.add_argument('--window', type=int, default=7 * 2 * 24, metavar='N', help='window_step')
parser.add_argument('--window_step', type=int, default=1, metavar='N', help='window_step')

para = parser.parse_known_args()[0]

data = WeatherDataPreprocessor(para)
