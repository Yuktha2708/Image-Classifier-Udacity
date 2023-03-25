import argparse

parser = argparse.ArgumentParser(description = 'Pass in the Train and Predict Parameters')
parser.add_argument('--data_dir', type = str, default = 'flowers')
parser.add_argument('--save_dir', type = str, default = './')
parser.add_argument('--arch', type = str, default = 'vgg16', choices=['vgg16', 'densenet121', 'alexnet'])
parser.add_argument('--epochs', type = int, default = 3)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--gpu', type = str,  default = 'GPU', choices=['GPU','CPU'])
parser.add_argument('--in_layers', type = int, default = 25088)
parser.add_argument('--hidden_layers', type = int, default = 4096)
parser.add_argument('--out_layers', type = int, default = 102)
parser.add_argument('--drop_rate', type = float, default = 0.5)
parser.add_argument('--topk', type = int, default = 3)


args = parser.parse_args()