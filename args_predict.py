import argparse

parser = argparse.ArgumentParser(description = 'Pass in the Train and Predict Parameters')
parser.add_argument('--image_path', type = str, default = './flowers/test/3/image_06634.jpg')
parser.add_argument('--checkpoint_path', type = str, default = './checkpoint.pth')
parser.add_argument('--cat_path', type = str, default = './cat_to_name.json')
parser.add_argument('--topk', type = int, default = 3)
parser.add_argument('--gpu', type = str,  default = 'GPU', choices=['GPU','CPU'])
args = parser.parse_args()