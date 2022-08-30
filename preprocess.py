import argparse
from utils.FeatureConverter import convert


parser = argparse.ArgumentParser(description="Converting image to features and features to npy")

parser.add_argument("-s", "--src", type=str, required=True)
parser.add_argument("-d", "--dst", type=str, required=True)
parser.add_argument("-f", "--filename", type=str, required=True)
parser.add_argument("--img_size", type=int, default = 224, required=True)
parser.add_argument("--interval", type=int, deafult = 45, required=True)

args = parser.parse_args()

source = args.src
destination = args.dst
filename = args.filename
IMG_SIZE = args.img_size
INTERVAL = args.interval

convert(source, destination, filename, IMG_SIZE, INTERVAL)