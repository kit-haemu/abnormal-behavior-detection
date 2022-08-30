import argparse
import glob

from utils.convert import save_npy, save_fps


parser = argparse.ArgumentParser(description="Converting image to features and features to npy")

# dataset = './원본/싸움/*/*/*.mp4'
# label = './싸움/*/*/*.xml'

parser.add_argument("-v", "--dataset", type=str, required=True,help = "location of video set")
parser.add_argument("-l", "--label", type=str, required=True,help = "location of video set")
parser.add_argument("-s", "--src", type=str, required=True)
parser.add_argument("-d", "--dst", type=str, required=True)
parser.add_argument("-f", "--filename", type=str, required=True)
parser.add_argument("--img_size", type=int, default = 224, required=True)
parser.add_argument("--interval", type=int, deafult = 45, required=True)

args = parser.parse_args()

dataset = args.dataset
source = args.src
destination = args.dst
filename = args.filename
IMG_SIZE = args.img_size
INTERVAL = args.interval


mov = glob.glob(dataset)
for m in mov:
    save_fps(m)

save_npy(source, destination, filename, IMG_SIZE, INTERVAL)