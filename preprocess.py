import argparse
import glob

from utils.convert import save_npy, save_fps, mov_img


parser = argparse.ArgumentParser(description="Converting image to features and features to npy")

# dataset = './원본/싸움/*/*/*.mp4'
# label = './싸움/*/*/*.xml'

parser.add_argument("-v", "--dataset", type=str, required=True,help = "folder location for video set")
parser.add_argument("-l", "--label", type=str, required=True, help = "folder location for annotation xml files")
parser.add_argument("-s", "--src", type=str, required=True)
parser.add_argument("-d", "--dst", type=str, required=True)
parser.add_argument("-f", "--filename", type=str, required=True)
parser.add_argument("--img_size", type=int, default = 224, required=True)
parser.add_argument("--interval", type=int, deafult = 45, required=True)

args = parser.parse_args()

dataset = args.dataset
label = args.label
source = args.src
destination = args.dst
filename = args.filename
IMG_SIZE = args.img_size
INTERVAL = args.interval

# 비디오를 프레임별 이미지로 분할
mov = glob.glob(dataset)
for m in mov:
    save_fps(m)

# xml파일에 따른 label 정리
mov = glob.glob(label)
for m in mov:
    mov_img(m)

# 특징 추출 후 npy 파일로 변환
save_npy(source, destination, filename, IMG_SIZE, INTERVAL)