import cv2
import os

import glob
import xml.etree.ElementTree as ET
import shutil
import numpy as np
from numpy import prod
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, concatenate, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "3"  # Set the GPU 2 to use


def save_fps(filepath):
    video = cv2.VideoCapture(filepath)

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    if not video.isOpened():
        print("Could not Open :", filepath)
        exit(0)
    try:
        if not os.path.exists(filepath[:-4]):
            print(f"make dir: {filepath[:-4]}")
            os.makedirs(filepath[:-4])
        else:
            print(f"exist dir: {filepath[:-4]}")
    except OSError:
        print('Error: Creating directory. ' + filepath[:-4])

    count = 0

    while (video.isOpened()):

        ret, image = video.read()
        if not ret:
            break
        if os.path.exists(filepath[:-4] + f"/{count}.jpg"):
            count += 1
            continue
        cv2.imwrite(filepath[:-4] + f"/{count}.jpg", image)
        count += 1

    video.release()


def mov_img(path):
    print(path)
    tree = ET.parse(path)
    root = tree.getroot()

    for object in root.findall("object"):
        for action in object.findall("action"):
            img = glob.glob(path[:-4] + '/*.jpg')
            for frame in action.findall("frame"):
                start = int(frame.find("start").text) - 1
                end = int(frame.find("end").text) - 1
                for i in img:
                    dir_path = i[:-4].split('/')[-2]
                    try:
                        if not os.path.exists('./fight/' + dir_path):
                            print(f"make dir: './fight/'{dir_path}")
                            os.makedirs('./fight/' + dir_path)
                    except OSError:
                        print('Error: Creating directory. ' + dir_path)

                    if start <= int(i.split('/')[-1].split('.')[0]) <= end:
                        if os.path.exists('./fight/' + dir_path + '/' + i.split('/')[-1]):
                            continue

                        shutil.move(i, './fight/' + dir_path)

    if os.path.exists(path[:-4]):
        shutil.move(path[:-4], './normal/')


def save_npy(source, destination, file_name, IMG_SIZE, INTERVAL):
    base = tf.keras.applications.MobileNetV3Small(input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet',
                                                  include_top=False)
    # features = np.empty((0, 28224))
    labels = []
    base_model = tf.keras.Sequential([
        base,
        Flatten()
    ])
    base_model.summary()
    base_model.trainable = False
    index = 0
    label = source[-1]
    for lb, i in enumerate(os.listdir(source)):  # source = new_datasets/train/1
        filedir = os.path.join(source, i)
        for j in os.listdir(filedir):
            split_video = os.path.join(filedir, j)
            frames = np.array(list(map(lambda x: os.path.join(split_video, x), os.listdir(split_video))))
            frames = sorted(frames, key=lambda x: int(x.split("/")[-1].strip(".png")))

            if len(frames) > 3000:
                print("frames over : ", len(frames))
                n = len(frames) - 3000
                n = n // 2
                frames = frames[n:-n]

            cutline = len(frames) % INTERVAL
            if cutline:
                frames = frames[:-cutline]

            if os.path.isfile(destination + '/' + file_name + '_' + str(index) + '.npy'):
                index += len(frames) // 45
                continue
            labels.append([label] * (len(frames)))
            frames = np.array(list(map(cv2.imread, frames)))
            frames = np.array(list(map(lambda x: cv2.resize(x, (IMG_SIZE, IMG_SIZE)), frames)))

            if len(frames) < 45:
                continue

            feature_list = base_model.predict(frames)
            for ind in range(0, len(feature_list), 45):
                np.save(destination + '/' + file_name + '_' + str(index) + '.npy', feature_list[ind:ind + 45])
                index += 1
        print("doing", index)
        if index > 6000:
            break
    print("done")