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