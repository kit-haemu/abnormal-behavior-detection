import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, concatenate, Dropout, MaxPooling2D, Flatten, GRU, TimeDistributed, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import argparse


from utils.grapher import draw
from utils.load import npy_to_csv, DataGenerator
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"]= "2"  # Set the GPU 2 to use

parser = argparse.ArgumentParser()

# dataset = ../dataset
parser.add_argument("-d", "--dataset", type=str, required=True)
parser.add_argument("--epochs", default=50, type=int)

parser.add_argument("--dim", default = 28224, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--n_classes", default = 2, type=int)
parser.add_argument("--n_channels", default = 30, type=int)
parser.add_argument("--shuffle", default=False, type=bool)

args = parser.parse_args()
datasets = args.dataset
epochs = args.epochs

params = {'dim': [args.dim],
          'batch_size': args.batch_size,
          'n_classes': args.n_classes,
          'n_channels': args.n_channels,
          'shuffle': args.shuffle}

strategy = tf.distribute.MirroredStrategy()

npy_to_csv(f"{datasets}/train", "train.csv")
npy_to_csv(f"{datasets}/val", "val.csv")

train = pd.read_csv('./train.csv')
val =  pd.read_csv('./val.csv')

train_ds = DataGenerator(train, **params)
val_ds = DataGenerator(val, **params)

with strategy.scope():
    input = keras.Input(shape=(30, 28224))  # 이미지 레이어

    x = GRU(512, activation='tanh')(input)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(2, activation='sigmoid')(x)
    model = keras.Model(input, output)

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

model.summary()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[
        ModelCheckpoint(filepath='./model.h5',monitor='val_accuracy',verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,patience=3,verbose=1,mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.00001, patience=50, verbose=1, mode='auto')
    ]
)

draw(history)
plt.show()