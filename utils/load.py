import os
import pandas as pd
import numpy as np
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [{"path": self.list_IDs["path"][k],"label": self.list_IDs["label"][k]}for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(ID["path"])
            # Store class
            y[i] = ID["label"]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def npy_to_csv(src, save):
    path_list = []
    label_list = []
    num_list = []
    g_list = []
    for (path, dir, files) in os.walk(src):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.npy':
                label = path.split('/')[-1]
                if label == '1':
                    continue
                if label == '2':
                    label = 1
                num = os.path.splitext(filename)[0].split('_')[-1]
                g = os.path.splitext(filename)[0].split('_')[-2][-1]
                if g == '2':
                    continue
                path_list.append(os.path.join(path, filename))
                label_list.append(label)
                num_list.append(int(num))

    df = pd.DataFrame(columns=['path', 'label', 'num'])
    df['path'] = path_list
    df['label'] = label_list
    df['num'] = num_list
    df = df.sort_values(by=['label', 'num'])

    df.to_csv(save)

