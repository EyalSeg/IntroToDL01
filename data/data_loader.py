import os
import numpy as np
from scipy.io import loadmat

data_dir = os.path.dirname(os.path.realpath(__file__))


class DataLoader:

    @staticmethod
    def load_dataset(dataset_name):
        path = os.path.join(data_dir, dataset_name)
        data = loadmat(path)
        #
        # if add_bias:
        #     data['Yt'] = DataLoader.__add_bias_to_input(data['Yt'])
        #     data['Yv'] = DataLoader.__add_bias_to_input(data['Yv'])

        return data

    @staticmethod
    def __add_bias_to_input(X):
        return np.pad(X, pad_width=([0, 1], [0, 0]), mode='constant', constant_values=(0, 1))

