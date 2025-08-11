import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Load CIFAR-100
cifar100 = tf.keras.datasets.cifar100
(x_train, y_train), _ = cifar100.load_data()
y_train = y_train.flatten()

# Specify num of clients and skew
num_clients = 10
delta = 0.01

# Holds data for each client
client_indices = [[] for _ in range(num_clients)]

# Go class by class
for c in np.unique(y_train):
    idx_c = np.where(y_train == c)[0]  # all indices for class c
    np.random.shuffle(idx_c)

    # Dirichlet split
    proportions = np.random.dirichlet(alpha=[delta] * num_clients)
    # Convert proportions to counts
    counts = (proportions * len(idx_c)).astype(int)

    # Split and assign
    start = 0
    for client_id, count in enumerate(counts):
        client_indices[client_id].extend(idx_c[start:start+count])
        start += count

# Convert to NumPy arrays
client_data = [(x_train[idx], y_train[idx]) for idx in client_indices]

# Example: size of each client's dataset
print([len(c[0]) for c in client_data])

