import tensorflow as tf
from tensorflow import keras
from keras import layers
import os
import random

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

# Save Client_Data
client_data_dir = "client_data"
os.makedirs(client_data_dir, exist_ok=True)

for i, (x, y) in enumerate(client_data):
    np.savez_compressed(os.path.join(client_data_dir, f"client_{i}.npz"), x=x, y=y)

def create_data_description(client_data):
    """
    Generates a detailed description of a client's data split.

    Args:
        client_data (tuple): A tuple containing the (x, y) data for one client.

    Returns:
        dict: A dictionary with detailed data characteristics.
    """
    _, y_data = client_data

    # Find the total number of samples
    total_samples = len(y_data)

    # Get unique classes and their counts
    unique_classes, counts = np.unique(y_data, return_counts=True)
    class_distribution = dict(zip(unique_classes.tolist(), counts.tolist()))

    # Calculate class skewness
    class_skew = np.std(counts)

    return {
        'total_samples': total_samples,
        'unique_classes_count': len(unique_classes),
        'class_distribution': class_distribution,
        'class_skew_metric': class_skew,
    }
    
all_client_descriptions = []

for i, client_split in enumerate(client_data):
    description = create_data_description(client_split)
    description['client_id'] = i
    all_client_descriptions.append(description)

for desc in all_client_descriptions:
    print(f"\n--- Client {desc['client_id']} Description ---")
    print(f"Total Samples: {desc['total_samples']}")
    print(f"Number of Unique Classes: {desc['unique_classes_count']}")
    print(f"Class Skewness (Std Dev): {desc['class_skew_metric']:.2f}")
    print("Class Distribution:", desc['class_distribution'])
