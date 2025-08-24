import tensorflow as tf
import numpy as np
import random
import os
import warnings

warnings.filterwarnings('ignore')

# =========================
# Load CIFAR-100
# =========================
cifar100 = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

# =========================
# Config
# =========================
num_clients = 10
mode = "fixed"   # choose: "dirichlet" or "fixed"
delta = 0.01     # used for dirichlet
classes_per_client = 10  # used for fixed

save_dir = "./splits_cifar100_noniid"
os.makedirs(save_dir, exist_ok=True)

# =========================
# Split Function
# =========================
def create_split(x, y, num_clients, mode="dirichlet", delta=0.01, classes_per_client=10):
    client_indices = [[] for _ in range(num_clients)]

    if mode == "dirichlet":
        for c in np.unique(y):
            idx_c = np.where(y == c)[0]
            np.random.shuffle(idx_c)
            proportions = np.random.dirichlet(alpha=[delta] * num_clients)
            counts = (proportions * len(idx_c)).astype(int)
            start = 0
            for client_id, count in enumerate(counts):
                client_indices[client_id].extend(idx_c[start:start+count])
                start += count

    elif mode == "fixed":
        all_classes = list(np.unique(y))
        random.shuffle(all_classes)
        class_splits = np.array_split(all_classes, num_clients)
        for client_id, client_classes in enumerate(class_splits):
            for c in client_classes:
                idx_c = np.where(y == c)[0]
                np.random.shuffle(idx_c)
                client_indices[client_id].extend(idx_c)

    client_data = [(x[idx], y[idx]) for idx in client_indices]
    return client_data

# =========================
# Description Function
# =========================
def create_data_description(client_data):
    _, y_data = client_data
    total_samples = len(y_data)
    unique_classes, counts = np.unique(y_data, return_counts=True)
    class_distribution = dict(zip(unique_classes.tolist(), counts.tolist()))
    class_skew = np.std(counts) if len(counts) > 0 else 0.0
    return {
        'total_samples': total_samples,
        'unique_classes_count': len(unique_classes),
        'class_distribution': class_distribution,
        'class_skew_metric': class_skew,
    }

# =========================
# Do the split
# =========================
train_splits = create_split(x_train, y_train, num_clients, mode, delta, classes_per_client)
test_splits  = create_split(x_test, y_test, num_clients, mode, delta, classes_per_client)

# =========================
# Save + print description
# =========================
for i, (train_data, test_data) in enumerate(zip(train_splits, test_splits)):
    x_tr, y_tr = train_data
    x_te, y_te = test_data

    # Save
    np.savez_compressed(
        os.path.join(save_dir, f"client_{i}_data.npz"),
        x_train=x_tr, y_train=y_tr,
        x_test=x_te, y_test=y_te
    )

    # Print description
    desc_tr = create_data_description(train_data)
    desc_te = create_data_description(test_data)

    print(f"\n--- Client {i} ---")
    print(f"Train Samples: {desc_tr['total_samples']} | "
          f"Unique Classes: {desc_tr['unique_classes_count']} | "
          f"Class Skew (std): {desc_tr['class_skew_metric']:.2f}")
    print(f"Train Class Dist: {desc_tr['class_distribution']}")
    print(f"Test Samples: {desc_te['total_samples']} | "
          f"Unique Classes: {desc_te['unique_classes_count']} | "
          f"Class Skew (std): {desc_te['class_skew_metric']:.2f}")
    print(f"Test Class Dist: {desc_te['class_distribution']}")

print(f"\n✅ Saved client splits in {save_dir}")
