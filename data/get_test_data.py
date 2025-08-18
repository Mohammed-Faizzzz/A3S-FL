import tensorflow as tf
import torch
from torch.utils.data import TensorDataset

# Load CIFAR-100
(_, _), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# Flatten labels
y_test = y_test.flatten()

# Convert to PyTorch tensors
x_test_t = torch.tensor(x_test).permute(0, 3, 1, 2).float() / 255.0  # NHWC -> NCHW, normalize
y_test_t = torch.tensor(y_test).long()

# Wrap in TensorDataset
test_dataset = TensorDataset(x_test_t, y_test_t)

# Save to disk
torch.save(
    {"x": x_test_t, "y": y_test_t},
    "global_test_dataset.pt"
)
print(f"Saved global test dataset: {len(test_dataset)} samples")
