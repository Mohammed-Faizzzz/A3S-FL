import tensorflow as tf
import torch
from torch.utils.data import TensorDataset

# Load CIFAR-100
(_, _), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# Flatten labels
y_test = y_test.flatten()

# Convert to PyTorch tensors
mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3,1,1)
std  = torch.tensor([0.2675, 0.2565, 0.2761]).view(3,1,1)

x_test_t = torch.tensor(x_test).permute(0, 3, 1, 2).float() / 255.0
x_test_t = (x_test_t - mean) / std
y_test_t = torch.tensor(y_test).long()

# Wrap in TensorDataset
test_dataset = TensorDataset(x_test_t, y_test_t)

# Save to disk
torch.save(
    {"x": x_test_t, "y": y_test_t},
    "global_test_dataset.pt"
)
print(f"Saved global test dataset: {len(test_dataset)} samples")
