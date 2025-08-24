import os
import numpy as np
import torch

# CIFAR-100 normalization constants
MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
STD  = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)

input_dir = "./splits_cifar100_noniid"   # where your .npz files are
output_dir = "./splits_cifar100_torch"   # where we'll save .pt files
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith(".npz"):
        continue

    path = os.path.join(input_dir, fname)
    data = np.load(path)
    print(data)

    # training data
    x_tr = torch.tensor(data["x"]).permute(0, 3, 1, 2).float() / 255.0
    x_tr = (x_tr - MEAN) / STD
    y_tr = torch.tensor(data["y"]).long()

    # test data
    # x_te = torch.tensor(data["x_test"]).permute(0, 3, 1, 2).float() / 255.0
    # x_te = (x_te - MEAN) / STD
    # y_te = torch.tensor(data["y_test"]).long()

    out = {
        "x_train": x_tr,
        "y_train": y_tr,
        # "x_test": x_te,
        # "y_test": y_te,
    }

    out_path = os.path.join(output_dir, fname.replace(".npz", ".pt"))
    torch.save(out, out_path)
    print(f"✅ Saved {out_path}")
