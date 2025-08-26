import json
import torch
import os
from torchvision import datasets, transforms
import numpy as np

# ===== Config =====
JSON_SPLIT_FILE = "dirichlet0.5_clients10.json"
SAVE_DIR = "./data"              # output directory
os.makedirs(SAVE_DIR, exist_ok=True)

# CIFAR-100 normalization values
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# CIFAR-100 fine label names
CIFAR100_FINE = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
    'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
    'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
    'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard',
    'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
    'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree',
    'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket','rose',
    'sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel',
    'streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train',
    'trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
]

def idx_to_name(idx):
    return CIFAR100_FINE[int(idx)]

def semantic_summary(y, top_k=5):
    classes, counts = np.unique(y, return_counts=True)
    order = np.argsort(-counts)
    total = counts.sum()

    tops = []
    for i in order[:top_k]:
        cls_idx = int(classes[i])
        name = idx_to_name(cls_idx)
        pct = 100.0 * counts[i] / total
        tops.append(f"{name} ({pct:.1f}%)")

    return total, tops

# ===== Load CIFAR-100 with normalization =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
])

trainset = datasets.CIFAR100(root=SAVE_DIR, train=True, download=True, transform=transform)
testset  = datasets.CIFAR100(root=SAVE_DIR, train=False, download=True, transform=transform)

# Convert to tensors
x_train = torch.stack([trainset[i][0] for i in range(len(trainset))])
y_train = torch.tensor([trainset[i][1] for i in range(len(trainset))])

x_test  = torch.stack([testset[i][0] for i in range(len(testset))])
y_test  = torch.tensor([testset[i][1] for i in range(len(testset))])

print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# ===== Load JSON split =====
with open(JSON_SPLIT_FILE, "r") as f:
    split = json.load(f)

metadata = {}

# ===== Save each client =====
for client_id, indices in split.items():
    indices = list(map(int, indices))
    x_client = x_train[indices]
    y_client = y_train[indices]

    save_path = os.path.join(SAVE_DIR, f"client_{client_id}.pt")
    torch.save({"x": x_client, "y": y_client}, save_path)

    # Stats
    unique, counts = torch.unique(y_client, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    total, tops = semantic_summary(y_client.numpy(), top_k=5)

    print(f"\n--- Client {client_id} ---")
    print(f"Samples: {len(y_client)}")
    print(f"Unique classes: {len(unique)}")
    print(f"Class distribution: {dist}")
    print(f"Top classes: {', '.join(tops)}")

    metadata[client_id] = {
        "total_samples": len(y_client),
        "unique_classes": len(unique),
        "class_distribution": dist,
        "top_classes": tops
    }

# ===== Save global test dataset =====
torch.save({"x": x_test, "y": y_test}, os.path.join(SAVE_DIR, "global_test_dataset.pt"))
print(f"\n Saved global test dataset with {len(y_test)} samples")

# ===== Save metadata =====
with open(os.path.join(SAVE_DIR, "client_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f" Metadata saved to {os.path.join(SAVE_DIR, 'client_metadata.json')}")
