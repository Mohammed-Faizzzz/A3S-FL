import numpy as np
import os

# CIFAR-100 fine label names (0–99)
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

# Path to the .npz files
data_dir = "./"   # adjust if needed

for fname in sorted(os.listdir(data_dir)):
    if fname.startswith("client_") and fname.endswith(".npz"):
        path = os.path.join(data_dir, fname)
        npz = np.load(path, allow_pickle=True)

        # Usually your npz contains 'x' and 'y' arrays
        x = npz["x"]
        y = npz["y"]

        total, tops = semantic_summary(y, top_k=5)

        print(f"\n>>> {fname}")
        print(f"Samples: {total}")
        print("Top classes:", ", ".join(tops))
