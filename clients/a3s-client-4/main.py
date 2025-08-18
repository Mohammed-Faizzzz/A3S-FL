from typing import Any, Dict
import httpx
from mcp.server.fastmcp import FastMCP
import os
import sys
import torch
import numpy as np
import torch.nn as nn
import io, base64
from torch.utils.data import DataLoader, Dataset
# /Users/mohammedfaiz/Desktop/FLock/MCP-Attack/clients/a3s-client-0/main.py
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

# Import the base model architecture
from models.cnn_model import CNN
mcp = FastMCP("a3s-client-4")

load_dotenv()
private_key = os.getenv("PRIVATE_KEY")
base_url = os.getenv("RESOURCE_SERVER_URL")
endpoint_path = os.getenv("ENDPOINT_PATH")

def log(*a, **kw):
    print(*a, file=sys.stderr, flush=True, **kw)

# --- Internal Data and Training Methods ---

LOCAL_DATA_PATH = os.path.join(ROOT, "data", "client_0.npz")

class MyDataset(Dataset):
    """
    A simple Dataset class to load pre-split data from a .npz file.
    """
    def __init__(self, data_path: str):
        data = np.load(data_path, allow_pickle=True)
        images_key = 'images' if 'images' in data else 'x'
        labels_key = 'labels' if 'labels' in data else 'y'

        self.images = torch.from_numpy(data[images_key]).float()
        if self.images.ndim == 4 and self.images.shape[-1] in (1, 3):
            self.images = self.images.permute(0, 3, 1, 2) / 255.0
        self.labels = torch.from_numpy(data[labels_key]).long()
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Any:
        return self.images[idx], self.labels[idx]

def _load_local_dataloader() -> DataLoader:
    """
    Internal method to load and return the client's local data loader.
    """
    try:
        local_dataset = MyDataset(LOCAL_DATA_PATH)
        return DataLoader(local_dataset, batch_size=32, shuffle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Local data not found at {LOCAL_DATA_PATH}")

def _state_dict_to_b64(sd):
    buf = io.BytesIO(); import torch as _t
    _t.save({k: v.cpu() for k, v in sd.items()}, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _b64_to_state_dict(b64):
    raw = base64.b64decode(b64.encode("ascii")); buf = io.BytesIO()
    import torch as _t
    return _t.load(buf.__class__(raw), map_location="cpu")

def _train_model_locally(model_params: Dict[str, torch.Tensor], dataloader: DataLoader, epochs: int = 5) -> Dict[str, torch.Tensor]:
    """
    Internal training logic for a single client.
    """
    model = CNN()
    model.load_state_dict(model_params)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    model.train()
    for epoch in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    
    log(f"Client {mcp.name} finished local training for {epochs} epochs.")
    return model.state_dict()

@mcp.tool()
async def train_model_with_local_data(global_model_params: str, epochs: int = 1) -> Dict[str, Any]:
    """
    Performs federated learning on a client with a non-IID subset of the
    CIFAR-100 dataset.
    total_samples: 4245,
    num_unique_classes: 14,
    class_skewness_std_dev: 221.57,
    class_distribution:
    {0: 2, 13: 478, 15: 44, 19: 341, 21: 384, 25: 488,
    28: 496, 40: 499, 54: 11, 71: 499, 76: 2, 78: 499, 82: 3, 99: 499}
    
    Args:
        global_model_params: A dictionary of the global model's parameters.
        epochs: The number of local training epochs.
    
    Returns:
        A dictionary of the locally trained model's updated parameters.
    """
    try:
        local_dataloader = _load_local_dataloader()
        # decode incoming params (base64 -> state_dict)
        global_sd = _b64_to_state_dict(global_model_params)
        updated_sd = _train_model_locally(global_sd, local_dataloader, epochs)
        return {
            "params_b64": _state_dict_to_b64(updated_sd),
            "num_samples": len(local_dataloader.dataset)
        }
    except FileNotFoundError as e:
        log(f"Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    log(f"Starting MCP server for {mcp.name}...")
    mcp.run(transport='stdio')