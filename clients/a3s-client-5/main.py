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
mcp = FastMCP("a3s-client-5")

load_dotenv()
private_key = os.getenv("PRIVATE_KEY")
base_url = os.getenv("RESOURCE_SERVER_URL")
endpoint_path = os.getenv("ENDPOINT_PATH")

def log(*a, **kw):
    print(*a, file=sys.stderr, flush=True, **kw)

# --- Internal Data and Training Methods ---

# LOCAL_DATA_PATH = os.path.join(ROOT, "data", "client_5.npz")
LOCAL_DATA_PATH = os.path.join(ROOT, "data", "client_5.pt")

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]
    
    # def __len__(self) -> int:
    #     return len(self.images)
    
    # def __getitem__(self, idx: int) -> Any:
    #     return self.images[idx], self.labels[idx]

def _load_local_dataloader(batch_size=32):
    data = torch.load(LOCAL_DATA_PATH)  # loads preprocessed .pt file
    dataset = MyDataset(data["x_train"], data["y_train"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def _state_dict_to_b64(sd: Dict[str, torch.Tensor]) -> str:
    buf = io.BytesIO()
    torch.save(
        {k: (v.cpu().half() if v.dtype.is_floating_point else v.cpu()) for k, v in sd.items()},
        buf,
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _b64_to_state_dict(b64: str) -> Dict[str, torch.Tensor]:
    raw = base64.b64decode(b64.encode("ascii"))
    buf = io.BytesIO(raw)
    sd = torch.load(buf, map_location="cpu")
    # cast back to float32 so training is unaffected
    for k, v in sd.items():
        if v.dtype == torch.float16:
            sd[k] = v.float()
    return sd

def _train_model_locally(model_params: Dict[str, torch.Tensor], dataloader: DataLoader, epochs: int = 10) -> Dict[str, torch.Tensor]:
    """
    Internal training logic for a single client.
    """
    model = CNN()
    model.load_state_dict(model_params)
    epochs = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
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
    Total Samples: 4865
    Top classes: chimpanzee (5.7%), tractor (4.5%), sea (4.2%), kangaroo (4.2%), lizard (3.7%)
    Number of Unique Classes: 85
    Class Skewness (Std Dev): 61.37
    Class Distribution: {0: 135, 1: 81, 2: 2, 3: 19, 4: 45, 5: 8, 6: 3, 7: 17, 8: 23, 9: 162,
    10: 48, 11: 31, 12: 14, 13: 61, 14: 5, 15: 77, 16: 176, 17: 3, 18: 12, 19: 83, 20: 8,
    21: 275, 22: 142, 24: 114, 25: 22, 26: 119, 28: 10, 29: 17, 30: 1, 31: 107, 32: 39,
    34: 12, 35: 87, 36: 19, 37: 49, 38: 204, 39: 100, 41: 7, 42: 1, 44: 179, 45: 4, 46: 16,
    47: 17, 48: 24, 49: 34, 50: 119, 52: 143, 53: 47, 55: 2, 56: 19, 57: 129, 58: 57, 60: 9,
    61: 68, 62: 53, 63: 1, 65: 40, 66: 1, 67: 5, 68: 1, 69: 92, 70: 6, 71: 204, 72: 14, 73: 4,
    74: 76, 76: 175, 77: 21, 79: 81, 80: 15, 82: 50, 83: 14, 85: 89, 87: 32, 88: 52, 89: 220,
    90: 101, 91: 1, 93: 49, 94: 124, 95: 16, 96: 90, 97: 5, 98: 21, 99: 7}
    
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