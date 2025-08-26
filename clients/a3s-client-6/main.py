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
mcp = FastMCP("a3s-client-6")

load_dotenv()
private_key = os.getenv("PRIVATE_KEY")
base_url = os.getenv("RESOURCE_SERVER_URL")
endpoint_path = os.getenv("ENDPOINT_PATH")

def log(*a, **kw):
    print(*a, file=sys.stderr, flush=True, **kw)

# --- Internal Data and Training Methods ---

# LOCAL_DATA_PATH = os.path.join(ROOT, "data", "client_6.npz")
LOCAL_DATA_PATH = os.path.join(ROOT, "data", "client_6.pt")

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
    epochs = 10

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
    Total Samples: 4939
    Top classes: elephant (4.6%), motorcycle (3.7%), table (3.7%), castle (3.5%), telephone (3.3%)
    Number of Unique Classes: 89
    Class Skewness (Std Dev): 52.88
    Class Distribution: {0: 11, 1: 23, 2: 6, 3: 45, 4: 4, 5: 27, 6: 78, 7: 50, 10: 70,
    11: 28, 12: 1, 13: 85, 14: 47, 17: 172, 18: 92, 19: 15, 20: 28, 21: 4, 22: 11,
    23: 164, 24: 17, 25: 35, 26: 3, 27: 18, 28: 132, 29: 34, 30: 12, 31: 226, 32: 7,
    33: 7, 34: 47, 35: 103, 36: 41, 37: 65, 38: 34, 39: 11, 40: 13, 41: 32, 42: 9, 43: 34,
    44: 122, 46: 126, 47: 160, 48: 182, 49: 78, 51: 16, 52: 10, 53: 104, 54: 8, 55: 93,
    57: 34, 58: 6, 59: 2, 60: 143, 61: 13, 62: 57, 63: 42, 64: 78, 65: 82, 66: 12, 67: 137,
    68: 26, 69: 5, 70: 122, 71: 16, 72: 18, 73: 6, 74: 110, 76: 6, 77: 58, 78: 52, 79: 7,
    80: 4, 81: 136, 82: 10, 83: 10, 84: 182, 86: 165, 87: 66, 89: 11, 90: 75, 91: 92, 92: 65,
    93: 75, 95: 90, 96: 11, 97: 48, 98: 86, 99: 41}
    
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