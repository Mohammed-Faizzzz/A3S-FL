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
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

# Import the base model architecture
from models.cnn_model import CNN
mcp = FastMCP("a3s-client-0")

load_dotenv()
private_key = os.getenv("PRIVATE_KEY")
base_url = os.getenv("RESOURCE_SERVER_URL")
endpoint_path = os.getenv("ENDPOINT_PATH")

def log(*a, **kw):
    print(*a, file=sys.stderr, flush=True, **kw)

# --- Internal Data and Training Methods ---

# LOCAL_DATA_PATH = os.path.join(ROOT, "data", "client_0.npz")
LOCAL_DATA_PATH = os.path.join(ROOT, "data", "client_0.pt")

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
    Total Samples: 5697
    Top classes: pine_tree (5.8%), forest (4.8%), leopard (4.2%), seal (3.8%), bee (3.6%)
    Number of Unique Classes: 90
    Class Skewness (Std Dev): 70.70
    Class Distribution: {0: 1, 1: 103, 2: 55, 3: 14, 4: 4, 5: 34, 6: 203, 7: 134, 8: 193,
    9: 159, 10: 8, 11: 110, 12: 23, 13: 13, 14: 19, 17: 26, 18: 2, 19: 18, 21: 39, 22: 22,
    23: 93, 25: 146, 26: 94, 27: 63, 28: 2, 29: 8, 30: 180, 31: 26, 32: 5, 33: 272, 34: 7,
    35: 38, 36: 29, 37: 1, 38: 6, 39: 202, 40: 47, 41: 64, 42: 239, 43: 160, 44: 74, 45: 41,
    46: 10, 47: 16, 49: 153, 50: 45, 51: 79, 52: 36, 54: 62, 55: 43, 56: 141, 57: 10, 58: 125,
    59: 329, 60: 13, 61: 21, 62: 5, 63: 65, 65: 1, 67: 73, 68: 9, 69: 7, 70: 72, 71: 5, 72: 218,
    73: 18, 75: 4, 76: 6, 77: 4, 78: 98, 79: 15, 80: 27, 81: 81, 82: 5, 83: 27, 84: 15, 85: 25,
    86: 17, 87: 89, 88: 39, 89: 88, 90: 200, 91: 12, 93: 2, 94: 76, 95: 15, 96: 166, 97: 92, 98: 60, 99: 1}    
    
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
        log(f"Returning {len(updated_sd)} params and {len(local_dataloader.dataset)} samples")
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