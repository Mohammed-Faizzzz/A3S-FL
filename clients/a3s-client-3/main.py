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
import logging

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

# Import the base model architecture
from models.cnn_model import CNN
mcp = FastMCP("a3s-client-3")

load_dotenv()
private_key = os.getenv("PRIVATE_KEY")
base_url = os.getenv("RESOURCE_SERVER_URL")
endpoint_path = os.getenv("ENDPOINT_PATH")

# --- Internal Data and Training Methods ---

LOCAL_DATA_PATH = os.path.join(ROOT, "data", "client_3.pt")

# Logging: file + stderr (not MCP’s stdio)
logfile = os.path.join(ROOT, f"{mcp.name}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(logfile, mode="a"),
        logging.StreamHandler(sys.__stderr__)  # safe to real stderr
    ]
)
logger = logging.getLogger(mcp.name)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]

def _load_local_dataloader(batch_size=32):
    data = torch.load(LOCAL_DATA_PATH)  # loads preprocessed .pt file
    dataset = MyDataset(data["x"], data["y"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f"Epoch {epoch+1}/{epochs}, loss={running_loss/len(dataloader):.4f}")

    logger.info(f"Client {mcp.name} finished local training for {epochs} epochs.")
    return model.state_dict()

@mcp.tool()
async def train_model_with_local_data(global_model_params: str, epochs: int = 10) -> Dict[str, Any]:
    """
    Performs federated learning on a client with a non-IID subset of the
    CIFAR-100 dataset.
    Samples: 3209
    Unique classes: 90
    Class distribution: {0: 8, 1: 29, 2: 50, 3: 36, 4: 4, 5: 34, 7: 56, 10: 83, 11: 8, 12: 4, 13: 124, 14: 34, 15: 25, 16: 7, 17: 9, 18: 68, 19: 3, 20: 2, 22: 19, 23: 2, 24: 83, 25: 1, 26: 80, 27: 34, 28: 3, 29: 105, 30: 1, 31: 34, 32: 14, 33: 100, 34: 2, 35: 10, 36: 69, 39: 2, 40: 21, 41: 68, 42: 29, 43: 4, 44: 3, 45: 29, 46: 110, 47: 117, 48: 26, 49: 1, 50: 23, 51: 20, 52: 7, 55: 14, 56: 2, 57: 10, 58: 32, 59: 7, 60: 3, 61: 1, 62: 82, 63: 3, 64: 1, 65: 47, 66: 202, 67: 4, 68: 1, 69: 28, 70: 56, 71: 83, 72: 17, 73: 2, 74: 34, 75: 39, 76: 29, 77: 28, 78: 7, 79: 9, 80: 41, 81: 107, 82: 37, 83: 14, 84: 13, 85: 23, 86: 13, 87: 13, 88: 113, 89: 9, 91: 62, 92: 85, 93: 13, 94: 50, 95: 54, 97: 94, 98: 106, 99: 20}
    Top classes: raccoon (6.3%), bus (3.9%), maple_tree (3.6%), tiger (3.5%), man (3.4%)
    
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
        logger.exception(f"Exception in {mcp.name}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info(f"Starting MCP server for {mcp.name}...")
    mcp.run(transport='stdio')