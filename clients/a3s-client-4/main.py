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
mcp = FastMCP("a3s-client-4")

load_dotenv()
private_key = os.getenv("PRIVATE_KEY")
base_url = os.getenv("RESOURCE_SERVER_URL")
endpoint_path = os.getenv("ENDPOINT_PATH")

# --- Internal Data and Training Methods ---
LOCAL_DATA_PATH = os.path.join(ROOT, "data", "client_4.pt")

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
async def train_model_with_local_data(global_model_params: str, epochs: int = 1) -> Dict[str, Any]:
    """
    Performs federated learning on a client with a non-IID subset of the
    CIFAR-100 dataset.
    Samples: 4386
    Unique classes: 93
    Class distribution: {0: 66, 1: 14, 2: 13, 3: 34, 4: 35, 5: 19, 6: 6, 7: 19, 8: 72, 9: 14, 10: 26, 11: 1, 13: 55, 14: 213, 15: 208, 16: 60, 17: 43, 18: 43, 19: 120, 20: 76, 21: 3, 22: 45, 24: 40, 25: 1, 26: 51, 27: 96, 28: 270, 29: 1, 30: 28, 31: 1, 32: 107, 33: 67, 34: 65, 35: 17, 36: 42, 37: 1, 38: 9, 39: 24, 40: 2, 41: 31, 42: 7, 43: 1, 44: 16, 45: 3, 46: 119, 47: 82, 48: 22, 49: 108, 50: 4, 52: 20, 53: 2, 54: 11, 55: 4, 56: 64, 57: 29, 58: 129, 59: 68, 60: 28, 61: 5, 62: 9, 63: 25, 64: 17, 66: 29, 67: 251, 68: 5, 69: 7, 70: 37, 71: 16, 73: 15, 74: 3, 75: 31, 76: 43, 77: 78, 78: 21, 79: 4, 80: 29, 81: 11, 82: 1, 84: 61, 85: 33, 86: 138, 87: 15, 88: 109, 89: 5, 90: 16, 92: 48, 93: 120, 94: 17, 95: 23, 96: 71, 97: 57, 98: 197, 99: 84}
    Top classes: cup (6.2%), ray (5.7%), butterfly (4.9%), camel (4.7%), woman (4.5%)
    
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