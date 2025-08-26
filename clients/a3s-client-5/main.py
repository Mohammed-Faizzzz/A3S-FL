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
mcp = FastMCP("a3s-client-5")

load_dotenv()
private_key = os.getenv("PRIVATE_KEY")
base_url = os.getenv("RESOURCE_SERVER_URL")
endpoint_path = os.getenv("ENDPOINT_PATH")

# --- Internal Data and Training Methods ---
LOCAL_DATA_PATH = os.path.join(ROOT, "data", "client_5.pt")

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
    Samples: 4599
    Unique classes: 94
    Class distribution: {0: 11, 1: 217, 2: 3, 3: 9, 4: 41, 5: 1, 6: 141, 7: 43, 8: 133, 9: 2, 10: 224, 11: 32, 12: 1, 13: 14, 14: 11, 15: 29, 16: 25, 17: 14, 18: 271, 19: 51, 20: 140, 21: 1, 22: 4, 23: 1, 24: 117, 25: 28, 26: 204, 27: 7, 28: 21, 29: 103, 30: 16, 31: 26, 33: 3, 34: 48, 35: 9, 36: 8, 37: 37, 38: 9, 39: 278, 40: 93, 41: 13, 42: 222, 43: 3, 44: 26, 45: 20, 46: 9, 49: 42, 50: 85, 51: 15, 52: 69, 53: 28, 54: 15, 55: 9, 56: 1, 57: 5, 58: 1, 60: 11, 61: 1, 62: 12, 63: 62, 64: 19, 65: 11, 66: 6, 67: 2, 68: 7, 69: 169, 70: 4, 71: 15, 72: 14, 73: 8, 74: 50, 75: 13, 76: 231, 77: 53, 78: 83, 80: 15, 81: 28, 82: 41, 83: 4, 84: 44, 85: 8, 86: 12, 87: 16, 88: 19, 90: 36, 91: 51, 92: 31, 93: 55, 94: 138, 95: 134, 96: 41, 97: 71, 98: 11, 99: 84}
    Top classes: keyboard (6.0%), caterpillar (5.9%), skyscraper (5.0%), bowl (4.9%), leopard (4.8%)

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