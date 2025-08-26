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
mcp = FastMCP("a3s-client-2")

load_dotenv()
private_key = os.getenv("PRIVATE_KEY")
base_url = os.getenv("RESOURCE_SERVER_URL")
endpoint_path = os.getenv("ENDPOINT_PATH")

# --- Internal Data and Training Methods ---

LOCAL_DATA_PATH = os.path.join(ROOT, "data", "client_2.pt")

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
    Samples: 4392
    Unique classes: 97
    Class distribution: {0: 88, 1: 24, 2: 3, 3: 27, 4: 33, 5: 15, 6: 91, 7: 58, 8: 80, 9: 344, 10: 66, 11: 15, 12: 13, 13: 3, 14: 19, 15: 55, 16: 95, 17: 3, 18: 7, 19: 21, 20: 80, 21: 1, 22: 28, 23: 35, 24: 9, 25: 74, 26: 4, 27: 114, 28: 62, 29: 53, 30: 154, 31: 5, 32: 90, 33: 72, 34: 23, 35: 17, 36: 32, 37: 6, 38: 134, 39: 1, 40: 4, 41: 159, 42: 3, 43: 38, 44: 82, 45: 27, 46: 29, 47: 58, 49: 258, 50: 7, 51: 47, 52: 3, 53: 19, 54: 4, 55: 66, 56: 23, 57: 5, 58: 4, 59: 6, 60: 16, 61: 13, 62: 24, 63: 8, 64: 122, 65: 1, 66: 1, 67: 2, 68: 2, 69: 16, 70: 4, 72: 95, 73: 23, 74: 61, 75: 53, 76: 17, 77: 36, 78: 2, 79: 6, 80: 4, 81: 34, 82: 31, 83: 11, 85: 19, 86: 2, 87: 184, 88: 25, 89: 1, 90: 157, 91: 8, 92: 42, 93: 164, 94: 12, 95: 88, 96: 117, 97: 1, 98: 91, 99: 3}
    Top classes: bottle (7.8%), mountain (5.9%), television (4.2%), turtle (3.7%), lawn_mower (3.6%)    
    
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