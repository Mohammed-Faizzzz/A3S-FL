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
mcp = FastMCP("a3s-client-1")

load_dotenv()
private_key = os.getenv("PRIVATE_KEY")
base_url = os.getenv("RESOURCE_SERVER_URL")
endpoint_path = os.getenv("ENDPOINT_PATH")

# --- Internal Data and Training Methods ---

LOCAL_DATA_PATH = os.path.join(ROOT, "data", "client_1.pt")


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
    Samples: 6004
    Unique classes: 89
    Class distribution: {0: 175, 1: 55, 2: 56, 5: 3, 6: 3, 7: 3, 8: 83, 9: 43, 10: 14, 11: 76, 12: 51, 13: 66, 14: 7, 15: 4, 16: 5, 18: 8, 19: 8, 20: 9, 21: 41, 22: 5, 23: 146, 24: 32, 26: 46, 27: 110, 28: 12, 29: 2, 30: 95, 31: 27, 32: 153, 33: 91, 34: 30, 35: 4, 36: 230, 37: 3, 39: 2, 40: 1, 41: 162, 42: 115, 43: 96, 44: 121, 45: 32, 46: 58, 47: 3, 48: 300, 49: 20, 50: 77, 51: 62, 52: 151, 53: 2, 54: 198, 55: 144, 56: 2, 57: 9, 58: 7, 59: 47, 60: 298, 61: 11, 62: 55, 63: 107, 66: 58, 67: 148, 68: 60, 69: 136, 71: 181, 72: 27, 73: 111, 75: 6, 77: 95, 78: 138, 79: 168, 80: 166, 81: 84, 82: 136, 83: 2, 84: 22, 85: 9, 86: 15, 87: 21, 89: 61, 90: 53, 91: 106, 92: 30, 93: 41, 94: 152, 95: 8, 96: 16, 97: 109, 98: 2, 99: 67}
    Top classes: motorcycle (5.0%), plain (5.0%), hamster (3.8%), orchid (3.3%), sea (3.0%)
    
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