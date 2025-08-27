import os, sys, io, base64, logging, argparse
from typing import Any, Dict

import torch, torch.nn as nn, numpy as np, httpx
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# === Parse arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--client-id", type=int, required=True)
args = parser.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.cnn_model import CNN

client_name = f"a3s-client-{args.client_id}"
mcp = FastMCP(client_name)

LOCAL_DATA_PATH = os.path.join(ROOT, "data", f"client_{args.client_id}.pt")

# logging per client
logfile = os.path.join(ROOT, "clients", "logs", f"{client_name}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(logfile, mode="a"),
        logging.StreamHandler(sys.__stderr__)
    ]
)
logger = logging.getLogger(client_name)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]

def _load_local_dataloader(batch_size=32):
    data = torch.load(LOCAL_DATA_PATH)
    dataset = MyDataset(data["x"], data["y"])

    x_sample = data["x"].float()
    y_sample = data["y"]

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

    logger.info(f"Client {client_name} finished local training for {epochs} epochs.")
    return model.state_dict()

@mcp.tool()
async def train_model_with_local_data(global_model_params: str, epochs: int = 1) -> Dict[str, Any]:
    """
    Performs federated learning on a client with a non-IID subset of the
    CIFAR-100 dataset.
    Samples: 6387
    Unique classes: 87
    Class distribution: {0: 51, 1: 41, 2: 277, 4: 41, 5: 264, 6: 5, 7: 41, 8: 35, 9: 59, 10: 49, 11: 44, 12: 6, 13: 141, 14: 1, 16: 16, 17: 57, 18: 5, 19: 9, 20: 56, 21: 10, 22: 3, 23: 60, 24: 4, 25: 194, 26: 34, 27: 5, 28: 26, 29: 4, 30: 60, 31: 11, 32: 103, 33: 30, 34: 16, 36: 26, 37: 72, 39: 2, 40: 185, 44: 3, 45: 197, 46: 1, 47: 96, 49: 14, 50: 168, 51: 42, 52: 92, 53: 306, 54: 7, 55: 139, 56: 19, 57: 3, 58: 159, 59: 21, 60: 12, 61: 20, 62: 172, 63: 8, 65: 11, 66: 126, 67: 14, 68: 376, 69: 48, 70: 3, 72: 37, 74: 136, 75: 51, 76: 33, 77: 13, 78: 6, 80: 114, 81: 9, 82: 84, 83: 245, 84: 114, 85: 234, 86: 12, 87: 24, 88: 171, 89: 195, 90: 97, 91: 76, 92: 172, 93: 83, 95: 108, 96: 112, 97: 91, 98: 6, 99: 64}
    Top classes: road (5.9%), orange (4.8%), baby (4.3%), bed (4.1%), sweet_pepper (3.8%)
    
    Args:
        global_model_params: A dictionary of the global model's parameters.
        epochs: The number of local training epochs.
    
    Returns:
        A dictionary of the locally trained model's updated parameters.
    """
    try:
        local_dataloader = _load_local_dataloader()
        global_sd = _b64_to_state_dict(global_model_params)
        updated_sd = _train_model_locally(global_sd, local_dataloader, epochs)
        logger.info(f"Returning {len(updated_sd)} params and {len(local_dataloader.dataset)} samples")
        return {
            "params_b64": _state_dict_to_b64(updated_sd),
            "num_samples": len(local_dataloader.dataset)
        }

    except Exception as e:
        logger.exception(f"Exception in {client_name}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info(f"Starting MCP server for {client_name}...")
    mcp.run(transport='stdio')
    