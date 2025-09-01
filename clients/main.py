import os, sys, io, base64, logging, argparse, json
from typing import Any, Dict

from dataclasses import dataclass
import torch, torch.nn as nn, numpy as np, httpx
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

@dataclass
class LocalKnobs:
    use_LB: bool = False
    lambda_prior: float = 0.0
    alpha_parallel: float = 0.1
    alpha_perp: float = 0.2
    rho: float = 0.0
    gamma_perp: float = 0.8

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LocalKnobs":
        return cls(**d)

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

with open(os.path.join(ROOT, "data", "client_metadata.json")) as f:
    CLIENT_DESCRIPTIONS = json.load(f)
    
stats = CLIENT_DESCRIPTIONS.get(str(args.client_id))
if stats:
    description = (
        f"Train on client {args.client_id}'s CIFAR-100 shard. "
        f"Samples: {stats['total_samples']}, "
        f"Unique classes: {stats['unique_classes']}, "
        f"Top classes: {', '.join(stats['top_classes'])}."
    )
else:
    description = f"Train on client {args.client_id}'s dataset shard."

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

def _train_model_locally(
    model_params: Dict[str, torch.Tensor],
    dataloader: DataLoader,
    epochs: int = 10,
    knobs: LocalKnobs = LocalKnobs()
) -> Dict[str, torch.Tensor]:
    """
    Internal training logic for a single client.
    All LocalKnobs parameters are used.
    """
    model = CNN()
    model.load_state_dict(model_params)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    # === Use knobs for optimizer ===
    # scale learning rate by (1 - lambda_prior), clip at min 1e-4
    base_lr = 0.01
    lr = max(1e-4, base_lr * (1.0 - 0.5 * knobs.lambda_prior))

    # momentum from rho (capped at [0,1])
    momentum = min(max(knobs.rho, 0.0), 0.99)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=1e-4 * knobs.alpha_parallel,  # L2 scaled by alpha_parallel
    )

    logger.info(f"[{client_name}] Training with knobs={knobs}")

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # === Use knobs in loss ===
            # extra regularization if LB is enabled
            if knobs.use_LB:
                reg = sum(torch.norm(p) for p in model.parameters())
                loss += knobs.lambda_prior * 1e-4 * reg

            # scale loss perpendicular vs parallel
            loss = loss * (1.0 + knobs.alpha_perp) * (1.0 - 0.5 * knobs.gamma_perp)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        logger.info(
            f"Epoch {epoch+1}/{epochs}, loss={running_loss/len(dataloader):.4f}"
        )

    logger.info(f"Client {client_name} finished local training for {epochs} epochs.")
    return model.state_dict()

@mcp.tool(
    name="train_model_with_local_data",
    description=description
)
async def train_model_with_local_data(
    global_model_params: str,
    epochs: int = 10,
    knobs: Dict[str, Any] = None
) -> Dict[str, Any]:
    try:
        local_dataloader = _load_local_dataloader()
        global_sd = _b64_to_state_dict(global_model_params)
        # if not LocalKnobs, instantiate default
        local_knobs = LocalKnobs() if not knobs else LocalKnobs.from_dict(knobs)

        updated_sd = _train_model_locally(global_sd, local_dataloader, epochs, local_knobs)

        logger.info(
            f"Returning {len(updated_sd)} params and {len(local_dataloader.dataset)} samples"
        )
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
    