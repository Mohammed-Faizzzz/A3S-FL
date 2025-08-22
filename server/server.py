import asyncio
from contextlib import AsyncExitStack
from typing import Dict, List, Tuple
import base64, io, torch
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os, sys, json
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.cnn_model import CNN  # your base model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[orchestrator] Using device: {DEVICE}")

# Hardcoded client configs
CLIENTS = [
    ("a3s-client-0", "clients/a3s-client-0/main.py"),
    ("a3s-client-1", "clients/a3s-client-1/main.py"),
    ("a3s-client-2", "clients/a3s-client-2/main.py"),
    ("a3s-client-3", "clients/a3s-client-3/main.py"),
    ("a3s-client-4", "clients/a3s-client-4/main.py"),
    ("a3s-client-5", "clients/a3s-client-5/main.py"),
    ("a3s-client-6", "clients/a3s-client-6/main.py"),
    ("a3s-client-7", "clients/a3s-client-7/main.py"),
    ("a3s-client-8", "clients/a3s-client-8/main.py"),
    ("a3s-client-9", "clients/a3s-client-9/main.py")
]

UV_CMD = "uv"

test_data = torch.load("./../data/global_test_dataset.pt")
test_dataset = TensorDataset(test_data["x"], test_data["y"])
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# === Utility functions ===
def state_dict_to_b64(sd: Dict[str, torch.Tensor]) -> str:
    buf = io.BytesIO()
    torch.save(
        {k: (v.cpu().half() if v.dtype.is_floating_point else v.cpu()) for k, v in sd.items()},
        buf,
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")

def b64_to_state_dict(b64: str) -> Dict[str, torch.Tensor]:
    raw = base64.b64decode(b64.encode("ascii"))
    buf = io.BytesIO(raw)
    sd = torch.load(buf, map_location="cpu")
    # cast back to float32 so training is unaffected
    for k, v in sd.items():
        if v.dtype == torch.float16:
            sd[k] = v.float()
    return sd

def evaluate(model: CNN, test_loader):
    model.eval()
    model.to(DEVICE)  # ensure model is on GPU if available
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"[orchestrator] Test Accuracy: {acc:.4f}")
    return acc


# === Orchestrator ===
class Orchestrator:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}

    async def connect_all(self):
        for name, script in CLIENTS:
            print(f"[orchestrator] Connecting to {name}...")

            client_dir = os.path.abspath(os.path.join(ROOT, script.rsplit("/", 1)[0]))
            client_script = script.rsplit("/", 1)[1]

            params = StdioServerParameters(
                command=UV_CMD,
                args=["--directory", client_dir, "run", client_script],
                env=None,
            )
            transport = await self.exit_stack.enter_async_context(stdio_client(params))
            stdio, write = transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            tools = await session.list_tools()
            print(f"[orchestrator] {name} tools: {[t.name for t in tools.tools]}")
            self.sessions[name] = session

    async def close(self):
        await self.exit_stack.aclose()
    
    async def train_round(self, global_model: CNN, epochs: int = 1) -> CNN:
        global_sd_b64 = state_dict_to_b64(global_model.state_dict())

        updates: List[Tuple[Dict[str, torch.Tensor], int]] = []
        for name, session in self.sessions.items():
            print(f"[orchestrator] Sending model to {name} for local training...")
            result = await session.call_tool(
                "train_model_with_local_data",
                {"global_model_params": global_sd_b64, "epochs": epochs},
            )

            if not result or not getattr(result, "content", None):
                print(f"[orchestrator] {name} error: no result content")
                continue

            payload = result.content[0].text if hasattr(result.content[0], "text") else result.content[0]
            if not payload:
                print(f"[orchestrator] {name} error: empty payload")
                continue

            try:
                data = json.loads(payload) if isinstance(payload, str) else payload
            except json.JSONDecodeError as e:
                print(f"[orchestrator] {name} error decoding JSON: {e}")
                continue

            if "error" in data:
                print(f"[orchestrator] {name} error: {data['error']}")
                continue

            updated_sd = b64_to_state_dict(data["params_b64"])
            updates.append((updated_sd, data["num_samples"]))

        # FedAvg aggregation
        if not updates:
            print("[orchestrator] No updates received this round.")
            return global_model

        # new_global_sd = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
        new_global_sd = {k: torch.zeros_like(v, device="cpu") for k, v in global_model.state_dict().items()}

        total_samples = sum(num for _, num in updates)
        
        for sd, num_samples in updates:
            weight = num_samples / total_samples
            for k in new_global_sd.keys():
                v = sd[k].cpu()   # ensure on CPU
                if new_global_sd[k].dtype.is_floating_point:
                    new_global_sd[k] += v * weight
                else:
                    new_global_sd[k] = v

        global_model.load_state_dict(new_global_sd)
        global_model.to(DEVICE)
        return global_model

# === Entry point ===
async def main(save_path: str):
    orch = Orchestrator()
    accuracies = []

    try:
        await orch.connect_all()
        global_model = CNN()
        global_model.to(DEVICE)

        ROUNDS = 15
        for r in range(ROUNDS):
            print(f"\n[orchestrator] ===== Round {r+1} =====")
            global_model = await orch.train_round(global_model, epochs=1)
            
            acc = evaluate(global_model, test_loader)
            accuracies.append(acc)

            print(f"[orchestrator] Saved global model after round {r+1}")

        # Save results for plotting
        torch.save(global_model.state_dict(), f"global_model_round.pth")
        np.save(save_path, np.array(accuracies))
        print(f"[orchestrator] Accuracies saved to {save_path}")

    finally:
        await orch.close()


if __name__ == "__main__":
    asyncio.run(main("results/fedavg.npy"))
