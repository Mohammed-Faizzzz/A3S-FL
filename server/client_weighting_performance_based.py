import asyncio
from contextlib import AsyncExitStack
from typing import Dict, List, Tuple
import base64, io, torch
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os, sys
import json
from anthropic import Anthropic
from dotenv import load_dotenv
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
print(ANTHROPIC_API_KEY)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

# Import the base model architecture
from models.cnn_model import CNN

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
anthropic = Anthropic()

test_data = torch.load("./../data/global_test_dataset.pt")
test_dataset = TensorDataset(test_data["x"], test_data["y"])
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Functions to help formatting changes between training rounds and MCP
def state_dict_to_b64(sd: Dict[str, torch.Tensor]) -> str:
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in sd.items()}, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def b64_to_state_dict(b64: str) -> Dict[str, torch.Tensor]:
    raw = base64.b64decode(b64.encode("ascii"))
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu")

def evaluate(model: CNN, test_loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"[orchestrator] Test Accuracy: {acc:.4f}")
        return acc


class Orchestrator:
    def __init__(self, user_goal: str):
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.tools_by_client: Dict[str, List[Dict[str, str]]] = {} # keep track of tools
        self.user_goal = user_goal
    
    async def connect_all(self):
        """
        Connect to all client sessions, a handshake and get all the info about the various tools.
        """
        for name, script in CLIENTS:
            print(f"[orchestrator] Connecting to {name}...")

            client_dir = os.path.abspath(os.path.join(ROOT, script.rsplit("/", 1)[0]))
            client_script = script.rsplit("/", 1)[1]
            
            # Debug print
            print(f"[debug] client_dir={client_dir}")
            print(f"[debug] client_script={client_script}")
            print(f"[debug] full_path={os.path.join(client_dir, client_script)}")

            # Check existence
            if not os.path.isdir(client_dir):
                print(f"[error] Directory does not exist: {client_dir}")
            if not os.path.isfile(os.path.join(client_dir, client_script)):
                print(f"[error] File does not exist: {os.path.join(client_dir, client_script)}")


            params = StdioServerParameters(
                command=UV_CMD,
                args=["--directory", client_dir, "run", client_script],
                env=None
            )
            transport = await self.exit_stack.enter_async_context(stdio_client(params))
            stdio, write = transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            tools = await session.list_tools()
            print(f"[orchestrator] {name} tools: {[t.name for t in tools.tools]}")
            self.tools_by_client[name] = [
                {"tool": t.name, "description": t.description} for t in tools.tools
            ]
            self.sessions[name] = session

    async def close(self):
        """
        Close all client sessions; opposite of connect_all method.
        """
        await self.exit_stack.aclose()

    async def train_round(self, global_model: CNN, round_num: int, epochs: int = 5) -> CNN:
        global_sd_b64 = state_dict_to_b64(global_model.state_dict())

        # === Step 1: Ask Claude which clients to use ===
        tool_info_json = json.dumps(self.tools_by_client, indent=2)

        msg = anthropic.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": f"""
    You are coordinating a federated learning training round.

    User goal: {self.user_goal}
    Round: {round_num}

    The following clients and their tools are available:
    {tool_info_json}

    Choose the most relevant clients and specify how many epochs to train.
    Respond ONLY in JSON with a list of objects like:
    [{{"client": "a3s-client-0", "tool": "train_model_with_local_data", "epochs": 5}}, ...]
    """
                }
            ],
        )

        try:
            plan = json.loads(msg.content[0].text)
            print("[orchestrator] Claude response plan:", plan) # CHECK THIS AND SEE WHAT OUTPUT YOU GET
        except Exception as e:
            print(f"[orchestrator] Claude response parse error: {e}")
            return global_model

        # === Step 2: Execute tool calls ===
        # 1. Reads Claude’s plan of which clients/tools to run
        # 2. Sends the global model to each client
        # 3. Collects their updated model weights and dataset size
        # 4. Filters out errors and invalid responses
        # 5. Builds a list of updates for federated averaging

        updates: List[Tuple[Dict[str, torch.Tensor], int]] = []
        for action in plan:
            client = action["client"]
            tool = action.get("tool", "train_model_with_local_data")  # default fallback
            local_epochs = action.get("epochs", epochs)

            if client not in self.sessions:
                print(f"[orchestrator] Unknown client {client}")
                continue

            print(f"[orchestrator] Sending model to {client} with tool {tool}...")
            result = await self.sessions[client].call_tool(
                tool,
                {"global_model_params": global_sd_b64, "epochs": local_epochs}
            )

            if not result or not getattr(result, "content", None):
                print(f"[orchestrator] {client} error: no result content")
                continue

            payload = result.content[0].text if hasattr(result.content[0], "text") else result.content[0]
            if not payload:
                print(f"[orchestrator] {client} error: empty payload")
                continue

            try:
                data = json.loads(payload) if isinstance(payload, str) else payload
            except json.JSONDecodeError as e:
                print(f"[orchestrator] {client} error decoding JSON: {e}")
                continue

            if "error" in data:
                print(f"[orchestrator] {client} error: {data['error']}")
                continue

            updated_sd = b64_to_state_dict(data["params_b64"])
            temp_model = CNN()
            temp_model.load_state_dict(updated_sd)
            acc = evaluate(temp_model, test_loader)
            updates.append((updated_sd, data["num_samples"], acc))

        # === Step 3: FedAvg aggregation ===
        if not updates:
            print("[orchestrator] No updates received this round.")
            return global_model

        new_global_sd = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
        total_samples = sum(num for _, num, _ in updates)

        # Normalize accuracies into weights
        total_acc = sum(acc for _, _, acc in updates)
        if total_acc == 0:  # fallback to equal weighting if all acc=0
            weights = [1/len(updates)] * len(updates)
        else:
            weights = [acc / total_acc for _, _, acc in updates]

        # Weighted aggregation
        for (sd, num_samples, _), weight in zip(updates, weights):
            for k in new_global_sd.keys():
                new_global_sd[k] += sd[k] * weight * (num_samples / total_samples)

        global_model.load_state_dict(new_global_sd)
        return global_model

async def main(user_goal: str, save_path: str):
    orch = Orchestrator(user_goal)
    accuracies = []

    try:
        await orch.connect_all()
        global_model = CNN()

        ROUNDS = 100
        for r in range(ROUNDS):
            print(f"\n[orchestrator] ===== Round {r+1} =====")
            global_model = await orch.train_round(global_model, round_num=r+1, epochs=1)
            torch.save(global_model.state_dict(), f"global_model_round_wpb.pth")
            
            acc = evaluate(global_model, test_loader)
            accuracies.append(acc)

            print(f"[orchestrator] Saved global model after round {r+1}")

        # Save results for plotting
        np.save(save_path, np.array(accuracies))
        print(f"[orchestrator] Accuracies saved to {save_path}")

    finally:
        await orch.close()

if __name__ == "__main__":
    goal = input("Enter your training goal (e.g. 'detect wounds'): ")
    asyncio.run(main(goal, "results/perf_weight.npy"))
