import asyncio
from contextlib import AsyncExitStack
from typing import Dict, List, Tuple
import base64, io, torch
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os, sys
import json

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

def state_dict_to_b64(sd: Dict[str, torch.Tensor]) -> str:
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in sd.items()}, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def b64_to_state_dict(b64: str) -> Dict[str, torch.Tensor]:
    raw = base64.b64decode(b64.encode("ascii"))
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu")

class Orchestrator:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}

    async def connect_all(self):
        for name, script in CLIENTS:
            print(f"[orchestrator] Connecting to {name}...")
            params = StdioServerParameters(
                command=UV_CMD,
                args=["--directory", str(script).rsplit("/", 1)[0], "run", str(script).rsplit("/", 1)[1]],
                env=None
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
                {"global_model_params": global_sd_b64, "epochs": epochs}
            )

            # Ensure result is valid and has content
            if not result or not getattr(result, "content", None):
                print(f"[orchestrator] {name} error: no result content")
                continue

            payload = None
            if hasattr(result.content[0], "text"):
                payload = result.content[0].text
            else:
                payload = result.content[0]
            # print(f"[orchestrator] {name} result: {payload}")

            if not payload:  # empty string, None, etc.
                print(f"[orchestrator] {name} error: empty payload")
                continue

            import json
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

        # Aggregate via FedAvg
        new_global_sd = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
        total_samples = sum(num for _, num in updates)
        for sd, num_samples in updates:
            weight = num_samples / total_samples
            for k in new_global_sd.keys():
                new_global_sd[k] += sd[k] * weight

        global_model.load_state_dict(new_global_sd)
        return global_model

async def main():
    orch = Orchestrator()
    try:
        await orch.connect_all()

        # Init global model
        global_model = CNN()

        ROUNDS = 3
        for r in range(ROUNDS):
            print(f"\n[orchestrator] ===== Round {r+1} =====")
            global_model = await orch.train_round(global_model, epochs=1)
            torch.save(global_model.state_dict(), f"global_model_round_{r+1}.pth")
            print(f"[orchestrator] Saved global model after round {r+1}")

        print("[orchestrator] Training complete.")

        await asyncio.sleep(999999)  # Keep connections alive if needed
    finally:
        await orch.close()

if __name__ == "__main__":
    asyncio.run(main())
