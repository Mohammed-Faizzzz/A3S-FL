from typing import Any, Dict
from mcp.server.fastmcp import FastMCP
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv

# Import the base model architecture
from models.cnn_model import CNN
mcp = FastMCP("a3s-client-0")

load_dotenv()
private_key = os.getenv("PRIVATE_KEY")
base_url = os.getenv("RESOURCE_SERVER_URL")
endpoint_path = os.getenv("ENDPOINT_PATH")

if not all([private_key, base_url, endpoint_path]):
    print("Error: Missing required environment variables")
    exit(1)

# --- Internal Data and Training Methods ---

LOCAL_DATA_PATH = "./data/client_0.npz"

class MyDataset(Dataset):
    """
    A simple Dataset class to load pre-split data from a .npz file.
    """
    def __init__(self, data_path: str):
        data = np.load(data_path, allow_pickle=True)
        self.images = torch.from_numpy(data['images']).float()
        self.labels = torch.from_numpy(data['labels']).long()
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Any:
        return self.images[idx], self.labels[idx]

def _load_local_dataloader() -> DataLoader:
    """
    Internal method to load and return the client's local data loader.
    """
    try:
        local_dataset = MyDataset(LOCAL_DATA_PATH)
        return DataLoader(local_dataset, batch_size=32, shuffle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Local data not found at {LOCAL_DATA_PATH}")

def _train_model_locally(model_params: Dict[str, torch.Tensor], dataloader: DataLoader, epochs: int = 1) -> Dict[str, torch.Tensor]:
    """
    Internal training logic for a single client.
    """
    model = CNN()
    model.load_state_dict(model_params)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    
    print(f"Client {mcp.name} finished local training for {epochs} epochs.")
    return model.state_dict()

@mcp.tool()
async def train_model_with_local_data(global_model_params: Dict[str, torch.Tensor], epochs: int = 1) -> Dict[str, torch.Tensor]:
    """
    Performs federated learning on a client with a non-IID subset of the
    CIFAR-100 dataset.
    total_samples: 7926,
    num_unique_classes: 20,
    class_skewness_std_dev: 148.39,
    class_distribution:
    {0: 450, 1: 369, 3: 201, 4: 500, 6: 62, 7: 499, 10: 500,
    23: 329, 32: 499, 34: 314, 35: 491, 36: 499, 38: 139, 47: 499,
    62: 499, 70: 100, 73: 496, 91: 499, 94: 499, 97: 482}
   
    
    Args:
        global_model_params: A dictionary of the global model's parameters.
        epochs: The number of local training epochs.
    
    Returns:
        A dictionary of the locally trained model's updated parameters.
    """
    try:
        local_dataloader = _load_local_dataloader()
        updated_params = _train_model_locally(global_model_params, local_dataloader, epochs)
        return updated_params
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print(f"Starting MCP server for {mcp.name}...")
    mcp.run(transport='stdio')