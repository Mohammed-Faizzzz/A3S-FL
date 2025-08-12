from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv

# Initialize FastMCP server
mcp = FastMCP("a3s-server1") # enables the selectable MCP server with the name "weather"

# Load environment variables
load_dotenv()

# Get environment variables
private_key = os.getenv("PRIVATE_KEY")
base_url = os.getenv("RESOURCE_SERVER_URL")
endpoint_path = os.getenv("ENDPOINT_PATH")

if not all([private_key, base_url, endpoint_path]):
    print("Error: Missing required environment variables")
    exit(1)

from huggingface_hub import HfApi, HfFolder, Repository
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import joblib
import os

@mcp.tool()
async def get_data_info() -> str:
    """
    Total Samples: 4190
    Number of Unique Classes: 11
    Class Skewness (Std Dev): 166.39
    Class Distribution: {2: 471, 23: 170, 28: 3, 51: 499, 58: 499, 66: 499,
    69: 490, 77: 220, 81: 341, 83: 499, 84: 499}
    """

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    