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
    Total Samples: 3354
    Number of Unique Classes: 10
    Class Skewness (Std Dev): 205.87
    Class Distribution: {16: 499, 38: 223, 44: 499, 52: 154, 56: 499, 75: 499, 79: 1, 80: 1, 85: 496, 95: 483}
    """

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    