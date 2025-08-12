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
    Total Samples: 6490
    Number of Unique Classes: 21
    Class Skewness (Std Dev): 184.39
    Class Distribution: {1: 130, 6: 296, 17: 499, 18: 499, 33: 499,
    38: 39, 42: 32, 48: 499, 52: 345, 53: 499, 54: 371, 63: 423,
    74: 206, 77: 275, 79: 498, 80: 476, 85: 3, 86: 282, 93: 103, 96: 499, 97: 17}
    """

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    