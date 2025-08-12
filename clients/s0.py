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
    Total Samples: 7926
    Number of Unique Classes: 20
    Class Skewness (Std Dev): 148.39
    Class Distribution: {0: 450, 1: 369, 3: 201, 4: 500, 6: 62,
    7: 499, 10: 500, 23: 329, 32: 499, 34: 314, 35: 491, 36: 499,
    38: 139, 47: 499, 62: 499, 70: 100, 73: 496, 91: 499, 94: 499, 97: 482}
    """

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    