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
    Total Samples: 5370
    Number of Unique Classes: 22
    Class Skewness (Std Dev): 226.64
    Class Distribution: {12: 37, 20: 131, 21: 39, 26: 499,
    27: 499, 31: 3, 35: 1, 39: 499, 41: 497, 45: 70, 49: 499,
    50: 22, 57: 494, 59: 499, 65: 499, 67: 499, 69: 9, 73: 3,
    81: 158, 86: 2, 93: 395, 95: 16}
    """

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    