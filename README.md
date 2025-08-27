# Federated Learning with Agent-as-a-Service (A3S)

### 1. Architecture Overview

Our proposed architecture is a novel paradigm called **Agent-as-a-Service (A3S)**, which integrates an autonomous AI agent with **Model Context Protocol (MCP)**-compliant servers. This framework is designed to overcome the limitations of traditional federated learning (FL) by introducing a strategic, adaptive layer for orchestrating training tasks.

The system consists of three main components:

* **Central AI Agent (A3S Agent):** The central orchestrator of the entire federated learning process. Unlike a traditional server with a fixed script, this agent is designed to be intelligent and adaptive. It makes dynamic decisions about which clients to engage, when to engage them, and what parameters to use, all based on the information provided by the MCP servers.
* **MCP Servers (Clients):** These are the participants in the federated learning process. Each server hosts a specific, non-IID subset of the data (e.g., a portion of the CIFAR-100 dataset). Critically, each MCP server exposes its capabilities and dataset characteristics through the **Model Context Protocol (MCP)**.
* **Model Context Protocol (MCP):** A standardized communication protocol that allows the AI agent to interact with the MCP servers. It defines the rules for exchanging model updates and, most importantly, provides a structured way for clients to describe their data and tools to the central agent.



---

### 2. The Training Flow

The federated training process is not a simple, repetitive loop. Instead, it is a dynamic cycle orchestrated by the central AI agent.

1.  **Initialization:** The central AI agent initializes a global model and sends it to a strategically selected subset of MCP servers.
2.  **Strategic Client Selection:** At the beginning of each training round, the AI agent queries the available MCP servers for their context. It then analyzes the servers' descriptions of their datasets (e.g., class distribution, sample size, skewness) to select the most relevant clients for that specific round.
3.  **Local Training:** The selected MCP servers receive the global model and train it locally on their private, non-IID data. All raw data remains on the individual servers.
4.  **Model Update Transmission:** After local training, each MCP server uses the **MCP** to send its updated model parameters (weights and biases) back to the central AI agent. The raw data is never transmitted.
5.  **Agent-Driven Aggregation:** The AI agent collects the updated models and performs a weighted aggregation, creating a new, improved global model. The agent can adjust this aggregation process based on its strategic goals (e.g., giving more weight to updates from clients with underrepresented data).
6.  **Iteration:** This process repeats for multiple rounds, with the agent continuously adapting its strategy to improve the global model's performance and efficiency.

---

### 3. Key Differentiators and Benefits

This architecture provides significant advantages over a standard federated learning setup:

* **Dynamic and Intelligent Orchestration:** The AI agent replaces a rigid, deterministic script with an adaptive manager, allowing the system to respond to real-time changes, client availability, and data distribution shifts.
* **Targeted Client Engagement:** By leveraging the MCP's descriptive capabilities, the agent can strategically select clients. This makes training more efficient and can help address model biases more effectively than random sampling.
* **Enhanced Security and Privacy:** The MCP ensures that all communication is standardized and secure, reinforcing the core privacy-preserving tenet of federated learning.
* **Improved Diagnosability:** The agent's decision-making process can be logged and audited. This transparency allows us to understand **why** certain clients were chosen or why a specific training path was taken, addressing potential "black box" concerns.

---

### 4. Set Up
 1. First, install uv and set up our Python project and environment:
    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

 2. Then, load the dependencies using the following command:
     ```
     uv sync
     ```
    You should have gotten a `uv.lock` file now.

 3. In the `./data` folder, you will see a `dirichlet0.5_client10.json` file. Replace it with your preferred data after splitting. (Will potentially replace this with a python script to perform the split ourselves.)

 4. Run the `data_processing.py` script. This will convert the data into `.pt` files for access by the client. We use `.pt` files over `.npz` files as `.npz` files incur additional time overhead from unzipping at each iteration. This will also generate `global_test_dataset.pt`, the test dataset that we will use for testing the accuracy of our global model after each round. Finally, it will also generate `client_metadata.json`. This file contains semantic data about the different data splits assigned to the various clients. We will be using this semantic data to modify our MCP tool's descriptions dynamically. This is important as AI Agents will be using the semantic data to make informed decisions on which clients to use at the various rounds.

 5. From the root of the folder, run the following command:
    ```
    uv run server/server.py
    ```
    This will launch the FedAvg training script, which in turn will spawn MCP tools dynamically. The number of MCP tools to spawn can be modified in this script.
