import numpy as np
import matplotlib.pyplot as plt

methods = {
    "FedAvg": "results/fedavg.npy",
    # "Agent Select": "results/agent_select.npy",
    # "Agent Weight": "results/agent_weight.npy",
    # "Perf Weight": "results/perf_weight.npy",
}

plt.figure(figsize=(10,6))
for name, path in methods.items():
    accs = np.load(path)
    plt.plot(range(1, len(accs)+1), accs, label=name)

plt.xlabel("Round")
plt.ylabel("Test Accuracy")
plt.title("Federated Learning Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.savefig("results/comparison.png")
plt.show()
