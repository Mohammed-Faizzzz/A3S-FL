import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Import the necessary components from your project files
from models.cnn_model import CNN
from server.server_main import Server

# --- Data Splitting and Preparation ---
def non_iid_split_cifar100(dataset, num_clients=10, alpha=0.01):
    """
    Splits a dataset among clients in a non-IID manner using a Dirichlet distribution.
    
    Args:
        dataset: The full CIFAR-100 dataset.
        num_clients: The number of clients to split the data for.
        alpha: The concentration parameter for the Dirichlet distribution. A smaller alpha 
               means a more severe non-IID split.
    
    Returns:
        A list of data indices for each client.
    """
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        # Get all indices for the current class
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)
        
        # Split indices among clients based on a Dirichlet distribution
        probabilities = np.random.dirichlet(np.repeat(alpha, num_clients))
        probabilities = (probabilities * len(class_indices)).astype(int)
        
        # Adjust for any rounding errors
        probabilities[-1] += len(class_indices) - np.sum(probabilities)

        start = 0
        for i in range(num_clients):
            end = start + probabilities[i]
            client_indices[i].extend(class_indices[start:end])
            start = end
            
    return client_indices

# --- Main Simulation Logic ---
if __name__ == "__main__":
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    ])

    # Load the CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Split the data in a non-IID way for 10 clients
    num_clients = 10
    client_indices = non_iid_split_cifar100(train_dataset, num_clients, alpha=0.01)

    # Create a list of DataLoaders, one for each client
    client_dataloaders = [
        DataLoader(Subset(train_dataset, indices), batch_size=32, shuffle=True)
        for indices in client_indices
    ]

    print("Data successfully split and assigned to 10 clients.")
    print("-" * 50)

    # Initialize the Server (which in turn initializes the Clients)
    # The Server will create its own global model and distribute it.
    server = Server(num_clients=num_clients, client_dataloaders=client_dataloaders)

    # Start the federated learning training process
    # This will run for 10 communication rounds
    final_global_model = server.federated_averaging_training(num_communication_rounds=10)

    # (Optional) Evaluate the final model on the global test set
    print("-" * 50)
    print("Evaluating final global model on the full test set...")
    final_global_model.eval()
    correct = 0
    total = 0
    test_loader = DataLoader(test_dataset, batch_size=128)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = final_global_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Final Model Test Accuracy: {accuracy:.2f}%")