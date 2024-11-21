import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTModel
import os

def calculate_accuracy(output, target):
    """Calculate accuracy for current batch"""
    _, predicted = torch.max(output.data, 1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    return 100 * correct / total

def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset with augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=7),  # Random rotation ±7 degrees
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training metrics
    running_loss = 0.0
    running_acc = 0.0
    log_interval = 100
    
    # Train for 1 epoch
    model.train()
    print(f"Training started on {device}")
    print("Data augmentation: Random rotation ±7 degrees")
    print("-" * 50)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        accuracy = calculate_accuracy(output, target)
        
        # Update running metrics
        running_loss += loss.item()
        running_acc += accuracy
        
        if batch_idx % log_interval == 0:
            avg_loss = running_loss / (log_interval if batch_idx > 0 else 1)
            avg_acc = running_acc / (log_interval if batch_idx > 0 else 1)
            
            print(f'Batch {batch_idx}/{len(train_loader)} '
                  f'Loss: {avg_loss:.4f} '
                  f'Accuracy: {avg_acc:.2f}%')
            
            running_loss = 0.0
            running_acc = 0.0
    
    print(f"\nTraining completed on {device}")
    
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), 'models/mnist_model.pth')
    return model

if __name__ == '__main__':
    train_model() 