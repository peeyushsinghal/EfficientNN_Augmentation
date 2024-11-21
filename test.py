import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import glob
import os
from model import MNISTModel 

def get_latest_model():
    """Get the most recent model file"""
    model_files = glob.glob('models/*.pth')
    if not model_files:
        raise FileNotFoundError("No model files found")
    return max(model_files, key=os.path.getctime)

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Load the latest model
    model_path = get_latest_model()
    print(f"Loading model from {model_path}")
    model = MNISTModel()  # Load the model architecture
    model.load_state_dict(torch.load(model_path))  # Load the state dictionary
    model.to(device)  # Move the model to the appropriate device
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

if __name__ == '__main__':
    test_model() 