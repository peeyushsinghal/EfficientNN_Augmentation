import os
import torch
from datetime import datetime
from train import train_model
from test import test_model
import unittest
from test_cases import TestMNISTModel

def create_model_directory():
    """Create models directory if it doesn't exist"""
    if not os.path.exists('models'):
        os.makedirs('models')

def get_model_filename():
    """Generate model filename with timestamp and device info"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return f'models/mnist_model_{timestamp}_{device}.pth'

def run_pipeline():
    # Create models directory
    create_model_directory()
    
    # Run unit tests first
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromTestCase(TestMNISTModel)
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    if not test_result.wasSuccessful():
        raise Exception("Unit tests failed")
    
    print("Unit tests passed successfully")
    
    # Train model
    print("\nTraining model...")
    model = train_model()
    
    # Save model with timestamp
    model_path = get_model_filename()
    torch.save(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Validate model
    print("\nValidating model...")
    accuracy = test_model()
    
    # Check if accuracy meets requirement
    if accuracy < 80:
        raise Exception(f"Model accuracy ({accuracy:.2f}%) is below the required 80%")
    
    print(f"\nValidation successful! Final accuracy: {accuracy:.2f}%")
    
    # Create a simple results file
    with open('models/results.txt', 'a') as f:
        f.write(f"{model_path}: {accuracy:.2f}%\n")

if __name__ == '__main__':
    run_pipeline() 