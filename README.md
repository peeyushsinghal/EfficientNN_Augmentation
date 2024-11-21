[![ML Pipeline](https://github.com/peeyushsinghal/EfficientNN_Augmentation/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/peeyushsinghal/EfficientNN_Augmentation/actions/workflows/ml-pipeline.yml)
# MNIST CNN Classification Project

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification with an emphasis on model efficiency and automated testing through CI/CD.

## Project Structure
```
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml
├── models/ # Saved model directory
├── model.py # CNN architecture
├── train.py # Training script
├── test.py # Testing script
├── test_cases.py # Unit tests
├── train_pipeline.py # Training pipeline
├── main.py # Main execution script
├── requirements.txt # Project dependencies
└── pytest.ini # Pytest configuration
```

## Model Architecture
- Input: 28x28 grayscale images
- Multiple convolutional layers reducing to 1x1 spatial dimension
- Final fully connected layer with 10 outputs
- Total parameters: < 25,000
- Receptive Field: 29x29

## Requirements
- Python 3.9+
- PyTorch 2.2.0
- torchvision 0.17.0
- pytest 7.4.4
- pytest-cov 4.1.0


### CI/CD Pipeline

The project includes a GitHub Actions workflow that:
1. Runs all unit tests
2. Trains the model
3. Validates model performance
4. Saves the trained model as an artifact

## Model Performance
- Target accuracy: > 95% on test set
- Training: 1 epoch
- Batch size: 64
- Optimizer: Adam (lr=0.001)

## Test Cases
The automated tests verify:
- Model parameter count (< 25,000)
- Input shape handling (28x28)
- Output shape (10 classes)
- Training capability
- Inference capability
- Model accuracy (> 95%)
- Training dynamics (gradient flow)
- Hardware compatibility (device movement)
- Numerical stability (batch invariance)

## Model Saving
Models are saved with state dictionaries in the `models/` directory with the format: models/mnist_model.pth

## Notes
- The model is designed to be lightweight while maintaining high accuracy
- Training logs display both loss and accuracy metrics
- Data augmentation helps improve model robustness
- All tests must pass and accuracy must exceed 95% for successful CI/CD pipeline completion
