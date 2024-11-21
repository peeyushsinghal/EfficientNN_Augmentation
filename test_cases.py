import unittest
import torch
from model import MNISTModel
import numpy as np

class TestMNISTModel(unittest.TestCase):
    def setUp(self):
        self.model = MNISTModel()
        
    def test_parameter_count(self):
        """Test if model has less than 25,000 parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertLess(total_params, 25000, 
                       f"Model has {total_params} parameters, which exceeds the limit of 25,000")
        print(f"Total parameters: {total_params}")
        
    def test_input_shape(self):
        """Test if model accepts 28x28 input"""
        batch_size = 32
        test_input = torch.randn(batch_size, 1, 28, 28)
        try:
            output = self.model(test_input)
        except Exception as e:
            self.fail(f"Model failed to process 28x28 input: {str(e)}")
            
    def test_output_shape(self):
        """Test if model outputs 10 classes"""
        batch_size = 32
        test_input = torch.randn(batch_size, 1, 28, 28)
        output = self.model(test_input)
        self.assertEqual(output.shape, (batch_size, 10),
                        f"Expected output shape {(batch_size, 10)}, but got {output.shape}")
        
    def test_model_training(self):
        """Test if model can be trained"""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Dummy data
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        
        # Try one training step
        try:
            optimizer.zero_grad()
            output = self.model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        except Exception as e:
            self.fail(f"Training step failed: {str(e)}")
            
    def test_model_inference(self):
        """Test if model can do inference"""
        self.model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 1, 28, 28)
            try:
                output = self.model(test_input)
                probabilities = torch.softmax(output, dim=1)
                self.assertTrue(torch.allclose(torch.sum(probabilities), torch.tensor(1.0), atol=1e-3), 
                                "Softmax probabilities should sum to approximately 1")
            except Exception as e:
                self.fail(f"Inference failed: {str(e)}")

    def test_model_gradient_flow(self):
        """Test if gradients are flowing through all layers"""
        self.model.train()
        
        # Forward pass
        x = torch.randn(1, 1, 28, 28, requires_grad=True)
        output = self.model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients in each layer
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.all(param.grad == 0), 
                           f"Zero gradients in {name}")

    def test_model_device_compatibility(self):
        """Test if model can be moved between CPU and CUDA"""
        if torch.cuda.is_available():
            try:
                model_cuda = self.model.cuda()
                self.assertTrue(next(model_cuda.parameters()).is_cuda,
                              "Model not properly moved to CUDA")
                
                model_cpu = model_cuda.cpu()
                self.assertFalse(next(model_cpu.parameters()).is_cuda,
                               "Model not properly moved back to CPU")
            except Exception as e:
                self.fail(f"Device movement failed: {str(e)}")
        else:
            print("CUDA not available, skipping device compatibility test")

    def test_model_batch_invariance(self):
        """Test if model produces consistent outputs for different batch sizes"""
        self.model.eval()
        with torch.no_grad():
            # Test with different batch sizes
            batch_sizes = [1, 16, 32]
            outputs = []
            
            for batch_size in batch_sizes:
                # Use same seed for reproducibility
                torch.manual_seed(42)
                input_tensor = torch.randn(batch_size, 1, 28, 28)
                output = self.model(input_tensor)
                
                # Store normalized output for first sample
                outputs.append(torch.softmax(output[0], dim=0))
            
            # Compare outputs
            for i in range(1, len(outputs)):
                self.assertTrue(
                    torch.allclose(outputs[0], outputs[i], atol=1e-6),
                    f"Model output varies with batch size {batch_sizes[i]}"
                )

if __name__ == '__main__':
    unittest.main() 