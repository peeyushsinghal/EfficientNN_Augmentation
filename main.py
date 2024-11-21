from train import train_model
from test import test_model
import unittest
from test_cases import TestMNISTModel
import sys
from io import StringIO

def run_tests():
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMNISTModel)
    
    # Run tests and capture output
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream)
    result = runner.run(suite)
    
    # Print results
    stream.seek(0)
    print(stream.read())
    
    if result.failures or result.errors:
        print("\nFailed Tests:")
        for failure in result.failures:
            print(f"- {failure[0]}: {failure[1]}")
        for error in result.errors:
            print(f"- {error[0]}: {error[1]}")
        return False
    return True

def main():
    try:
        # Train the model
        print("Training model...")
        model = train_model()
        
        # Run test cases
        print("\nRunning unit tests...")
        tests_passed = run_tests()
        if not tests_passed:
            print("\nSome tests failed. Please check the output above.")
            return
        
        # Test model accuracy
        print("\nTesting model accuracy...")
        accuracy = test_model()
        
        # Check if accuracy meets requirement
        if accuracy <= 80:
            print(f"\nFAILED: Model accuracy ({accuracy:.2f}%) is below the required 80%")
            return
            
        print("\nAll tests passed successfully!")
        print(f"Final accuracy: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main() 