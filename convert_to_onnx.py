"""
ONNX Conversion Script for ImageNet Classifier

This script converts the PyTorch model to ONNX format for deployment.
It handles the conversion process and validates the converted model.
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pytorch_model import ImageNetClassifier, load_model
import argparse
import os


def convert_pytorch_to_onnx(
    pytorch_model_path='pytorch_model_weights.pth',
    onnx_model_path='imagenet_classifier.onnx',
    input_shape=(1, 3, 224, 224)
):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        pytorch_model_path (str): Path to PyTorch model weights
        onnx_model_path (str): Output path for ONNX model
        input_shape (tuple): Input tensor shape (batch_size, channels, height, width)
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        print(f"Loading PyTorch model from {pytorch_model_path}...")
        
        # Load the PyTorch model
        model = load_model(pytorch_model_path)
        model.eval()
        
        # Create dummy input for tracing
        dummy_input = torch.randn(*input_shape)
        
        print(f"Converting to ONNX format...")
        
        # Export the model to ONNX format
        torch.onnx.export(
            model,                          # Model to export
            dummy_input,                    # Model input (or a tuple for multiple inputs)
            onnx_model_path,               # Where to save the model
            export_params=True,            # Store the trained parameter weights
            opset_version=11,              # ONNX version to export the model to
            do_constant_folding=True,      # Whether to execute constant folding for optimization
            input_names=['input'],         # Model input names
            output_names=['output'],       # Model output names
            dynamic_axes={
                'input': {0: 'batch_size'}, # Variable length axes
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"Model successfully converted to {onnx_model_path}")
        return True
        
    except Exception as e:
        print(f" Error during conversion: {e}")
        return False


def validate_onnx_model(onnx_model_path='imagenet_classifier.onnx'):
    """
    Validate the converted ONNX model.
    
    Args:
        onnx_model_path (str): Path to ONNX model
        
    Returns:
        bool: True if validation successful, False otherwise
    """
    try:
        print(f"Validating ONNX model: {onnx_model_path}")
        
        # Load and check the ONNX model
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        
        print("ONNX model validation passed")
        
        # Print model info
        print(f"Model inputs: {[input.name for input in onnx_model.graph.input]}")
        print(f"Model outputs: {[output.name for output in onnx_model.graph.output]}")
        
        return True
        
    except Exception as e:
        print(f"ONNX model validation failed: {e}")
        return False


def test_onnx_inference(onnx_model_path='imagenet_classifier.onnx'):
    """
    Test the ONNX model inference capability.
    
    Args:
        onnx_model_path (str): Path to ONNX model
        
    Returns:
        bool: True if inference test successful, False otherwise
    """
    try:
        print("Testing ONNX model inference...")
        
        # Create inference session
        ort_session = ort.InferenceSession(onnx_model_path)
        
        # Get input/output names
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Run inference
        outputs = ort_session.run([output_name], {input_name: dummy_input})
        
        # Check output shape
        output_shape = outputs[0].shape
        expected_shape = (1, 1000)  # Batch size 1, 1000 classes
        
        if output_shape == expected_shape:
            print(f" Inference test passed. Output shape: {output_shape}")
            print(f" Output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
            return True
        else:
            print(f" Unexpected output shape: {output_shape}, expected: {expected_shape}")
            return False
            
    except Exception as e:
        print(f" Inference test failed: {e}")
        return False


def compare_pytorch_onnx_outputs(
    pytorch_model_path='pytorch_model_weights.pth',
    onnx_model_path='imagenet_classifier.onnx',
    tolerance=1e-5
):
    """
    Compare outputs between PyTorch and ONNX models to ensure consistency.
    
    Args:
        pytorch_model_path (str): Path to PyTorch model weights
        onnx_model_path (str): Path to ONNX model
        tolerance (float): Tolerance for output comparison
        
    Returns:
        bool: True if outputs match within tolerance, False otherwise
    """
    try:
        print("Comparing PyTorch and ONNX model outputs...")
        
        # Load PyTorch model
        pytorch_model = load_model(pytorch_model_path)
        pytorch_model.eval()
        
        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_model_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        # Create test input
        test_input = torch.randn(1, 3, 224, 224)
        
        # Get PyTorch output
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()
        
        # Get ONNX output
        onnx_output = ort_session.run([output_name], {input_name: test_input.numpy()})[0]
        
        # Compare outputs
        diff = np.abs(pytorch_output - onnx_output)
        max_diff = np.max(diff)
        
        if max_diff < tolerance:
            print(f" Model outputs match within tolerance. Max difference: {max_diff:.2e}")
            return True
        else:
            print(f" Model outputs differ. Max difference: {max_diff:.2e} (tolerance: {tolerance})")
            return False
            
    except Exception as e:
        print(f" Output comparison failed: {e}")
        return False


def main():
    """Main conversion pipeline."""
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--pytorch_model', default='pytorch_model_weights.pth',
                       help='Path to PyTorch model weights')
    parser.add_argument('--onnx_model', default='imagenet_classifier.onnx',
                       help='Output path for ONNX model')
    parser.add_argument('--skip_validation', action='store_true',
                       help='Skip model validation')
    parser.add_argument('--skip_comparison', action='store_true',
                       help='Skip output comparison')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PyTorch to ONNX Conversion Pipeline")
    print("=" * 60)
    
    # Check if PyTorch model exists
    if not os.path.exists(args.pytorch_model):
        print(f" PyTorch model not found: {args.pytorch_model}")
        return False
    
    success = True
    
    # 1. Convert PyTorch to ONNX
    if not convert_pytorch_to_onnx(args.pytorch_model, args.onnx_model):
        return False
    
    # 2. Validate ONNX model
    if not args.skip_validation:
        if not validate_onnx_model(args.onnx_model):
            success = False
    
    # 3. Test ONNX inference
    if not test_onnx_inference(args.onnx_model):
        success = False
    
    # 4. Compare PyTorch and ONNX outputs
    if not args.skip_comparison:
        if not compare_pytorch_onnx_outputs(args.pytorch_model, args.onnx_model):
            success = False
    
    print("=" * 60)
    if success:
        print(" All conversion steps completed successfully!")
        print(f" ONNX model saved to: {args.onnx_model}")
    else:
        print(" Some conversion steps failed. Please check the errors above.")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    main() 