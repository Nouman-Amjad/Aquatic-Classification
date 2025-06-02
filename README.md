# ImageNet Classification with Cerebrium Deployment

This project deploys a ResNet-18 ImageNet classifier on Cerebrium's serverless GPU platform using Docker containers. The model achieves production-ready inference times under 3 seconds as required.

## Project Overview

- **Model**: ResNet-18 trained on ImageNet (1000 classes)
- **Format**: ONNX for optimized inference
- **Deployment**: Cerebrium serverless GPU platform
- **Container**: Custom Docker image
- **Performance**: < 3 seconds inference time

## Project Structure

```
Aquatic-Classification/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker container definition
│
├── pytorch_model.py            # Original PyTorch ResNet-18 implementation
├── convert_to_onnx.py          # PyTorch to ONNX conversion script
├── model.py                    # ONNX model classes (ImagePreprocessor, ONNXModelLoader, ImageNetClassifier)
├── app.py                      # Main Cerebrium application
│
├── test.py                     # Local comprehensive test suite
├── test_server.py              # Cerebrium deployment testing script
│
├── pytorch_model_weights.pth   # Pre-trained model weights
├── imagenet_classifier.onnx    # Converted ONNX model
│
└── test_images/                # Sample test images
    ├── n01440764_tench.jpg     # Tench (class 0)
    └── n01667114_mud_turtle.jpg # Mud turtle (class 35)
```

## Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to the project directory
cd Aquatic-Classification

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Local Components

```bash
# Test PyTorch model
python pytorch_model.py

# Convert to ONNX
python convert_to_onnx.py

# Test ONNX model
python model.py

# Run comprehensive tests
python test.py
```

### 3. Test Local Cerebrium App

```bash
# Test the app locally before deployment
python app.py
```

## Deliverables

### 1. convert_to_onnx.py
Converts the PyTorch ResNet-18 model to ONNX format with validation.

**Usage:**
```bash
python convert_to_onnx.py
python convert_to_onnx.py --pytorch_model custom_weights.pth --onnx_model custom_output.onnx
```

**Features:**
- PyTorch to ONNX conversion
- Model validation and verification
- Output comparison between PyTorch and ONNX
- Comprehensive error handling

### 2. model.py
Contains optimized ONNX model classes for production inference.

**Classes:**
- `ImagePreprocessor`: Handles image preprocessing (resize, normalize, format conversion)
- `ONNXModelLoader`: Manages ONNX model loading and inference
- `ImageNetClassifier`: Complete end-to-end classification pipeline

**Usage:**
```python
from model import ImageNetClassifier

# Initialize classifier
classifier = ImageNetClassifier("imagenet_classifier.onnx")

# Classify single image
class_id, confidence, probabilities = classifier.classify_image("image.jpg")

# Get top-k predictions
top_5 = classifier.get_top_k_predictions("image.jpg", k=5)

# Batch processing
results = classifier.classify_batch(["img1.jpg", "img2.jpg"])
```

### 3. test.py
Comprehensive local testing suite with performance benchmarks.

**Usage:**
```bash
# Run all tests
python test.py

# Run with pytest (if available)
pytest test.py -v
```

**Test Coverage:**
- Image preprocessing (RGB conversion, resizing, normalization)
- ONNX model loading and inference
- Classification accuracy and consistency
- Performance benchmarks (< 3 seconds requirement)
- Error handling and edge cases
- PyTorch vs ONNX output consistency

### 4. Cerebrium Deployment Files

#### Dockerfile
Custom Docker image for Cerebrium deployment following their requirements.

#### app.py
Main Cerebrium application with REST API endpoints.

**Endpoints:**
- `predict()`: Main prediction function
- `init()`: Model initialization
- `health_check()`: Health monitoring

### 5. test_server.py
Production deployment testing script for Cerebrium.

**Usage:**
```bash
# Test single image (replace with your API details)
python test_server.py --api_key YOUR_KEY --model_url YOUR_URL --image_path test_images/tench.jpg

# Run comprehensive test suite
python test_server.py --run_tests

# Performance benchmark
python test_server.py --image_path test_images/tench.jpg --benchmark --num_requests 20
```

**Features:**
- Single image classification testing
- Top-k predictions testing
- Performance benchmarking
- Error handling validation
- Health endpoint monitoring
- Comprehensive test reporting

## Deployment Instructions

### Step 1: Prepare Files
Ensure these files are in your project directory:
- `Dockerfile`
- `app.py`
- `model.py`
- `requirements.txt`
- `imagenet_classifier.onnx`

### Step 2: Deploy to Cerebrium

1. **Sign up** for Cerebrium account at [https://www.cerebrium.ai/](https://www.cerebrium.ai/)
2. **Get API Key** from your dashboard
3. **Create new deployment** using custom Docker image
4. **Upload files** or connect to GitHub repository
5. **Configure settings**:
   - Runtime: Custom Docker
   - GPU: Any available (model works on CPU/GPU)
   - Memory: 4GB minimum
   - Timeout: 30 seconds

### Step 3: Test Deployment

1. **Update test_server.py** with your API key and model URL:
```python
API_KEY = "your-cerebrium-api-key"
MODEL_URL = "https://api.cortex.cerebrium.ai/v2/p-d44d3bb7/imagenet-classifier/predict"
```

2. **Run tests**:
```bash
python test_server.py --run_tests
```

## Testing & Validation

### Local Testing
```bash
# Quick validation
python pytorch_model.py  # Test PyTorch model
python convert_to_onnx.py # Convert and validate ONNX
python test.py           # Run comprehensive tests

# Expected output: All tests pass with < 3s inference time
```

### Deployment Testing
```bash
# Test deployed model
python test_server.py --image_path test_images/n01440764_tench.jpg

# Expected output: 
# {
#   "predicted_class": 0,
#   "confidence": 0.85,
#   "status": "success",
#   "inference_time": 1.2
# }
# 
# Predicted Class ID: 0
```

### Performance Requirements 
-  Inference time < 3 seconds
-  ONNX format deployment
-  Custom Docker container
-  Production-ready error handling
-  Comprehensive test coverage

## Model Performance

### Test Results (Local)
- **Model**: ResNet-18 (1000 ImageNet classes)
- **Input**: 224x224 RGB images
- **Preprocessing**: ImageNet normalization
- **Inference time**: ~0.2-0.5 seconds (local CPU)
- **Accuracy**: Matches PyTorch implementation (within 1e-5 tolerance)

### Expected Cerebrium Performance
- **Cold start**: ~2-5 seconds (first request)
- **Warm inference**: ~0.5-1.5 seconds
- **GPU acceleration**: Available on GPU instances
- **Batch processing**: Supported for multiple images

## API Documentation

### Request Format
```json
{
  "image": "base64_encoded_image_string",
  "top_k": 5,
  "include_probabilities": false
}
```

### Response Format
```json
{
  "predicted_class": 107,
  "confidence": 0.8532,
  "top_predictions": [
    {"class_id": 107, "probability": 0.8532},
    {"class_id": 562, "probability": 0.0821},
    {"class_id": 971, "probability": 0.0543}
  ],
  "status": "success"
}
```

### Error Response
```json
{
  "error": "Error description",
  "status": "error"
}
```

## Development Notes

### Architecture Decisions
1. **ONNX Format**: Chosen for cross-platform compatibility and optimized inference
2. **Separate Classes**: Modular design allows easy testing and maintenance
3. **Docker Container**: Ensures consistent deployment environment
4. **Comprehensive Testing**: Covers all edge cases and performance requirements

### Performance Optimizations
1. **Model Quantization**: Can be added for further speed improvements
2. **Batch Processing**: Implemented for handling multiple images
3. **Caching**: Model loaded once during container initialization
4. **Error Handling**: Graceful degradation for invalid inputs

### Future Improvements
1. **Model Ensemble**: Combine multiple models for better accuracy
2. **Dynamic Batching**: Automatic batching for concurrent requests
3. **Metrics Collection**: Add detailed performance monitoring
4. **A/B Testing**: Support multiple model versions

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Ensure all dependencies installed
pip install -r requirements.txt
```

**2. ONNX Conversion Fails**
```bash
# Solution: Check PyTorch model compatibility
python pytorch_model.py  # Verify model loads correctly
```

**3. Slow Inference**
```bash
# Check: 
# - Image size (should be reasonable)
# - Model on correct device (CPU/GPU)
# - No memory leaks in preprocessing
```

**4. Deployment Issues**
- Verify all files are included in Docker context
- Check Cerebrium logs for detailed error messages
- Ensure model file is accessible in container

### Getting Help
1. Check test outputs for detailed error messages
2. Review Cerebrium documentation for deployment issues
3. Validate model locally before deployment
4. Use health check endpoint for monitoring

## Success Criteria 
- PyTorch to ONNX conversion working
- Separate preprocessing and model classes
- Comprehensive test suite (test.py)
- Cerebrium-compatible Docker deployment
- Server testing script (test_server.py)
- < 3 second inference requirement
- Production-ready error handling
- Complete documentation
- Git repository with meaningful commits 