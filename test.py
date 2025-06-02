"""
Comprehensive Test for ImageNet Classifier

This script tests all components of the classification system including:
- ONNX model loading and inference
- Image preprocessing
- Error handling
- Performance benchmarks
"""

import pytest
import numpy as np
import os
import time
import tempfile
from PIL import Image
import logging

# Import our modules
from model import ImagePreprocessor, ONNXModelLoader, ImageNetClassifier, quick_classify
from pytorch_model import load_model as load_pytorch_model, predict_image

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestImagePreprocessor:
    """Test cases for ImagePreprocessor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.preprocessor = ImagePreprocessor()
        self.test_image_path = None
        
        # Create a temporary test image
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            Image.fromarray(test_image).save(f.name)
            self.test_image_path = f.name
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_image_path and os.path.exists(self.test_image_path):
            os.unlink(self.test_image_path)
    
    def test_preprocess_image_from_path(self):
        """Test preprocessing image from file path."""
        result = self.preprocessor.preprocess_image(self.test_image_path)
        
        assert result.shape == (1, 3, 224, 224), f"Expected shape (1, 3, 224, 224), got {result.shape}"
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
        
        # Check if normalization is applied (values should be around ImageNet range)
        assert result.min() >= -3.0 and result.max() <= 3.0, "Values not in expected normalized range"
    
    def test_preprocess_pil_image(self):
        """Test preprocessing PIL Image object."""
        pil_image = Image.open(self.test_image_path)
        result = self.preprocessor.preprocess_image(pil_image)
        
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32
    
    def test_preprocess_numpy_array(self):
        """Test preprocessing numpy array."""
        numpy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = self.preprocessor.preprocess_image(numpy_image)
        
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32
    
    def test_preprocess_grayscale_conversion(self):
        """Test conversion of grayscale image to RGB."""
        # Create grayscale image
        gray_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        pil_gray = Image.fromarray(gray_image, mode='L')
        
        result = self.preprocessor.preprocess_image(pil_gray)
        
        assert result.shape == (1, 3, 224, 224), "Grayscale image not converted to RGB properly"
    
    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        # Create multiple test images
        test_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        result = self.preprocessor.preprocess_batch(test_images)
        
        assert result.shape == (3, 3, 224, 224), f"Expected batch shape (3, 3, 224, 224), got {result.shape}"
        assert result.dtype == np.float32
    
    def test_invalid_input_handling(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            self.preprocessor.preprocess_image(123)  # Invalid type (integer)
        
        with pytest.raises(FileNotFoundError):
            self.preprocessor.preprocess_image("nonexistent_file.jpg")


class TestONNXModelLoader:
    """Test cases for ONNXModelLoader class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model_path = "imagenet_classifier.onnx"
        self.model_loader = None
    
    def test_model_loading(self):
        """Test ONNX model loading."""
        if not os.path.exists(self.model_path):
            pytest.skip(f"ONNX model not found: {self.model_path}")
        
        self.model_loader = ONNXModelLoader(self.model_path)
        
        assert self.model_loader.session is not None, "Model session not initialized"
        assert self.model_loader.input_name is not None, "Input name not extracted"
        assert self.model_loader.output_name is not None, "Output name not extracted"
        assert self.model_loader.input_shape is not None, "Input shape not extracted"
        assert self.model_loader.output_shape is not None, "Output shape not extracted"
    
    def test_model_prediction(self):
        """Test model prediction functionality."""
        if not os.path.exists(self.model_path):
            pytest.skip(f"ONNX model not found: {self.model_path}")
        
        self.model_loader = ONNXModelLoader(self.model_path)
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Test raw prediction
        raw_output = self.model_loader.predict(dummy_input)
        assert raw_output.shape == (1, 1000), f"Expected output shape (1, 1000), got {raw_output.shape}"
        
        # Test softmax prediction
        prob_output = self.model_loader.predict_with_softmax(dummy_input)
        assert prob_output.shape == (1, 1000), f"Expected probability shape (1, 1000), got {prob_output.shape}"
        assert np.allclose(prob_output.sum(axis=1), 1.0, atol=1e-6), "Probabilities don't sum to 1"
        assert np.all(prob_output >= 0) and np.all(prob_output <= 1), "Probabilities not in [0, 1] range"
    
    def test_top_k_predictions(self):
        """Test top-k prediction functionality."""
        if not os.path.exists(self.model_path):
            pytest.skip(f"ONNX model not found: {self.model_path}")
        
        self.model_loader = ONNXModelLoader(self.model_path)
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        k = 5
        top_indices, top_probs = self.model_loader.get_top_predictions(dummy_input, k)
        
        assert top_indices.shape == (1, k), f"Expected top indices shape (1, {k}), got {top_indices.shape}"
        assert top_probs.shape == (1, k), f"Expected top probabilities shape (1, {k}), got {top_probs.shape}"
        
        # Check if probabilities are sorted in descending order
        for i in range(k - 1):
            assert top_probs[0, i] >= top_probs[0, i + 1], "Probabilities not sorted in descending order"
    
    def test_invalid_model_path(self):
        """Test error handling for invalid model path."""
        with pytest.raises(FileNotFoundError):
            ONNXModelLoader("nonexistent_model.onnx")
    
    def test_invalid_input_shape(self):
        """Test error handling for invalid input shapes."""
        if not os.path.exists(self.model_path):
            pytest.skip(f"ONNX model not found: {self.model_path}")
        
        self.model_loader = ONNXModelLoader(self.model_path)
        
        # Test with wrong input shape
        wrong_input = np.random.randn(1, 3, 100, 100).astype(np.float32)
        
        with pytest.raises(ValueError):
            self.model_loader.predict(wrong_input)


class TestImageNetClassifier:
    """Test cases for complete ImageNetClassifier class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model_path = "imagenet_classifier.onnx"
        self.classifier = None
        self.test_image_path = None
        
        # Create a temporary test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            Image.fromarray(test_image).save(f.name)
            self.test_image_path = f.name
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_image_path and os.path.exists(self.test_image_path):
            os.unlink(self.test_image_path)
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        if not os.path.exists(self.model_path):
            pytest.skip(f"ONNX model not found: {self.model_path}")
        
        self.classifier = ImageNetClassifier(self.model_path)
        
        assert self.classifier.preprocessor is not None, "Preprocessor not initialized"
        assert self.classifier.model is not None, "Model not initialized"
    
    def test_single_image_classification(self):
        """Test single image classification."""
        if not os.path.exists(self.model_path):
            pytest.skip(f"ONNX model not found: {self.model_path}")
        
        self.classifier = ImageNetClassifier(self.model_path)
        
        predicted_class, confidence, probabilities = self.classifier.classify_image(self.test_image_path)
        
        assert isinstance(predicted_class, int), f"Expected int, got {type(predicted_class)}"
        assert 0 <= predicted_class < 1000, f"Class ID out of range: {predicted_class}"
        assert isinstance(confidence, float), f"Expected float, got {type(confidence)}"
        assert 0 <= confidence <= 1, f"Confidence out of range: {confidence}"
        assert probabilities.shape == (1000,), f"Expected probabilities shape (1000,), got {probabilities.shape}"
    
    def test_batch_classification(self):
        """Test batch image classification."""
        if not os.path.exists(self.model_path):
            pytest.skip(f"ONNX model not found: {self.model_path}")
        
        self.classifier = ImageNetClassifier(self.model_path)
        
        # Create multiple test images
        test_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        results = self.classifier.classify_batch(test_images)
        
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        
        for predicted_class, confidence, probabilities in results:
            assert isinstance(predicted_class, int)
            assert 0 <= predicted_class < 1000
            assert isinstance(confidence, float)
            assert 0 <= confidence <= 1
            assert probabilities.shape == (1000,)
    
    def test_top_k_predictions(self):
        """Test top-k predictions."""
        if not os.path.exists(self.model_path):
            pytest.skip(f"ONNX model not found: {self.model_path}")
        
        self.classifier = ImageNetClassifier(self.model_path)
        
        k = 3
        results = self.classifier.get_top_k_predictions(self.test_image_path, k)
        
        assert len(results) == k, f"Expected {k} results, got {len(results)}"
        
        for i, (class_id, probability) in enumerate(results):
            assert isinstance(class_id, int)
            assert 0 <= class_id < 1000
            assert isinstance(probability, float)
            assert 0 <= probability <= 1
            
            # Check if probabilities are sorted in descending order
            if i > 0:
                prev_prob = results[i - 1][1]
                assert prev_prob >= probability, "Top-k results not sorted by probability"


class TestPerformance:
    """Performance benchmarks and stress tests."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model_path = "imagenet_classifier.onnx"
        self.classifier = None
        
        if os.path.exists(self.model_path):
            self.classifier = ImageNetClassifier(self.model_path)
    
    def test_inference_speed(self):
        """Benchmark inference speed."""
        if self.classifier is None:
            pytest.skip(f"ONNX model not found: {self.model_path}")
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Warm up the model
        for _ in range(3):
            self.classifier.classify_image(test_image)
        
        # Benchmark inference time
        num_iterations = 10
        start_time = time.time()
        
        for _ in range(num_iterations):
            self.classifier.classify_image(test_image)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        logger.info(f"Average inference time: {avg_time:.4f} seconds")
        
        # Assert inference time is reasonable (should be under 3 seconds as per requirements)
        assert avg_time < 3.0, f"Inference too slow: {avg_time:.4f}s (required: <3s)"
    
    def test_batch_processing_speed(self):
        """Benchmark batch processing speed."""
        if self.classifier is None:
            pytest.skip(f"ONNX model not found: {self.model_path}")
        
        # Create batch of test images
        batch_size = 5
        test_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]
        
        # Benchmark batch processing time
        start_time = time.time()
        results = self.classifier.classify_batch(test_images)
        end_time = time.time()
        
        batch_time = end_time - start_time
        avg_time_per_image = batch_time / batch_size
        
        logger.info(f"Batch processing time: {batch_time:.4f}s for {batch_size} images")
        logger.info(f"Average time per image in batch: {avg_time_per_image:.4f}s")
        
        assert len(results) == batch_size, "Batch processing returned wrong number of results"


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_quick_classify(self):
        """Test quick classification function."""
        model_path = "imagenet_classifier.onnx"
        
        if not os.path.exists(model_path):
            pytest.skip(f"ONNX model not found: {model_path}")
        
        # Create temporary test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            Image.fromarray(test_image).save(f.name)
            test_image_path = f.name
        
        try:
            predicted_class = quick_classify(model_path, test_image_path)
            
            assert isinstance(predicted_class, int)
            assert 0 <= predicted_class < 1000
            
        finally:
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)


class TestIntegration:
    """Integration tests comparing PyTorch and ONNX outputs."""
    
    def test_pytorch_onnx_consistency(self):
        """Test that PyTorch and ONNX models produce similar outputs."""
        pytorch_weights_path = "pytorch_model_weights.pth"
        onnx_model_path = "imagenet_classifier.onnx"
        
        if not os.path.exists(pytorch_weights_path):
            pytest.skip(f"PyTorch weights not found: {pytorch_weights_path}")
        
        if not os.path.exists(onnx_model_path):
            pytest.skip(f"ONNX model not found: {onnx_model_path}")
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            Image.fromarray(test_image).save(f.name)
            test_image_path = f.name
        
        try:
            # Get PyTorch prediction
            pytorch_model = load_pytorch_model(pytorch_weights_path)
            pytorch_class, pytorch_probs = predict_image(pytorch_model, test_image_path)
            
            # Get ONNX prediction
            onnx_classifier = ImageNetClassifier(onnx_model_path)
            onnx_class, onnx_confidence, onnx_probs = onnx_classifier.classify_image(test_image_path)
            
            # Compare predictions
            logger.info(f"PyTorch predicted class: {pytorch_class}")
            logger.info(f"ONNX predicted class: {onnx_class}")
            
            # Classes should match or be close (within top-5)
            pytorch_top5 = np.argsort(pytorch_probs)[-5:]
            assert onnx_class in pytorch_top5, f"ONNX prediction {onnx_class} not in PyTorch top-5: {pytorch_top5}"
            
        finally:
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)


def run_all_tests():
    """
    Run all tests with detailed output.
    """
    print("=" * 60)
    print("Running Comprehensive Test Suite for ImageNet Classifier")
    print("=" * 60)
    
    test_classes = [
        TestImagePreprocessor,
        TestONNXModelLoader,
        TestImageNetClassifier,
        TestPerformance,
        TestUtilityFunctions,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning tests for {test_class.__name__}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            total_tests += 1
            test_instance = test_class()
            
            try:
                # Setup
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test
                test_method = getattr(test_instance, test_method_name)
                test_method()
                
                print(f"   {test_method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f" {test_method_name}: {e}")
                failed_tests += 1
                
            finally:
                # Teardown
                if hasattr(test_instance, 'teardown_method'):
                    try:
                        test_instance.teardown_method()
                    except:
                        pass
    
    print("=" * 60)
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    if failed_tests > 0:
        print(f"WARNING: {failed_tests} tests failed")
    else:
        print("All tests passed! ")
    print("=" * 60)
    
    return failed_tests == 0


if __name__ == "__main__":
    # Check if pytest is available
    try:
        import pytest
        print("Running tests with pytest...")
        pytest.main([__file__, "-v", "--tb=short"])
    except ImportError:
        print("pytest not available, running manual test suite...")
        run_all_tests() 