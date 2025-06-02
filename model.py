#!/usr/bin/env python3
"""
ONNX Model Implementation for ImageNet Classification

This module contains classes for ONNX model loading, prediction, and image preprocessing.
Designed for production deployment with optimized inference performance.
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import logging
from typing import Tuple, Union, Optional
import os


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing class for ImageNet classification.
    Handles all preprocessing steps required by the model.
    """
    
    def __init__(self):
        """Initialize the preprocessor with ImageNet normalization parameters."""
        self.target_size = (224, 224)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
    def preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess an image for model inference.
        
        Args:
            image_input: Can be a file path (str), PIL Image, or numpy array
            
        Returns:
            np.ndarray: Preprocessed image tensor in NCHW format, ready for model input
            
        Raises:
            ValueError: If image format is not supported
            FileNotFoundError: If image file doesn't exist
        """
        try:
            # Load image based on input type
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input)
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.debug(f"Converted image from {image.mode} to RGB")
            
            # Resize to target size using bilinear interpolation
            image = image.resize(self.target_size, Image.BILINEAR)
            
            # Convert to numpy array
            image_np = np.array(image, dtype=np.float32)
            
            # Normalize to [0, 1] range
            image_np = image_np / 255.0
            
            # Apply ImageNet normalization (subtract mean, divide by std)
            image_np = (image_np - self.mean) / self.std
            
            # Convert from HWC to CHW format and add batch dimension (NCHW)
            image_tensor = image_np.transpose(2, 0, 1)[np.newaxis, ...]
            
            logger.debug(f"Preprocessed image shape: {image_tensor.shape}")
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def preprocess_batch(self, image_list: list) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            image_list: List of image inputs (paths, PIL Images, or arrays)
            
        Returns:
            np.ndarray: Batch of preprocessed images in NCHW format
        """
        preprocessed_images = []
        
        for i, image_input in enumerate(image_list):
            try:
                processed_image = self.preprocess_image(image_input)
                preprocessed_images.append(processed_image)
            except Exception as e:
                logger.error(f"Error preprocessing image {i}: {e}")
                raise
        
        # Stack all images into a single batch
        batch_tensor = np.concatenate(preprocessed_images, axis=0)
        logger.debug(f"Preprocessed batch shape: {batch_tensor.shape}")
        
        return batch_tensor


class ONNXModelLoader:
    """
    ONNX model loader and inference class.
    Handles model loading, optimization, and prediction.
    """
    
    def __init__(self, model_path: str, providers: Optional[list] = None):
        """
        Initialize the ONNX model loader.
        
        Args:
            model_path (str): Path to the ONNX model file
            providers (list, optional): List of execution providers. 
                                      Defaults to ['CPUExecutionProvider']
        """
        self.model_path = model_path
        self.providers = providers or ['CPUExecutionProvider']
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the ONNX model and extract metadata."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
            
            logger.info(f"Loading ONNX model from: {self.model_path}")
            
            # Create inference session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=self.providers
            )
            
            # Extract input/output metadata
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_shape = self.session.get_outputs()[0].shape
            
            logger.info(f"Model loaded successfully!")
            logger.info(f"Input: {self.input_name} - Shape: {self.input_shape}")
            logger.info(f"Output: {self.output_name} - Shape: {self.output_shape}")
            logger.info(f"Execution providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            raise
    
    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference on the input tensor.
        
        Args:
            input_tensor (np.ndarray): Preprocessed input tensor
            
        Returns:
            np.ndarray: Model predictions (raw logits)
        """
        try:
            if self.session is None:
                raise RuntimeError("Model not loaded. Call _load_model() first.")
            
            # Validate input shape
            expected_shape = tuple(self.input_shape[1:])  # Skip batch dimension
            actual_shape = input_tensor.shape[1:]
            
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Input shape mismatch. Expected: {expected_shape}, "
                    f"Got: {actual_shape}"
                )
            
            # Run inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor}
            )
            
            return outputs[0]
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
    
    def predict_with_softmax(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference and apply softmax to get probabilities.
        
        Args:
            input_tensor (np.ndarray): Preprocessed input tensor
            
        Returns:
            np.ndarray: Class probabilities (0-1 range, sum to 1)
        """
        logits = self.predict(input_tensor)
        
        # Apply softmax to convert logits to probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probabilities
    
    def get_top_predictions(
        self, 
        input_tensor: np.ndarray, 
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-k predictions with class indices and probabilities.
        
        Args:
            input_tensor (np.ndarray): Preprocessed input tensor
            top_k (int): Number of top predictions to return
            
        Returns:
            tuple: (top_k_indices, top_k_probabilities)
        """
        probabilities = self.predict_with_softmax(input_tensor)
        
        # Get top-k indices and probabilities
        top_k_indices = np.argsort(probabilities, axis=1)[:, -top_k:][:, ::-1]
        top_k_probabilities = np.take_along_axis(probabilities, top_k_indices, axis=1)
        
        return top_k_indices, top_k_probabilities


class ImageNetClassifier:
    """
    Complete ImageNet classifier combining preprocessing and ONNX inference.
    This is the main class for end-to-end image classification.
    """
    
    def __init__(self, model_path: str, providers: Optional[list] = None):
        """
        Initialize the complete classifier.
        
        Args:
            model_path (str): Path to the ONNX model file
            providers (list, optional): ONNX execution providers
        """
        self.preprocessor = ImagePreprocessor()
        self.model = ONNXModelLoader(model_path, providers)
        
        logger.info("ImageNet classifier initialized successfully")
    
    def classify_image(
        self, 
        image_input: Union[str, Image.Image, np.ndarray]
    ) -> Tuple[int, float, np.ndarray]:
        """
        Classify a single image and return the predicted class.
        
        Args:
            image_input: Image to classify (path, PIL Image, or array)
            
        Returns:
            tuple: (predicted_class_id, confidence, all_probabilities)
        """
        try:
            # Preprocess the image
            input_tensor = self.preprocessor.preprocess_image(image_input)
            
            # Get predictions
            probabilities = self.model.predict_with_softmax(input_tensor)
            
            # Get the predicted class and confidence
            predicted_class = np.argmax(probabilities, axis=1)[0]
            confidence = probabilities[0, predicted_class]
            
            logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
            
            return int(predicted_class), float(confidence), probabilities[0]
            
        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            raise
    
    def classify_batch(
        self, 
        image_list: list
    ) -> list:
        """
        Classify a batch of images.
        
        Args:
            image_list: List of images to classify
            
        Returns:
            list: List of tuples (predicted_class_id, confidence, probabilities)
        """
        try:
            # Preprocess the batch
            input_batch = self.preprocessor.preprocess_batch(image_list)
            
            # Get predictions for the batch
            probabilities_batch = self.model.predict_with_softmax(input_batch)
            
            # Process results for each image
            results = []
            for i in range(probabilities_batch.shape[0]):
                probabilities = probabilities_batch[i]
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]
                
                results.append((int(predicted_class), float(confidence), probabilities))
            
            logger.info(f"Classified batch of {len(image_list)} images")
            
            return results
            
        except Exception as e:
            logger.error(f"Error classifying batch: {e}")
            raise
    
    def get_top_k_predictions(
        self, 
        image_input: Union[str, Image.Image, np.ndarray], 
        k: int = 5
    ) -> list:
        """
        Get top-k predictions for an image.
        
        Args:
            image_input: Image to classify
            k: Number of top predictions to return
            
        Returns:
            list: List of tuples (class_id, probability)
        """
        try:
            # Preprocess the image
            input_tensor = self.preprocessor.preprocess_image(image_input)
            
            # Get top-k predictions
            top_k_indices, top_k_probabilities = self.model.get_top_predictions(input_tensor, k)
            
            # Format results
            results = [
                (int(idx), float(prob)) 
                for idx, prob in zip(top_k_indices[0], top_k_probabilities[0])
            ]
            
            logger.info(f"Top-{k} predictions retrieved")
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting top-k predictions: {e}")
            raise


# Utility function for quick inference
def quick_classify(model_path: str, image_path: str) -> int:
    """
    Quick classification function for single image.
    
    Args:
        model_path (str): Path to ONNX model
        image_path (str): Path to image file
        
    Returns:
        int: Predicted class ID
    """
    classifier = ImageNetClassifier(model_path)
    predicted_class, _, _ = classifier.classify_image(image_path)
    return predicted_class


if __name__ == "__main__":
    # Example usage
    model_path = "imagenet_classifier.onnx"
    
    if os.path.exists(model_path):
        print("Testing ONNX model...")
        
        # Initialize classifier
        classifier = ImageNetClassifier(model_path)
        
        # Test with dummy data if no real images available
        print("Creating dummy test image...")
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Classify the dummy image
        predicted_class, confidence, _ = classifier.classify_image(dummy_image)
        print(f"Dummy image classified as class {predicted_class} with confidence {confidence:.4f}")
        
        # Test top-k predictions
        top_k = classifier.get_top_k_predictions(dummy_image, k=3)
        print(f"Top-3 predictions: {top_k}")
        
    else:
        print(f"ONNX model not found: {model_path}")
        print("Please run convert_to_onnx.py first to create the ONNX model.") 