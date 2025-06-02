"""
Cerebrium ImageNet Classifier Application

This module provides the main application interface for the deployed ImageNet classifier.
It includes functions for initialization, prediction, and health checking.
"""

import os
import io
import base64
import logging
import numpy as np
from PIL import Image
from model import ImageNetClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
classifier = None

def init():
    """
    Initialize the model. This function is called once when the container starts.
    """
    global classifier
    
    model_path = "imagenet_classifier.onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info("Initializing classifier...")
    classifier = ImageNetClassifier(model_path)
    logger.info("Ready")


def predict(item: dict) -> dict:
    """
    Main prediction function called by Cerebrium.
    
    Args:
        item: Dictionary containing the input data
        
    Returns:
        Dictionary containing the prediction results
    """
    if "image" not in item:
        return {"error": "Missing 'image' field", "status": "error"}
    
    try:
        image = _decode_image(item["image"])
        top_k = item.get("top_k", 1)
        
        if top_k == 1:
            cls, conf, probs = classifier.classify_image(image)
            result = {"predicted_class": cls, "confidence": float(conf), "status": "success"}
            if item.get("include_probabilities", False):
                result["all_probabilities"] = probs.tolist()
        else:
            results = classifier.get_top_k_predictions(image, top_k)
            result = {
                "top_predictions": [{"class_id": int(c), "probability": float(p)} for c, p in results],
                "predicted_class": results[0][0],
                "confidence": results[0][1],
                "status": "success"
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": str(e), "status": "error"}


def health_check() -> dict:
    """
    Health check endpoint for monitoring.
    
    Returns:
        Dictionary containing health status
    """
    if not classifier:
        return {"status": "unhealthy", "reason": "Not initialized"}
    
    try:
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        cls, conf, _ = classifier.classify_image(img)
        return {"status": "healthy", "model_loaded": True, "test_class": int(cls), "test_conf": float(conf)}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "reason": str(e)}


def _decode_image(data: str) -> Image.Image:
    """Decode base64 image data to PIL Image."""
    if data.startswith('data:image'):
        data = data.split(',')[1]
    
    return Image.open(io.BytesIO(base64.b64decode(data)))


# For local testing
if __name__ == "__main__":
    import json
    
    print("Starting ImageNet Classifier Application...")
    
    # Initialize the model
    init()
    
    # Test health check
    health = health_check()
    print(f"Health check: {json.dumps(health, indent=2)}")
    
    # Test with a dummy image
    print("\nTesting with dummy image...")
    
    # Create a dummy image and encode it as base64
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Test single prediction
    test_input = {
        "image": image_base64,
        "top_k": 1,
        "include_probabilities": False
    }
    
    result = predict(test_input)
    print(f"Single prediction result: {json.dumps(result, indent=2)}")
    
    # Test top-k predictions
    test_input_topk = {
        "image": image_base64,
        "top_k": 5,
        "include_probabilities": False
    }
    
    result_topk = predict(test_input_topk)
    print(f"Top-5 prediction result: {json.dumps(result_topk, indent=2)}")
    
    print("\nApplication test completed successfully!") 