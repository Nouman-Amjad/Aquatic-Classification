"""
Cerebrium Server Testing Script

This script tests the deployed Cerebrium model endpoints.
It can test individual images or run comprehensive test suites.

Usage:
    python test_server.py --image_path <path_to_image>
    python test_server.py --run_tests
"""

import os
import json
import base64
import time
import argparse
import logging
from typing import Dict, Any, List
import requests
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default Cerebrium configuration (update these with your deployment details)
API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWQ0NGQzYmI3IiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyMDY0MzgzNjI3fQ.50hUhd13kjIdVk4YLtoTKrNARH5x_HDdRdGJNHT03JyGPyQBZ7LVEikOP7XXMvLrMMmWMZ2nihOgDLN2gCpxVcivfD7gUe0d39eyRkUDMMR8zIWkG3hDgIzqssw4fiLhb8Np72BXnzlM_jQsVAKZz1iRBhOGtv3TGQOg5E4Rfl6FotEgCqsa8SrJBHN72oh7gC2upXsTi_HF_gb37rQ3SI_NqCBp0b7kczKo1AVYtj8n1mq_DMSpOiDGkezMQ7gNkLBhi2Bjl102Zh47bZbH89tnHJRJntAFqfj5gtiMU2iLTiEsi8NMULIS57z1OOAqZfFlA2Kc72CLlATERTdidQ"
MODEL_URL = "https://api.cortex.cerebrium.ai/v4/p-d44d3bb7/imagenet-classifier/predict"

class CerebriumTester:
    """ 
    Test client for Cerebrium deployed model.
    """
    
    def __init__(self, api_key=None, model_url=None):
        """
        Initialize the tester with API credentials.
        
        Args:
            api_key: Cerebrium API key
            model_url: Model endpoint URL
        """
        self.api_key = api_key or API_KEY
        self.model_url = model_url or MODEL_URL
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        
        logger.info(f"Tester ready for {self.model_url}")
    
    def encode_image(self, path: str) -> str:
        """
        Encode an image file to base64 string.
        
        Args:
            path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def test_image(self, path: str, top_k=1, probs=False) -> Dict[str, Any]:
        """
        Test classification of a single image.
        
        Args:
            path: Path to image file
            top_k: Number of top predictions to return
            probs: Whether to include all class probabilities
            
        Returns:
            Dictionary containing prediction results
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        
        logger.info(f"Testing {path}")
        
        payload = {"item": {"image": self.encode_image(path), "top_k": top_k, "include_probabilities": probs}}
        
        start = time.time()
        resp = self.session.post(self.model_url, json=payload, timeout=30)
        duration = time.time() - start
        
        if resp.status_code == 200:
            result = resp.json()
            result.update({'inference_time': duration, 'image_path': path})
            
            logger.info(f" Success in {duration:.3f}s")
            if 'result' in result and 'predicted_class' in result['result']:
                cls = result['result']['predicted_class']
                conf = result['result'].get('confidence', 0)
                logger.info(f"  Class: {cls}, Confidence: {conf:.4f}")
            
            return result
        else:
            error = {'error': f"HTTP {resp.status_code}: {resp.text}", 'status': 'error', 'inference_time': duration, 'image_path': path}
            logger.error(f"✗ Failed: {error['error']}")
            return error
    
    def benchmark(self, path: str, n=10) -> Dict[str, Any]:
        """
        Run performance benchmark on the deployed model.
        
        Args:
            path: Path to test image
            n: Number of requests to make
            
        Returns:
            Performance benchmark results
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        
        logger.info(f"Running benchmark with {n} requests")
        
        payload = {"item": {"image": self.encode_image(path), "top_k": 1, "include_probabilities": False}}
        
        times = []
        success = 0
        
        for i in range(n):
            try:
                start = time.time()
                resp = self.session.post(self.model_url, json=payload, timeout=30)
                duration = time.time() - start
                times.append(duration)
                
                if resp.status_code == 200:
                    success += 1
                
                logger.info(f"  Request {i+1}/{n}: {duration:.3f}s")
            except Exception as e:
                logger.error(f"  Request {i+1}/{n} failed: {e}")
        
        if times:
            avg = sum(times) / len(times)
            p95 = sorted(times)[int(len(times) * 0.95)]
            
            results = {
                'total': n,
                'success': success,
                'success_rate': success / n * 100,
                'avg_time': avg,
                'p95_time': p95,
                'status': 'success'
            }
            
            logger.info(f" Benchmark complete: {results['success_rate']:.1f}% success, avg {avg:.3f}s")
            return results
        else:
            return {
                'error': 'No successful requests',
                'status': 'failed',
                'total': n
            }
    
    def run_tests(self, images=None) -> Dict[str, Any]:
        """
        Run comprehensive test suite.
        
        Args:
            images: List of test image paths
            
        Returns:
            Comprehensive test results
        """
        logger.info("=" * 40 + "\nRunning Tests\n" + "=" * 40)
        
        if images is None:
            images = ['test_images/n01440764_tench.jpg', 'test_images/n01667114_mud_turtle.jpg']
            images = [img for img in images if os.path.exists(img)]
        
        if not images:
            img = Image.new('RGB', (224, 224), 'red')
            temp_path = 'temp.jpg'
            img.save(temp_path)
            images = [temp_path]
        
        results = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'tests': {}}
        
        # Single image tests
        logger.info("\n1. Single Image Tests...")
        results['tests']['single'] = [self.test_image(img) for img in images]
        
        # Top-K test
        logger.info("\n2. Top-K Test...")
        results['tests']['topk'] = self.test_image(images[0], top_k=5)
        
        # Benchmark
        logger.info("\n3. Benchmark...")
        results['tests']['benchmark'] = self.benchmark(images[0], 5)
        
        # Cleanup
        if 'temp.jpg' in images:
            try:
                os.remove('temp.jpg')
            except:
                pass
        
        # Summary
        total = passed = 0
        for name, test in results['tests'].items():
            if isinstance(test, list):
                for t in test:
                    total += 1
                    if t.get('status') == 'success' or 'result' in t:
                        passed += 1
            else:
                total += 1
                if test.get('status') == 'success' or 'result' in test:
                    passed += 1
        
        results['summary'] = {'total': total, 'passed': passed, 'rate': passed / total * 100 if total else 0}
        logger.info(f"\n {passed}/{total} tests passed ({results['summary']['rate']:.1f}%)")
        return results


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Test Cerebrium model')
    parser.add_argument('--image_path', help='Image file path')
    parser.add_argument('--api_key', help='API key')
    parser.add_argument('--model_url', help='Model URL')
    parser.add_argument('--run_tests', action='store_true', help='Run test suite')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--top_k', type=int, default=1, help='Top-K predictions')
    parser.add_argument('--num_requests', type=int, default=10, help='Benchmark requests')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = CerebriumTester(api_key=args.api_key, model_url=args.model_url)
    
    if args.run_tests:
        # Run comprehensive tests
        results = tester.run_tests()
        
        # Save results to file
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("Test results saved to test_results.json")
        
    elif args.image_path:
        # Test single image
        if args.benchmark:
            # Run benchmark with the image
            result = tester.benchmark(args.image_path, args.num_requests)
        else:
            # Single prediction
            result = tester.test_image(args.image_path, top_k=args.top_k, probs=True)
        
        print(json.dumps(result, indent=2))
        
        # Print the class ID as required
        if 'result' in result and 'predicted_class' in result['result']:
            print(f"\nPredicted Class ID: {result['result']['predicted_class']}")
        
    else:
        parser.print_help()
        print("\nExample usage:")
        print("python test_server.py --image_path path/to/image.jpg")
        print("python test_server.py --run_tests")
        print("python test_server.py --api_key YOUR_KEY --model_url YOUR_URL --image_path path/to/image.jpg")


if __name__ == "__main__":
    main() 