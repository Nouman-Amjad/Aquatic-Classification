"""
Cerebrium Deployment Helper Script

This script helps deploy the ImageNet classifier to Cerebrium with proper configuration.
"""

import os
import subprocess
import sys

REQUIRED_FILES = ['app.py', 'model.py', 'requirements.txt', 'Dockerfile', 'imagenet_classifier.onnx', 'cerebrium.toml']

def check_files():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        print(f" Missing: {', '.join(missing)}")
        return False
    print(" All files present")
    return True

def check_model():
    size_mb = os.path.getsize('imagenet_classifier.onnx') / (1024 * 1024)
    print(f" Model: {size_mb:.1f} MB")
    return True

def deploy():
    print(" Deploying...")
    try:
        subprocess.run(['cerebrium', 'deploy', '--config-file', 'cerebrium.toml', '--disable-syntax-check'], check=True)
        print(" Deployed!")
        return True
    except subprocess.CalledProcessError:
        print(" Failed. Try: cerebrium login")
        return False

def show_info():
    try:
        result = subprocess.run(['cerebrium', 'app', 'list'], capture_output=True, text=True, check=True)
        print(f" Apps:\n{result.stdout}")
    except:
        print("Check: https://dashboard.cerebrium.ai/")

def main():
    print(" Cerebrium Deployment\n" + "="*30)
    
    if not (check_files() and check_model()):
        sys.exit(1)
    
    if not deploy():
        sys.exit(1)
    
    show_info()
    print("\n Done! Test with: python test_server.py --run_tests")

if __name__ == "__main__":
    main() 