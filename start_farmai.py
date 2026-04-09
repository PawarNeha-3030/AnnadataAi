"""
FarmAI Assistant Startup Script
"""
import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask',
        'tensorflow',
        'pillow',
        'numpy',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_model_files():
    """Check if model files exist"""
    print("\n🤖 Checking AI model files...")
    
    model_files = [
        'models/plant_disease_model.h5',
        'models/disease_info.pkl'
    ]
    
    missing_files = []
    
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ Missing model files. Training model...")
        try:
            subprocess.run([sys.executable, 'train_model.py'], check=True)
            print("✅ Model training completed!")
        except subprocess.CalledProcessError:
            print("❌ Model training failed!")
            return False
    
    return True

def start_server():
    """Start the Flask server"""
    print("\n🚀 Starting FarmAI Assistant...")
    print("="*60)
    print("🌐 Web Interface: http://localhost:5000")
    print("🦠 Disease Detection: http://localhost:5000/diseases")
    print("📊 Profit Prediction: http://localhost:5000/profit-prediction")
    print("🤖 AI Chatbot: Available on diseases page")
    print("="*60)
    print("\n⚠️ Press Ctrl+C to stop the server")
    print("\n" + "="*60)
    
    try:
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n\n👋 FarmAI Assistant stopped. Thank you!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

def main():
    """Main startup function"""
    print("🌱 FarmAI Assistant - Startup Script")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first!")
        return
    
    # Check model files
    if not check_model_files():
        print("\n❌ Model files are missing and couldn't be created!")
        return
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()