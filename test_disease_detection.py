# test_disease_detection.py - Test the disease detection system
import requests
import json
from pathlib import Path

def test_disease_detection():
    """Test the disease detection API"""
    
    print("🧪 Testing Disease Detection System")
    print("="*50)
    
    # Test API endpoint
    url = "http://localhost:5000/api/detect-disease"
    
    # Check if there are any test images
    upload_dir = Path("static/uploads")
    if upload_dir.exists():
        test_images = list(upload_dir.glob("*.jpg"))
        
        if test_images:
            print(f"📸 Found {len(test_images)} test images")
            
            # Test with first available image
            test_image = test_images[0]
            print(f"🔍 Testing with: {test_image.name}")
            
            try:
                with open(test_image, 'rb') as f:
                    files = {'image': f}
                    response = requests.post(url, files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print("\n✅ Detection Results:")
                    print(f"   Disease: {result.get('disease_name', 'Unknown')}")
                    print(f"   Confidence: {result.get('confidence', 0)}%")
                    print(f"   ML Status: {result.get('ml_status', 'Unknown')}")
                    print(f"   Crop Type: {result.get('crop_type', 'Unknown')}")
                    print(f"   Severity: {result.get('severity', 'Unknown')}")
                    
                    if result.get('treatment_steps'):
                        print(f"   Treatment: {len(result['treatment_steps'])} steps available")
                    
                    return True
                    
                else:
                    print(f"❌ API Error: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
                    
            except requests.exceptions.ConnectionError:
                print("❌ Cannot connect to server. Make sure app.py is running.")
                return False
            except Exception as e:
                print(f"❌ Test error: {e}")
                return False
        else:
            print("📷 No test images found in static/uploads/")
            print("   Upload an image through the web interface first")
            return False
    else:
        print("📁 Upload directory not found")
        return False

def test_api_endpoints():
    """Test other API endpoints"""
    
    print("\n🔗 Testing API Endpoints:")
    print("-" * 30)
    
    endpoints = [
        ("GET", "/api/disease-stats", "Disease Statistics"),
        ("GET", "/api/all-diseases", "Disease Database")
    ]
    
    for method, endpoint, description in endpoints:
        try:
            url = f"http://localhost:5000{endpoint}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {description}: OK")
                if 'count' in data:
                    print(f"   Diseases in DB: {data['count']}")
            else:
                print(f"❌ {description}: Error {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {description}: Server not running")
        except Exception as e:
            print(f"❌ {description}: {e}")

if __name__ == "__main__":
    print("🌿 FarmAI Disease Detection Test Suite")
    print("="*50)
    
    # Test main detection
    detection_ok = test_disease_detection()
    
    # Test other endpoints
    test_api_endpoints()
    
    print("\n" + "="*50)
    if detection_ok:
        print("✅ Disease detection system is working!")
        print("💡 Try uploading different plant images to test accuracy")
    else:
        print("⚠️  Issues detected. Check server logs.")
    
    print("\n🌐 Open http://localhost:5000/diseases to test the web interface")