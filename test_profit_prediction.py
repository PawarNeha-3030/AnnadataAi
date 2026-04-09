"""
Test script for profit prediction functionality
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_profit_prediction_page():
    """Test if profit prediction page loads"""
    print("🌱 Testing Profit Prediction Page")
    print("="*50)
    
    try:
        response = requests.get(f"{BASE_URL}/profit-prediction")
        if response.status_code == 200:
            print("✅ Profit prediction page loads successfully")
            print(f"   📄 Page size: {len(response.content)} bytes")
            
            # Check if page contains expected elements
            content = response.text
            if "Crop Profit Prediction" in content:
                print("✅ Page title found")
            if "Analyze Profit Potential" in content:
                print("✅ Analysis button found")
            if "nav_profit_prediction" in content:
                print("✅ Navigation translation key found")
                
        else:
            print(f"❌ Page failed to load: HTTP {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the Flask server is running")
        print("   Start it with: python app.py")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_navigation_links():
    """Test navigation links on other pages"""
    print("\n🔗 Testing Navigation Links")
    print("="*50)
    
    pages = [
        ('/', 'Home'),
        ('/location', 'Location'),
        ('/crops', 'Crops'),
        ('/water', 'Water'),
        ('/diseases', 'Diseases')
    ]
    
    for url, name in pages:
        try:
            response = requests.get(f"{BASE_URL}{url}")
            if response.status_code == 200:
                content = response.text
                if '/profit-prediction' in content:
                    print(f"✅ {name} page has correct profit prediction link")
                else:
                    print(f"❌ {name} page missing profit prediction link")
            else:
                print(f"❌ {name} page failed to load: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error testing {name} page: {e}")

def test_api_endpoint():
    """Test the profit analysis API endpoint"""
    print("\n🔬 Testing Profit Analysis API")
    print("="*50)
    
    test_data = {
        "crop_name": "tomato",
        "planting_month": 3,
        "farm_area": 2.5,
        "location": "Pune"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/analyze-crop-profit",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ API endpoint working")
                analysis = data.get('analysis', {})
                print(f"   📊 Prediction: {analysis.get('profit_prediction', 'N/A')}")
                print(f"   ⏱️ Growth Duration: {analysis.get('growth_duration', 'N/A')}")
                print(f"   🎯 Confidence: {analysis.get('confidence_score', 'N/A')}%")
            else:
                print(f"❌ API returned error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ API failed: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ API test error: {e}")

def main():
    """Run all tests"""
    print("📊 PROFIT PREDICTION TEST SUITE")
    print("="*60)
    print("Make sure the Flask server is running: python app.py")
    print("="*60)
    
    # Test server connectivity
    try:
        response = requests.get(BASE_URL, timeout=5)
        print("✅ Server is running!\n")
    except:
        print("❌ Server is not running! Please start it first.\n")
        return
    
    # Run tests
    test_profit_prediction_page()
    test_navigation_links()
    test_api_endpoint()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("🌐 Visit: http://localhost:5000/profit-prediction")
    print("="*60)

if __name__ == "__main__":
    main()