"""
Test script for chatbot API endpoints
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_pesticide_info():
    """Test pesticide information endpoint"""
    print("\n" + "="*60)
    print("Testing Pesticide Info API")
    print("="*60)
    
    crops = ['tomato', 'rice', 'cotton', 'wheat', 'unknown_crop']
    
    for crop in crops:
        print(f"\n🌱 Testing crop: {crop}")
        response = requests.post(
            f"{BASE_URL}/api/chatbot/pesticide-info",
            json={'crop': crop}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"   ✅ Success! Found {len(data['pesticides'])} pesticides")
                print(f"   📋 Diseases: {', '.join(data.get('diseases', ['N/A']))}")
                print(f"   💊 First pesticide: {data['pesticides'][0]['name']}")
            else:
                print(f"   ❌ Failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")

def test_find_shops():
    """Test shop finder endpoint"""
    print("\n" + "="*60)
    print("Testing Shop Finder API")
    print("="*60)
    
    locations = ['Pune', 'Mumbai', 'Delhi', 'Unknown City']
    
    for location in locations:
        print(f"\n📍 Testing location: {location}")
        response = requests.post(
            f"{BASE_URL}/api/chatbot/find-shops",
            json={'location': location, 'crop': 'tomato'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"   ✅ Success! Found {data['total_found']} shops")
                if data['shops']:
                    print(f"   🏪 First shop: {data['shops'][0]['name']}")
                    print(f"   📞 Phone: {data['shops'][0]['phone']}")
            else:
                print(f"   ❌ Failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")

def test_organic_alternatives():
    """Test organic alternatives endpoint"""
    print("\n" + "="*60)
    print("Testing Organic Alternatives API")
    print("="*60)
    
    crops = ['tomato', 'rice', 'general']
    
    for crop in crops:
        print(f"\n🌿 Testing crop: {crop}")
        response = requests.post(
            f"{BASE_URL}/api/chatbot/organic-alternatives",
            json={'crop': crop}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"   ✅ Success! Found {len(data['organic_solutions'])} solutions")
                if data['organic_solutions']:
                    print(f"   🌱 First solution: {data['organic_solutions'][0]['name']}")
            else:
                print(f"   ❌ Failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")

def test_emergency_contact():
    """Test emergency contact endpoint"""
    print("\n" + "="*60)
    print("Testing Emergency Contact API")
    print("="*60)
    
    print(f"\n🚨 Testing emergency contacts")
    response = requests.post(
        f"{BASE_URL}/api/chatbot/emergency-contact",
        json={'location': 'Pune'}
    )
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            print(f"   ✅ Success! Found {len(data['emergency_contacts'])} contacts")
            if data['emergency_contacts']:
                print(f"   👨‍⚕️ First contact: {data['emergency_contacts'][0]['name']}")
                print(f"   📞 Phone: {data['emergency_contacts'][0]['phone']}")
            if data.get('helpline'):
                print(f"   🆘 Helpline: {data['helpline']['name']} - {data['helpline']['phone']}")
        else:
            print(f"   ❌ Failed: {data.get('error', 'Unknown error')}")
    else:
        print(f"   ❌ HTTP Error: {response.status_code}")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🤖 CHATBOT API TEST SUITE")
    print("="*60)
    print("\nMake sure the Flask server is running on http://localhost:5000")
    print("Start it with: python app.py")
    
    try:
        # Test if server is running
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            print("✅ Server is running!")
        else:
            print("⚠️ Server responded but with unexpected status")
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running! Please start it first.")
        return
    
    # Run all tests
    test_pesticide_info()
    test_find_shops()
    test_organic_alternatives()
    test_emergency_contact()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
