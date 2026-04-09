#!/usr/bin/env python3
"""
Test script to verify improved disease detection
"""

import requests
import json
from PIL import Image, ImageDraw
import io
import numpy as np

def create_test_image(disease_type="healthy"):
    """Create a test image with specific disease characteristics"""
    # Create a 300x300 green base image (healthy plant)
    img = Image.new('RGB', (300, 300), (50, 150, 50))
    draw = ImageDraw.Draw(img)
    
    if disease_type == "early_blight":
        # Add dark spots with yellow halos (Early Blight)
        for i in range(5):
            x, y = 50 + i*40, 50 + i*30
            # Dark center
            draw.ellipse([x-15, y-15, x+15, y+15], fill=(30, 20, 10))
            # Yellow halo
            draw.ellipse([x-25, y-25, x+25, y+25], outline=(200, 200, 50), width=3)
            
    elif disease_type == "powdery_mildew":
        # Add white powdery patches
        for i in range(8):
            x, y = 40 + i*30, 60 + (i%3)*80
            draw.ellipse([x-20, y-20, x+20, y+20], fill=(240, 240, 240))
            
    elif disease_type == "late_blight":
        # Add large dark water-soaked areas
        draw.rectangle([50, 50, 150, 120], fill=(20, 15, 10))
        draw.rectangle([180, 100, 250, 180], fill=(25, 20, 15))
        
    elif disease_type == "leaf_spot":
        # Add small brown circular spots
        for i in range(12):
            x, y = 30 + (i%4)*60, 40 + (i//4)*60
            draw.ellipse([x-8, y-8, x+8, y+8], fill=(80, 50, 30))
    
    return img

def test_disease_detection():
    """Test the disease detection API with different image types"""
    
    print("🧪 Testing Improved Disease Detection System")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ("healthy", "Healthy Plant"),
        ("early_blight", "Early Blight"),
        ("powdery_mildew", "Powdery Mildew"),
        ("late_blight", "Late Blight"),
        ("leaf_spot", "Leaf Spot")
    ]
    
    results = []
    
    for disease_type, expected_name in test_cases:
        print(f"\n🔬 Testing: {expected_name}")
        
        # Create test image
        test_img = create_test_image(disease_type)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Send to API
        try:
            files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
            response = requests.post('http://localhost:5000/api/detect-disease', files=files)
            
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    detected = data['disease_name']
                    confidence = data['confidence']
                    method = data.get('detection_method', 'Unknown')
                    
                    print(f"   ✅ Detected: {detected}")
                    print(f"   📊 Confidence: {confidence}%")
                    print(f"   🔧 Method: {method}")
                    
                    # Check if detection is reasonable
                    is_correct = any(keyword in detected.lower() for keyword in expected_name.lower().split())
                    results.append({
                        'test': expected_name,
                        'detected': detected,
                        'confidence': confidence,
                        'correct': is_correct,
                        'method': method
                    })
                    
                    if is_correct:
                        print(f"   ✅ CORRECT detection!")
                    else:
                        print(f"   ⚠️  Different detection (not necessarily wrong)")
                else:
                    print(f"   ❌ API Error: {data.get('error', 'Unknown')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("   ❌ Connection Error: Make sure the Flask server is running on port 5000")
            return
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    
    print(f"✅ Correct detections: {correct_count}/{total_count}")
    print(f"📈 Accuracy: {(correct_count/total_count)*100:.1f}%")
    
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    print(f"📊 Average confidence: {avg_confidence:.1f}%")
    
    methods = set(r['method'] for r in results)
    print(f"🔧 Detection methods used: {', '.join(methods)}")
    
    print("\n💡 Tips to improve accuracy:")
    print("   • Click 'Improve AI Model' button in the web interface")
    print("   • Use clear, well-lit images of plant leaves")
    print("   • Ensure disease symptoms are visible in the image")
    print("   • Try multiple images of the same plant for consistency")

if __name__ == "__main__":
    test_disease_detection()