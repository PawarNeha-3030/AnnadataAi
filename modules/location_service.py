# modules/location_service.py - UPDATED WITH SOIL DATASET SUPPORT
import sqlite3
import requests
import json
from datetime import datetime
import pandas as pd
import numpy as np
import os
import time
import re
import math

class LocationService:
    def __init__(self, db_path='data/annadatai_locations.db'):
        self.db_path = db_path
        self.init_database()
        self.load_soil_dataset()
        
    def load_soil_dataset(self):
        """Load soil dataset from CSV file"""
        self.soil_dataset = None
        self.soil_loaded = False
        
        try:
            # Try different possible dataset paths
            possible_paths = [
                'datasets/Agritech.csv',
                'datasets/soil_data.csv',
                'datasets/soil_dataset.csv',
                'datasets/crop_yield.csv'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Loading soil dataset from: {path}")
                    self.soil_dataset = pd.read_csv(path)
                    
                    # Clean column names
                    self.soil_dataset.columns = self.soil_dataset.columns.str.strip()
                    
                    # Check if we have location columns
                    location_columns = ['Latitude', 'latitude', 'lat', 'LAT']
                    longitude_columns = ['Longitude', 'longitude', 'lon', 'LON', 'lng']
                    soil_type_columns = ['Soil_Type', 'soil_type', 'SoilType', 'Soil', 'soil']
                    
                    # Find actual column names
                    self.lat_col = next((col for col in location_columns if col in self.soil_dataset.columns), None)
                    self.lon_col = next((col for col in longitude_columns if col in self.soil_dataset.columns), None)
                    self.soil_col = next((col for col in soil_type_columns if col in self.soil_dataset.columns), None)
                    
                    if self.lat_col and self.lon_col and self.soil_col:
                        print(f"Dataset loaded successfully with {len(self.soil_dataset)} records")
                        print(f"Columns: Latitude={self.lat_col}, Longitude={self.lon_col}, Soil={self.soil_col}")
                        self.soil_loaded = True
                        break
                    else:
                        print(f"Dataset found but missing required columns in {path}")
                        self.soil_dataset = None
                        
        except Exception as e:
            print(f"Error loading soil dataset: {e}")
            self.soil_dataset = None
            self.soil_loaded = False
    
    def get_soil_from_coordinates(self, latitude, longitude):
        """Get soil type from dataset based on coordinates"""
        if not self.soil_loaded or self.soil_dataset is None:
            return self.get_soil_fallback(latitude, longitude)
        
        try:
            # Convert to numeric
            self.soil_dataset[self.lat_col] = pd.to_numeric(self.soil_dataset[self.lat_col], errors='coerce')
            self.soil_dataset[self.lon_col] = pd.to_numeric(self.soil_dataset[self.lon_col], errors='coerce')
            
            # Filter out null values
            valid_data = self.soil_dataset.dropna(subset=[self.lat_col, self.lon_col, self.soil_col])
            
            if len(valid_data) == 0:
                return self.get_soil_fallback(latitude, longitude)
            
            # Find nearest soil data point
            distances = []
            for idx, row in valid_data.iterrows():
                try:
                    lat = float(row[self.lat_col])
                    lon = float(row[self.lon_col])
                    
                    # Calculate distance using Haversine formula
                    dlat = math.radians(latitude - lat)
                    dlon = math.radians(longitude - lon)
                    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat)) * math.cos(math.radians(latitude)) * math.sin(dlon/2) * math.sin(dlon/2)
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                    distance = 6371 * c  # Earth radius in km
                    
                    distances.append((distance, row))
                except (ValueError, TypeError):
                    continue
            
            if not distances:
                return self.get_soil_fallback(latitude, longitude)
            
            # Get the nearest point
            distances.sort(key=lambda x: x[0])
            nearest_distance, nearest_row = distances[0]
            
            soil_type = str(nearest_row[self.soil_col]).strip()
            
            # Get additional soil properties if available
            soil_properties = self.get_soil_properties(soil_type)
            
            return {
                'success': True,
                'soil_type': soil_type,
                'distance_km': round(nearest_distance, 2),
                'data_source': 'dataset',
                'dataset_location': {
                    'dataset_lat': float(nearest_row[self.lat_col]),
                    'dataset_lon': float(nearest_row[self.lon_col])
                },
                'properties': soil_properties,
                'dataset_size': len(self.soil_dataset)
            }
            
        except Exception as e:
            print(f"Error getting soil from coordinates: {e}")
            return self.get_soil_fallback(latitude, longitude)
    
    def get_soil_properties(self, soil_type):
        """Get soil properties based on soil type"""
        # Soil properties database
        soil_properties_db = {
            'red': {
                'name': 'Red Soil',
                'color': 'red',
                'ph': '5.5 - 6.5 (Acidic)',
                'texture': 'Sandy to Clay Loam',
                'drainage': 'Good',
                'nutrients': 'Low in Nitrogen, Phosphorus; Rich in Iron',
                'waterRetention': 'Medium',
                'improvement': 'Add organic compost, lime for pH adjustment, use nitrogen-fixing crops',
                'crops': ['Cotton', 'Wheat', 'Millets', 'Pulses', 'Oilseeds', 'Tobacco']
            },
            'black': {
                'name': 'Black Soil (Regur)',
                'color': 'black',
                'ph': '7.5 - 8.5 (Alkaline)',
                'texture': 'Clayey',
                'drainage': 'Poor',
                'nutrients': 'Rich in Iron, Lime, Alumina, Magnesia; Poor in Phosphorus',
                'waterRetention': 'High',
                'improvement': 'Add gypsum for drainage, organic matter, phosphorus fertilizers',
                'crops': ['Cotton', 'Sugarcane', 'Wheat', 'Tobacco', 'Oilseeds', 'Citrus']
            },
            'alluvial': {
                'name': 'Alluvial Soil',
                'color': 'alluvial',
                'ph': '6.5 - 7.5 (Neutral)',
                'texture': 'Sandy Loam to Silty Loam',
                'drainage': 'Good to Moderate',
                'nutrients': 'Rich in Potash, Lime; Moderate in Phosphorus',
                'waterRetention': 'Medium to High',
                'improvement': 'Crop rotation, green manure, balanced fertilization',
                'crops': ['Rice', 'Wheat', 'Sugarcane', 'Jute', 'Maize', 'Oilseeds', 'Vegetables']
            },
            'laterite': {
                'name': 'Laterite Soil',
                'color': 'laterite',
                'ph': '5.0 - 6.0 (Acidic)',
                'texture': 'Porous',
                'drainage': 'Excessive',
                'nutrients': 'Poor in Nitrogen, Phosphorus, Potash; Rich in Iron, Aluminum',
                'waterRetention': 'Low',
                'improvement': 'Add organic matter, lime, phosphorus fertilizers, terracing for water conservation',
                'crops': ['Tea', 'Coffee', 'Rubber', 'Cashew', 'Coconut', 'Tapioca']
            },
            'sandy': {
                'name': 'Sandy Soil',
                'color': 'sandy',
                'ph': '6.0 - 7.0 (Slightly Acidic to Neutral)',
                'texture': 'Coarse',
                'drainage': 'Very Good',
                'nutrients': 'Very Low',
                'waterRetention': 'Very Low',
                'improvement': 'Add clay, organic compost, mulch heavily, use drip irrigation',
                'crops': ['Groundnut', 'Watermelon', 'Pulses', 'Millets', 'Cucumber', 'Carrot']
            },
            'clay': {
                'name': 'Clay Soil',
                'color': 'clay',
                'ph': '6.0 - 7.0',
                'texture': 'Fine',
                'drainage': 'Poor',
                'nutrients': 'Rich in Nutrients',
                'waterRetention': 'High',
                'improvement': 'Add sand, organic matter, improve drainage',
                'crops': ['Rice', 'Wheat', 'Cabbage', 'Broccoli']
            },
            'loam': {
                'name': 'Loam Soil',
                'color': 'loam',
                'ph': '6.0 - 7.0',
                'texture': 'Medium',
                'drainage': 'Good',
                'nutrients': 'Balanced',
                'waterRetention': 'Good',
                'improvement': 'Maintain organic matter, regular composting',
                'crops': ['Most vegetables', 'Fruits', 'Grains', 'Flowers']
            }
        }
        
        # Normalize soil type for lookup
        soil_lower = soil_type.lower()
        
        # Check for keywords in soil type
        for key, properties in soil_properties_db.items():
            if key in soil_lower:
                return properties
        
        # Check for common soil type patterns
        if any(word in soil_lower for word in ['red', 'lal']):
            return soil_properties_db['red']
        elif any(word in soil_lower for word in ['black', 'regur', 'kali']):
            return soil_properties_db['black']
        elif any(word in soil_lower for word in ['alluvial', 'river', 'silt']):
            return soil_properties_db['alluvial']
        elif any(word in soil_lower for word in ['laterite', 'lateritic']):
            return soil_properties_db['laterite']
        elif any(word in soil_lower for word in ['sandy', 'sand', 'desert']):
            return soil_properties_db['sandy']
        elif any(word in soil_lower for word in ['clay', 'clayey']):
            return soil_properties_db['clay']
        elif any(word in soil_lower for word in ['loam', 'loamy']):
            return soil_properties_db['loam']
        
        # Default fallback
        return soil_properties_db['loam']
    
    def get_soil_fallback(self, latitude, longitude):
        """Fallback soil detection based on coordinates"""
        # India-specific soil mapping
        try:
            # Northern Plains (Alluvial)
            if latitude > 20 and latitude < 32 and longitude > 70 and longitude < 88:
                soil_type = 'Alluvial Soil'
            # Deccan Plateau (Black Soil)
            elif latitude > 15 and latitude < 22 and longitude > 72 and longitude < 80:
                soil_type = 'Black Soil'
            # Southern Peninsula (Red Soil)
            elif latitude > 8 and latitude < 20 and longitude > 76 and longitude < 88:
                soil_type = 'Red Soil'
            # Western Ghats (Laterite)
            elif latitude > 8 and latitude < 20 and longitude > 73 and longitude < 77:
                soil_type = 'Laterite Soil'
            # Rajasthan/Desert (Sandy)
            elif latitude > 24 and latitude < 30 and longitude > 69 and longitude < 76:
                soil_type = 'Sandy Soil'
            # Coastal Regions (Sandy)
            elif latitude < 15 or (latitude > 8 and longitude > 78 and longitude < 82):
                soil_type = 'Sandy Soil'
            # Default to alluvial for most of India
            else:
                soil_type = 'Alluvial Soil'
            
            properties = self.get_soil_properties(soil_type)
            
            return {
                'success': True,
                'soil_type': soil_type,
                'distance_km': None,
                'data_source': 'geographic_fallback',
                'dataset_location': None,
                'properties': properties,
                'dataset_size': 0
            }
            
        except Exception as e:
            print(f"Error in soil fallback: {e}")
            # Ultimate fallback
            properties = self.get_soil_properties('loam')
            return {
                'success': True,
                'soil_type': 'Loam Soil',
                'distance_km': None,
                'data_source': 'ultimate_fallback',
                'properties': properties
            }
    
    def get_soil_for_state(self, state_name):
        """Get typical soil type for a state"""
        state_soil_map = {
            'Maharashtra': 'Black Soil',
            'Gujarat': 'Black Soil',
            'Madhya Pradesh': 'Black Soil',
            'Karnataka': 'Red Soil',
            'Tamil Nadu': 'Red Soil',
            'Andhra Pradesh': 'Red Soil',
            'Telangana': 'Red Soil',
            'Odisha': 'Red Soil',
            'Uttar Pradesh': 'Alluvial Soil',
            'Punjab': 'Alluvial Soil',
            'Haryana': 'Alluvial Soil',
            'Bihar': 'Alluvial Soil',
            'West Bengal': 'Alluvial Soil',
            'Kerala': 'Laterite Soil',
            'Goa': 'Laterite Soil',
            'Rajasthan': 'Sandy Soil',
            'Assam': 'Alluvial Soil',
            'Jharkhand': 'Red Soil',
            'Chhattisgarh': 'Red Soil',
            'Uttarakhand': 'Mountain Soil',
            'Himachal Pradesh': 'Mountain Soil'
        }
        
        soil_type = state_soil_map.get(state_name, 'Loam Soil')
        properties = self.get_soil_properties(soil_type)
        
        return {
            'success': True,
            'soil_type': soil_type,
            'data_source': 'state_mapping',
            'properties': properties
        }
    
    def init_database(self):
        """Initialize database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create user locations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_locations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    latitude REAL,
                    longitude REAL,
                    village TEXT,
                    district TEXT,
                    state TEXT,
                    pincode TEXT,
                    accuracy REAL,
                    source TEXT,
                    soil_type TEXT,
                    soil_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create states table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS states (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL
                )
            ''')
            
            # Create districts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS districts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state_id TEXT,
                    name TEXT NOT NULL,
                    FOREIGN KEY (state_id) REFERENCES states(id)
                )
            ''')
            
            # Create villages table for common locations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS villages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    district TEXT,
                    state TEXT,
                    latitude REAL,
                    longitude REAL
                )
            ''')
            
            # Populate states and districts if empty
            cursor.execute('SELECT COUNT(*) FROM states')
            if cursor.fetchone()[0] == 0:
                self.populate_states_districts(conn, cursor)
            
            # Populate sample villages if empty
            cursor.execute('SELECT COUNT(*) FROM villages')
            if cursor.fetchone()[0] == 0:
                self.populate_villages(conn, cursor)
            
            conn.commit()
            conn.close()
            print(f"Database initialized at {self.db_path}")
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def populate_states_districts(self, conn, cursor):
        """Populate states and districts tables with Indian data"""
        # Indian states data
        states_data = [
            ('MH', 'Maharashtra'),
            ('UP', 'Uttar Pradesh'),
            ('KA', 'Karnataka'),
            ('GJ', 'Gujarat'),
            ('RJ', 'Rajasthan'),
            ('TN', 'Tamil Nadu'),
            ('MP', 'Madhya Pradesh'),
            ('AP', 'Andhra Pradesh'),
            ('WB', 'West Bengal'),
            ('PB', 'Punjab'),
            ('OD', 'Odisha'),
            ('KL', 'Kerala'),
            ('HR', 'Haryana'),
            ('BR', 'Bihar'),
            ('TG', 'Telangana'),
            ('AS', 'Assam'),
            ('JH', 'Jharkhand'),
            ('CH', 'Chhattisgarh'),
            ('UK', 'Uttarakhand'),
            ('HP', 'Himachal Pradesh'),
        ]
        
        # Districts data for each state
        districts_data = {
            'MH': ['Pune', 'Mumbai', 'Nagpur', 'Nashik', 'Aurangabad', 'Thane', 'Solapur', 'Kolhapur'],
            'UP': ['Lucknow', 'Kanpur', 'Varanasi', 'Agra', 'Allahabad', 'Meerut', 'Ghaziabad', 'Aligarh'],
            'KA': ['Bangalore', 'Mysore', 'Hubli', 'Mangalore', 'Belgaum', 'Gulbarga', 'Davanagere'],
            'GJ': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Bhavnagar', 'Jamnagar', 'Junagadh'],
            'RJ': ['Jaipur', 'Jodhpur', 'Udaipur', 'Kota', 'Ajmer', 'Bikaner', 'Alwar'],
            'TN': ['Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli', 'Salem', 'Tirunelveli'],
            'MP': ['Bhopal', 'Indore', 'Jabalpur', 'Gwalior', 'Ujjain', 'Sagar', 'Rewa'],
            'AP': ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Nellore', 'Kurnool', 'Kakinada'],
            'WB': ['Kolkata', 'Howrah', 'Asansol', 'Siliguri', 'Durgapur', 'Bardhaman'],
            'PB': ['Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala', 'Bathinda', 'Mohali'],
            'OD': ['Bhubaneswar', 'Cuttack', 'Rourkela', 'Berhampur', 'Sambalpur', 'Puri'],
            'KL': ['Thiruvananthapuram', 'Kochi', 'Kozhikode', 'Thrissur', 'Malappuram', 'Kannur'],
            'HR': ['Gurgaon', 'Faridabad', 'Panipat', 'Ambala', 'Yamunanagar', 'Rohtak'],
            'BR': ['Patna', 'Gaya', 'Bhagalpur', 'Muzaffarpur', 'Purnia', 'Darbhanga'],
            'TG': ['Hyderabad', 'Warangal', 'Nizamabad', 'Khammam', 'Karimnagar', 'Ramagundam'],
            'AS': ['Guwahati', 'Silchar', 'Dibrugarh', 'Jorhat', 'Nagaon', 'Tinsukia'],
            'JH': ['Ranchi', 'Jamshedpur', 'Dhanbad', 'Bokaro', 'Hazaribagh', 'Deoghar'],
            'CH': ['Raipur', 'Bhilai', 'Bilaspur', 'Korba', 'Raigarh', 'Jagdalpur'],
            'UK': ['Dehradun', 'Haridwar', 'Roorkee', 'Haldwani', 'Rudrapur', 'Kashipur'],
            'HP': ['Shimla', 'Solan', 'Dharamshala', 'Mandi', 'Palampur', 'Kullu'],
        }
        
        # Insert states
        cursor.executemany('INSERT INTO states (id, name) VALUES (?, ?)', states_data)
        
        # Insert districts
        for state_id, districts in districts_data.items():
            for district in districts:
                cursor.execute('INSERT INTO districts (state_id, name) VALUES (?, ?)', (state_id, district))
    
    def populate_villages(self, conn, cursor):
        """Populate villages table with common locations"""
        villages = [
            # Hyderabad area villages
            ('Gachibowli', 'Hyderabad', 'Telangana', 17.4401, 78.3489),
            ('Madhapur', 'Hyderabad', 'Telangana', 17.4483, 78.3915),
            ('HITEC City', 'Hyderabad', 'Telangana', 17.4454, 78.3813),
            ('Kondapur', 'Hyderabad', 'Telangana', 17.4639, 78.3636),
            ('Jubilee Hills', 'Hyderabad', 'Telangana', 17.4250, 78.4266),
            ('Banjara Hills', 'Hyderabad', 'Telangana', 17.4130, 78.4436),
            ('Ameerpet', 'Hyderabad', 'Telangana', 17.4372, 78.4458),
            ('Kukatpally', 'Hyderabad', 'Telangana', 17.4849, 78.4139),
            ('Miyapur', 'Hyderabad', 'Telangana', 17.4967, 78.3573),
            ('Serilingampally', 'Hyderabad', 'Telangana', 17.4930, 78.3312),
            
            # Pune area villages
            ('Hinjewadi', 'Pune', 'Maharashtra', 18.5921, 73.7386),
            ('Wagholi', 'Pune', 'Maharashtra', 18.5761, 73.9835),
            ('Kharadi', 'Pune', 'Maharashtra', 18.5521, 73.9470),
            ('Baner', 'Pune', 'Maharashtra', 18.5595, 73.7866),
            ('Aundh', 'Pune', 'Maharashtra', 18.5611, 73.8075),
            ('Kothrud', 'Pune', 'Maharashtra', 18.5074, 73.8075),
            ('Katraj', 'Pune', 'Maharashtra', 18.4529, 73.8652),
            ('Hadapsar', 'Pune', 'Maharashtra', 18.5066, 73.9410),
            
            # Bangalore area villages
            ('Whitefield', 'Bangalore', 'Karnataka', 12.9698, 77.7499),
            ('Electronic City', 'Bangalore', 'Karnataka', 12.8456, 77.6602),
            ('Marathahalli', 'Bangalore', 'Karnataka', 12.9593, 77.6974),
            ('Koramangala', 'Bangalore', 'Karnataka', 12.9279, 77.6271),
            ('Indiranagar', 'Bangalore', 'Karnataka', 12.9784, 77.6408),
            ('Jayanagar', 'Bangalore', 'Karnataka', 12.9304, 77.5835),
        ]
        
        for village in villages:
            cursor.execute('''
                INSERT INTO villages (name, district, state, latitude, longitude)
                VALUES (?, ?, ?, ?, ?)
            ''', village)
    
    def reverse_geocode_nominatim(self, latitude, longitude):
        """
        Use OpenStreetMap Nominatim API to get address from coordinates
        """
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?lat={latitude}&lon={longitude}&format=json&addressdetails=1&zoom=18"
            headers = {
                'User-Agent': 'FarmAI-Hackathon-App/1.0 (contact@example.com)',
                'Accept-Language': 'en'
            }
            
            # Add delay to respect Nominatim usage policy
            time.sleep(1)
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                address = data.get('address', {})
                
                # Extract detailed address components
                village = self.extract_location_name(address, latitude, longitude)
                
                # Get more specific names
                locality = address.get('locality') or address.get('suburb') or address.get('neighbourhood') or ''
                road = address.get('road') or address.get('street') or ''
                district = address.get('county') or address.get('district') or address.get('city') or address.get('city_district') or ''
                state = address.get('state') or address.get('state_district') or ''
                pincode = address.get('postcode') or ''
                
                # If we don't have a village name, try to construct one
                if not village:
                    if locality:
                        village = locality
                    elif road:
                        village = road
                    else:
                        # Try to find nearest known village
                        village = self.find_nearest_village(latitude, longitude)
                
                # Create a detailed display name
                display_parts = []
                if village:
                    display_parts.append(village)
                if locality and locality != village:
                    display_parts.append(locality)
                if district:
                    display_parts.append(district)
                if state:
                    display_parts.append(state)
                
                full_address = ', '.join(display_parts) if display_parts else data.get('display_name', '')
                
                return {
                    'village': village,
                    'locality': locality,
                    'road': road,
                    'district': district,
                    'state': state,
                    'pincode': pincode,
                    'full_address': full_address,
                    'display_name': data.get('display_name', ''),
                    'source': 'nominatim',
                    'detailed': True
                }
            else:
                print(f"Nominatim API error: {response.status_code}")
                return self.reverse_geocode_offline(latitude, longitude)
                
        except requests.exceptions.Timeout:
            print("Nominatim API timeout")
            return self.reverse_geocode_offline(latitude, longitude)
        except Exception as e:
            print(f"Reverse geocoding error: {e}")
            return self.reverse_geocode_offline(latitude, longitude)
    
    def extract_location_name(self, address, latitude, longitude):
        """Extract the most specific location name from address"""
        # Try different address components in order of specificity
        location_keys = [
            'village', 'hamlet', 'locality', 'suburb', 'neighbourhood',
            'town', 'city_district', 'city', 'county', 'district'
        ]
        
        location_parts = []
        
        # Get the primary location name
        primary_location = None
        for key in location_keys:
            if key in address and address[key]:
                primary_location = address[key]
                break
        
        # Build detailed location string
        if primary_location:
            location_parts.append(primary_location)
            
        # Add area/suburb if different from primary
        if address.get('suburb') and address.get('suburb') != primary_location:
            location_parts.append(f"({address['suburb']} area)")
            
        # Add road if available
        if address.get('road'):
            location_parts.append(f"near {address['road']}")
            
        # Add city if not already primary
        if address.get('city') and address.get('city') != primary_location:
            location_parts.append(f"in {address['city']}")
            
        if location_parts:
            return ', '.join(location_parts)
        
        # If no specific name found, try to extract from display name
        return self.extract_name_from_coordinates(latitude, longitude)
    
    def extract_name_from_coordinates(self, latitude, longitude):
        """Extract a name from coordinates based on known locations"""
        # Known landmarks and their coordinates
        landmarks = {
            # Hyderabad landmarks
            'Gachibowli': (17.4401, 78.3489),
            'HITEC City': (17.4454, 78.3813),
            'Cyber Towers': (17.4436, 78.3818),
            'IIIT Hyderabad': (17.4448, 78.3495),
            'Shilparamam': (17.4489, 78.3730),
            'KBR Park': (17.4208, 78.4250),
            'Charminar': (17.3616, 78.4747),
            'Golconda Fort': (17.3833, 78.4011),
            
            # General locations
            'Industrial Area': (17.4700, 78.3800),
            'Residential Area': (17.4500, 78.3900),
            'Commercial Area': (17.4400, 78.3800),
        }
        
        # Find the nearest landmark
        nearest_landmark = None
        min_distance = float('inf')
        
        for landmark, (landmark_lat, landmark_lon) in landmarks.items():
            distance = ((latitude - landmark_lat) ** 2 + (longitude - landmark_lon) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_landmark = landmark
        
        # Convert distance to approximate distance in km
        distance_km = min_distance * 111  # 1 degree ≈ 111km
        
        if nearest_landmark and distance_km < 5:  # Within 5km
            return nearest_landmark
        elif nearest_landmark and distance_km < 10:  # Within 10km
            return f"Near {nearest_landmark}"
        
        return None
    
    def find_nearest_village(self, latitude, longitude):
        """Find the nearest known village from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT name, district, state, latitude, longitude
                FROM villages
                WHERE ABS(latitude - ?) < 0.5 AND ABS(longitude - ?) < 0.5
                ORDER BY (ABS(latitude - ?) + ABS(longitude - ?))
                LIMIT 1
            ''', (latitude, longitude, latitude, longitude))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                village_name, district, state, village_lat, village_lon = result
                distance = ((latitude - village_lat) ** 2 + (longitude - village_lon) ** 2) ** 0.5
                distance_km = distance * 111
                
                if distance_km < 2:  # Within 2km
                    return village_name
                elif distance_km < 5:  # Within 5km
                    return f"Near {village_name}"
                elif distance_km < 10:  # Within 10km
                    return f"Area around {village_name}"
            
            return None
            
        except Exception as e:
            print(f"Error finding nearest village: {e}")
            return None
    
    def reverse_geocode_offline(self, latitude, longitude):
        """Enhanced offline reverse geocoding"""
        try:
            # First try to find exact match in villages database
            nearest_village = self.find_nearest_village(latitude, longitude)
            
            if nearest_village:
                # Get district and state for this village
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT district, state FROM villages WHERE name = ? LIMIT 1
                ''', (nearest_village.split()[-1] if ' ' in nearest_village else nearest_village,))
                
                result = cursor.fetchone()
                if result:
                    district, state = result
                    
                    # Calculate approximate distance
                    cursor.execute('''
                        SELECT latitude, longitude FROM villages WHERE name = ?
                    ''', (nearest_village.split()[-1] if ' ' in nearest_village else nearest_village,))
                    
                    coords = cursor.fetchone()
                    if coords:
                        village_lat, village_lon = coords
                        distance = ((latitude - village_lat) ** 2 + (longitude - village_lon) ** 2) ** 0.5
                        distance_km = distance * 111
                        
                        if distance_km < 1:
                            location_desc = nearest_village
                        elif distance_km < 3:
                            location_desc = f"Near {nearest_village}"
                        else:
                            location_desc = f"Area around {nearest_village}"
                    else:
                        location_desc = nearest_village
                    
                    conn.close()
                    
                    return {
                        'village': nearest_village,
                        'locality': nearest_village,
                        'district': district,
                        'state': state,
                        'pincode': '',
                        'full_address': f"{location_desc}, {district}, {state}",
                        'source': 'offline_village_db',
                        'detailed': True
                    }
            
            # If no village found, fall back to city-based estimation
            city_coordinates = {
                'Hyderabad': (17.3850, 78.4867),
                'Pune': (18.5204, 73.8567),
                'Mumbai': (19.0760, 72.8777),
                'Bangalore': (12.9716, 77.5946),
                'Delhi': (28.6139, 77.2090),
                'Chennai': (13.0827, 80.2707),
                'Kolkata': (22.5726, 88.3639),
                'Ahmedabad': (23.0225, 72.5714),
                'Jaipur': (26.9124, 75.7873),
            }
            
            nearest_city = None
            min_distance = float('inf')
            
            for city, (city_lat, city_lon) in city_coordinates.items():
                distance = ((latitude - city_lat) ** 2 + (longitude - city_lon) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    nearest_city = city
            
            if nearest_city:
                distance_km = min_distance * 111
                
                # Determine area within city
                area_name = self.get_city_area_name(nearest_city, latitude, longitude)
                
                if distance_km < 20:  # Within 20km of city center
                    if area_name:
                        location_desc = f"{area_name}, {nearest_city}"
                    else:
                        location_desc = f"Area of {nearest_city}"
                else:
                    location_desc = f"Region near {nearest_city}"
                
                # Get state for the city
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT s.name 
                    FROM districts d
                    JOIN states s ON d.state_id = s.id
                    WHERE d.name LIKE ?
                    LIMIT 1
                ''', (f'%{nearest_city}%',))
                
                result = cursor.fetchone()
                state = result[0] if result else 'Unknown'
                conn.close()
                
                return {
                    'village': area_name or nearest_city,
                    'locality': area_name or '',
                    'district': nearest_city,
                    'state': state,
                    'pincode': '',
                    'full_address': location_desc,
                    'source': 'offline_city_estimation',
                    'detailed': True
                }
            
            # Final fallback
            return {
                'village': '',
                'locality': '',
                'district': '',
                'state': '',
                'pincode': '',
                'full_address': f"Coordinates: {latitude:.6f}, {longitude:.6f}",
                'source': 'coordinates_only',
                'detailed': False
            }
            
        except Exception as e:
            print(f"Offline reverse geocoding error: {e}")
            return {
                'village': '',
                'locality': '',
                'district': '',
                'state': '',
                'pincode': '',
                'full_address': f"Location at {latitude:.6f}, {longitude:.6f}",
                'source': 'error_fallback',
                'detailed': False
            }
    
    def get_city_area_name(self, city, latitude, longitude):
        """Get specific area name within a city based on coordinates"""
        # Hyderabad areas
        if city == 'Hyderabad':
            if 17.43 <= latitude <= 17.46 and 78.34 <= longitude <= 78.39:
                return 'Gachibowli/HITEC City'
            elif 17.41 <= latitude <= 17.43 and 78.42 <= longitude <= 78.45:
                return 'Jubilee Hills'
            elif 17.44 <= latitude <= 17.46 and 78.44 <= longitude <= 78.47:
                return 'Ameerpet'
            elif 17.36 <= latitude <= 17.38 and 78.47 <= longitude <= 78.48:
                return 'Charminar Area'
        
        # Pune areas
        elif city == 'Pune':
            if 18.59 <= latitude <= 18.61 and 73.73 <= longitude <= 73.75:
                return 'Hinjewadi'
            elif 18.55 <= latitude <= 18.57 and 73.94 <= longitude <= 73.96:
                return 'Kharadi'
            elif 18.55 <= latitude <= 18.57 and 73.78 <= longitude <= 73.80:
                return 'Baner'
        
        # Bangalore areas
        elif city == 'Bangalore':
            if 12.96 <= latitude <= 12.98 and 77.74 <= longitude <= 77.76:
                return 'Whitefield'
            elif 12.84 <= latitude <= 12.86 and 77.66 <= longitude <= 77.68:
                return 'Electronic City'
        
        return None
    
    def save_location(self, user_id, location_data):
        """Save location to database with soil data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            latitude = location_data.get('latitude')
            longitude = location_data.get('longitude')
            village = location_data.get('village', '')
            district = location_data.get('district', '')
            state = location_data.get('state', '')
            pincode = location_data.get('pincode', '')
            accuracy = location_data.get('accuracy')
            source = location_data.get('source', 'unknown')
            soil_type = location_data.get('soil_type', '')
            soil_data = location_data.get('soil_data', '')
            
            if isinstance(soil_data, dict):
                soil_data = json.dumps(soil_data)
            
            cursor.execute('''
                INSERT INTO user_locations 
                (user_id, latitude, longitude, village, district, state, pincode, accuracy, source, soil_type, soil_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                latitude,
                longitude,
                village,
                district,
                state,
                pincode,
                accuracy,
                source,
                soil_type,
                soil_data
            ))
            
            conn.commit()
            location_id = cursor.lastrowid
            conn.close()
            
            return {'success': True, 'location_id': location_id}
            
        except Exception as e:
            print(f"Save location error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_recent_location(self, user_id):
        """Get most recent location for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, village, district, state, latitude, longitude, pincode, accuracy, source, 
                       soil_type, soil_data, timestamp
                FROM user_locations 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''', (user_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                try:
                    soil_data = json.loads(row[10]) if row[10] else {}
                except:
                    soil_data = {}
                
                return {
                    'id': row[0],
                    'village': row[1],
                    'district': row[2],
                    'state': row[3],
                    'latitude': row[4],
                    'longitude': row[5],
                    'pincode': row[6],
                    'accuracy': row[7],
                    'source': row[8],
                    'soil_type': row[9],
                    'soil_data': soil_data,
                    'timestamp': row[11]
                }
            return None
            
        except Exception as e:
            print(f"Get recent location error: {e}")
            return None
    
    def get_all_user_locations(self, user_id):
        """Get all saved locations for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, village, district, state, latitude, longitude, pincode, accuracy, source, 
                       soil_type, soil_data, timestamp
                FROM user_locations 
                WHERE user_id = ? 
                ORDER BY timestamp DESC
            ''', (user_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            locations = []
            for row in rows:
                try:
                    soil_data = json.loads(row[10]) if row[10] else {}
                except:
                    soil_data = {}
                
                locations.append({
                    'id': row[0],
                    'village': row[1],
                    'district': row[2],
                    'state': row[3],
                    'latitude': row[4],
                    'longitude': row[5],
                    'pincode': row[6],
                    'accuracy': row[7],
                    'source': row[8],
                    'soil_type': row[9],
                    'soil_data': soil_data,
                    'timestamp': row[11]
                })
            
            return locations
            
        except Exception as e:
            print(f"Get all locations error: {e}")
            return []
    
    def get_location_by_id(self, user_id, location_id):
        """Get a specific location by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, village, district, state, latitude, longitude, pincode, accuracy, source, 
                       soil_type, soil_data, timestamp
                FROM user_locations 
                WHERE user_id = ? AND id = ?
            ''', (user_id, location_id))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                try:
                    soil_data = json.loads(row[10]) if row[10] else {}
                except:
                    soil_data = {}
                
                return {
                    'id': row[0],
                    'village': row[1],
                    'district': row[2],
                    'state': row[3],
                    'latitude': row[4],
                    'longitude': row[5],
                    'pincode': row[6],
                    'accuracy': row[7],
                    'source': row[8],
                    'soil_type': row[9],
                    'soil_data': soil_data,
                    'timestamp': row[11]
                }
            return None
            
        except Exception as e:
            print(f"Get location by ID error: {e}")
            return None
    
    def get_districts_by_state(self, state_code):
        """Get all districts for a given state"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT name FROM districts 
                WHERE state_id = ? 
                ORDER BY name
            ''', (state_code,))
            
            districts = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return districts
            
        except Exception as e:
            print(f"Get districts error: {e}")
            return self.get_districts_fallback(state_code)
    
    def get_districts_fallback(self, state_code):
        """Fallback districts data"""
        fallback_data = {
            'MH': ['Pune', 'Mumbai', 'Nagpur', 'Nashik', 'Aurangabad', 'Thane'],
            'UP': ['Lucknow', 'Kanpur', 'Varanasi', 'Agra', 'Allahabad', 'Meerut'],
            'KA': ['Bangalore', 'Mysore', 'Hubli', 'Mangalore', 'Belgaum'],
            'GJ': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Bhavnagar'],
            'RJ': ['Jaipur', 'Jodhpur', 'Udaipur', 'Kota', 'Ajmer'],
            'TN': ['Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli', 'Salem'],
            'MP': ['Bhopal', 'Indore', 'Jabalpur', 'Gwalior', 'Ujjain'],
            'AP': ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Nellore', 'Kurnool'],
            'WB': ['Kolkata', 'Howrah', 'Asansol', 'Siliguri', 'Durgapur'],
            'PB': ['Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala', 'Bathinda'],
        }
        return fallback_data.get(state_code, [])
    
    def search_location(self, query):
        """Search for locations by name"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            search_term = f'%{query}%'
            results = []
            
            cursor.execute('''
                SELECT d.name, s.name as state_name
                FROM districts d
                JOIN states s ON d.state_id = s.id
                WHERE d.name LIKE ?
                ORDER BY d.name
                LIMIT 20
            ''', (search_term,))
            
            for row in cursor.fetchall():
                results.append({
                    'type': 'district',
                    'name': row[0],
                    'state': row[1],
                    'display': f"{row[0]}, {row[1]}"
                })
            
            conn.close()
            
            if not results and len(query) >= 3:
                results = self.get_mock_search_results(query)
            
            return results
            
        except Exception as e:
            print(f"Search location error: {e}")
            return []
    
    def get_mock_search_results(self, query):
        """Return mock search results"""
        query_lower = query.lower()
        mock_villages = [
            {'name': 'Wagholi', 'district': 'Pune', 'state': 'Maharashtra'},
            {'name': 'Hinjewadi', 'district': 'Pune', 'state': 'Maharashtra'},
            {'name': 'Kothrud', 'district': 'Pune', 'state': 'Maharashtra'},
            {'name': 'Baner', 'district': 'Pune', 'state': 'Maharashtra'},
            {'name': 'Aundh', 'district': 'Pune', 'state': 'Maharashtra'},
            {'name': 'Hadapsar', 'district': 'Pune', 'state': 'Maharashtra'},
            {'name': 'Katraj', 'district': 'Pune', 'state': 'Maharashtra'},
            {'name': 'Kondhwa', 'district': 'Pune', 'state': 'Maharashtra'},
            {'name': 'Kharadi', 'district': 'Pune', 'state': 'Maharashtra'},
            {'name': 'Viman Nagar', 'district': 'Pune', 'state': 'Maharashtra'},
        ]
        
        results = []
        for village in mock_villages:
            if query_lower in village['name'].lower() or query_lower in village['district'].lower():
                results.append({
                    'type': 'village',
                    'name': village['name'],
                    'district': village['district'],
                    'state': village['state'],
                    'display': f"{village['name']}, {village['district']}, {village['state']}"
                })
        
        return results
    
    def validate_pincode(self, pincode):
        """Validate Indian pincode format"""
        if not pincode:
            return True
        
        pincode = str(pincode).strip()
        
        if len(pincode) != 6:
            return False
        
        if not pincode.isdigit():
            return False
        
        first_digit = int(pincode[0])
        if first_digit < 1 or first_digit > 8:
            return False
        
        return True
    
    def clear_user_locations(self, user_id):
        """Clear all locations for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM user_locations WHERE user_id = ?', (user_id,))
            
            conn.commit()
            deleted_count = cursor.rowcount
            conn.close()
            
            return {'success': True, 'deleted_count': deleted_count}
            
        except Exception as e:
            print(f"Clear locations error: {e}")
            return {'success': False, 'error': str(e)}
    
    # ==================== CROP RECOMMENDATION METHODS ====================
    
    def get_crop_recommendations(self, state, district=None, season=None):
        """
        Get crop recommendations based on crop_yield.csv dataset
        """
        try:
            dataset_path = 'datasets/crop_yield.csv'
            
            if not os.path.exists(dataset_path):
                print(f"Dataset not found at: {dataset_path}")
                return self.get_mock_crop_recommendations(state, district, season)
            
            df = pd.read_csv(dataset_path)
            df.columns = df.columns.str.strip()
            
            required_columns = ['State', 'Crop', 'Yield']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Missing columns in dataset: {missing_columns}")
                return self.get_mock_crop_recommendations(state, district, season)
            
            state_mask = df['State'].astype(str).str.contains(state, case=False, na=False)
            state_data = df[state_mask]
            
            if len(state_data) == 0:
                state_matches = df[df['State'].astype(str).str.contains(state.split()[0], case=False, na=False)]
                if len(state_matches) > 0:
                    state_data = state_matches
                else:
                    print(f"No data found for state: {state}")
                    return self.get_mock_crop_recommendations(state, district, season)
            
            if district and 'District' in df.columns:
                district_mask = state_data['District'].astype(str).str.contains(district, case=False, na=False)
                location_data = state_data[district_mask]
                if len(location_data) > 0:
                    state_data = location_data
            
            if season and 'Season' in df.columns:
                season_mask = state_data['Season'].astype(str).str.contains(season, case=False, na=False)
                seasonal_data = state_data[season_mask]
                if len(seasonal_data) > 0:
                    state_data = seasonal_data
            
            if len(state_data) > 0:
                crop_stats = state_data.groupby('Crop').agg({
                    'Yield': ['mean', 'count', 'std', 'min', 'max']
                }).round(2)
                
                crop_stats.columns = ['yield_mean', 'yield_count', 'yield_std', 'yield_min', 'yield_max']
                crop_stats = crop_stats.reset_index()
                crop_stats = crop_stats.sort_values('yield_mean', ascending=False)
                
                recommendations = []
                for _, row in crop_stats.head(10).iterrows():
                    crop_data = {
                        'crop': row['Crop'],
                        'avg_yield': float(row['yield_mean']),
                        'data_points': int(row['yield_count']),
                        'yield_range': f"{row['yield_min']} - {row['yield_max']}",
                        'reliability': 'high' if row['yield_count'] > 10 else 'medium'
                    }
                    recommendations.append(crop_data)
                
                seasonal_analysis = {}
                if 'Season' in state_data.columns:
                    season_counts = state_data['Season'].value_counts()
                    total = season_counts.sum()
                    for season_name, count in season_counts.items():
                        seasonal_analysis[season_name] = {
                            'percentage': round((count / total) * 100, 1),
                            'crop_count': int(count)
                        }
                
                overall_stats = {
                    'total_records': len(state_data),
                    'unique_crops': state_data['Crop'].nunique(),
                    'avg_yield_overall': round(state_data['Yield'].mean(), 2),
                    'max_yield_overall': round(state_data['Yield'].max(), 2),
                    'data_year_range': self.get_dataset_year_range(df)
                }
                
                return {
                    'success': True,
                    'state': state,
                    'district': district or 'Not specified',
                    'season': season or 'All seasons',
                    'recommended_crops': recommendations,
                    'seasonal_analysis': seasonal_analysis,
                    'overall_statistics': overall_stats,
                    'data_source': 'crop_yield.csv',
                    'timestamp': datetime.now().isoformat()
                }
            
            return self.get_mock_crop_recommendations(state, district, season)
            
        except Exception as e:
            print(f"Error in get_crop_recommendations: {e}")
            import traceback
            traceback.print_exc()
            return self.get_mock_crop_recommendations(state, district, season)
    
    def get_mock_crop_recommendations(self, state, district=None, season=None):
        """Provide mock recommendations when dataset is unavailable"""
        
        state_crops = {
            'Maharashtra': [
                {'crop': 'Sugarcane', 'avg_yield': 75000, 'season': 'Annual'},
                {'crop': 'Cotton', 'avg_yield': 500, 'season': 'Kharif'},
                {'crop': 'Soybean', 'avg_yield': 1200, 'season': 'Kharif'},
                {'crop': 'Rice', 'avg_yield': 2500, 'season': 'Kharif'},
                {'crop': 'Wheat', 'avg_yield': 3200, 'season': 'Rabi'}
            ],
            'Punjab': [
                {'crop': 'Wheat', 'avg_yield': 4500, 'season': 'Rabi'},
                {'crop': 'Rice', 'avg_yield': 4000, 'season': 'Kharif'},
                {'crop': 'Maize', 'avg_yield': 3000, 'season': 'Kharif'},
                {'crop': 'Cotton', 'avg_yield': 600, 'season': 'Kharif'}
            ],
            'Karnataka': [
                {'crop': 'Ragi', 'avg_yield': 1500, 'season': 'Kharif'},
                {'crop': 'Rice', 'avg_yield': 2800, 'season': 'Kharif'},
                {'crop': 'Maize', 'avg_yield': 3500, 'season': 'Kharif'},
                {'crop': 'Coffee', 'avg_yield': 800, 'season': 'Annual'}
            ],
            'default': [
                {'crop': 'Rice', 'avg_yield': 2000, 'season': 'Kharif'},
                {'crop': 'Wheat', 'avg_yield': 2500, 'season': 'Rabi'},
                {'crop': 'Maize', 'avg_yield': 2800, 'season': 'Kharif'},
                {'crop': 'Pulses', 'avg_yield': 800, 'season': 'Rabi'}
            ]
        }
        
        state_key = next((key for key in state_crops.keys() if key.lower() == state.lower()), 'default')
        crops_data = state_crops[state_key]
        
        if season:
            crops_data = [crop for crop in crops_data if crop['season'].lower() == season.lower()]
        
        recommendations = []
        for crop in crops_data:
            recommendations.append({
                'crop': crop['crop'],
                'avg_yield': crop['avg_yield'],
                'season': crop['season'],
                'data_points': 100,
                'reliability': 'medium'
            })
        
        return {
            'success': True,
            'state': state,
            'district': district or 'Not specified',
            'season': season or 'All seasons',
            'recommended_crops': recommendations,
            'seasonal_analysis': {'Kharif': 55, 'Rabi': 45} if not season else {},
            'overall_statistics': {
                'total_records': 100,
                'unique_crops': len(crops_data),
                'avg_yield_overall': sum(c['avg_yield'] for c in crops_data) / len(crops_data),
                'data_source': 'mock_data'
            },
            'data_source': 'mock',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_dataset_year_range(self, df):
        """Extract year range from dataset if available"""
        try:
            if 'Year' in df.columns:
                years = df['Year'].dropna().unique()
                if len(years) > 0:
                    return f"{int(min(years))} - {int(max(years))}"
            return "Year data not available"
        except:
            return "Unknown"
    
    def get_seasonal_calendar(self, state):
        """Get farming calendar for a state"""
        calendar = {
            'Kharif': {
                'sowing': 'June - July',
                'harvest': 'September - October',
                'major_crops': ['Rice', 'Maize', 'Cotton', 'Soybean']
            },
            'Rabi': {
                'sowing': 'October - November',
                'harvest': 'March - April',
                'major_crops': ['Wheat', 'Barley', 'Mustard', 'Gram']
            },
            'Zaid': {
                'sowing': 'March - April',
                'harvest': 'June - July',
                'major_crops': ['Watermelon', 'Cucumber', 'Bitter Gourd']
            }
        }
        
        return {
            'state': state,
            'calendar': calendar,
            'current_season': self.get_current_season(),
            'next_activity': self.get_next_farming_activity()
        }
    
    def get_current_season(self):
        """Determine current farming season"""
        current_month = datetime.now().month
        
        if 6 <= current_month <= 10:
            return 'Kharif'
        elif 11 <= current_month or current_month <= 2:
            return 'Rabi'
        else:
            return 'Zaid'
    
    def get_next_farming_activity(self):
        """Suggest next farming activity based on season"""
        current_season = self.get_current_season()
        
        activities = {
            'Kharif': 'Prepare for Rabi sowing in Oct-Nov',
            'Rabi': 'Harvest time, prepare for Zaid crops',
            'Zaid': 'Prepare for Kharif sowing in June'
        }
        
        return activities.get(current_season, 'Regular farm maintenance')

if __name__ == "__main__":
    service = LocationService()
    print("Location Service initialized")
    print(f"Soil dataset loaded: {service.soil_loaded}")
    if service.soil_loaded:
        print(f"Dataset size: {len(service.soil_dataset)} records")