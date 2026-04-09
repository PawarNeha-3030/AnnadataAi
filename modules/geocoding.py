import sqlite3
import json
import requests
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import time

class LocationService:
    def __init__(self, db_path='data/indian_locations.db'):
        self.db_path = db_path
        self.geolocator = Nominatim(user_agent="farmai_assistant")
        self.init_database()
        
    def init_database(self):
        """Initialize database with Indian locations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS states (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                code TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS districts (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                state_id INTEGER,
                latitude REAL,
                longitude REAL,
                FOREIGN KEY (state_id) REFERENCES states(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS villages (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                district_id INTEGER,
                pincode TEXT,
                latitude REAL,
                longitude REAL,
                FOREIGN KEY (district_id) REFERENCES districts(id)
            )
        ''')
        
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
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'gps'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_gps_location(self):
        """Get current GPS location (to be called from frontend)"""
        # This will be implemented with JavaScript in frontend
        # Returns template data structure
        return {
            "latitude": None,
            "longitude": None,
            "accuracy": None,
            "timestamp": None
        }
    
    def reverse_geocode(self, latitude, longitude):
        """Convert coordinates to address"""
        try:
            location = self.geolocator.reverse(f"{latitude}, {longitude}")
            if location:
                address = location.raw.get('address', {})
                return {
                    'village': address.get('village', address.get('hamlet', '')),
                    'district': address.get('county', address.get('district', '')),
                    'state': address.get('state', ''),
                    'pincode': address.get('postcode', ''),
                    'full_address': location.address
                }
        except Exception as e:
            print(f"Reverse geocoding error: {e}")
        return None
    
    def search_location(self, query):
        """Search for locations by name (village/district)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        results = []
        query = f"%{query}%"
        
        # Search villages
        cursor.execute('''
            SELECT v.name, d.name as district, s.name as state, 
                   v.latitude, v.longitude, v.pincode
            FROM villages v
            JOIN districts d ON v.district_id = d.id
            JOIN states s ON d.state_id = s.id
            WHERE v.name LIKE ? 
            LIMIT 10
        ''', (query,))
        
        for row in cursor.fetchall():
            results.append({
                'type': 'village',
                'name': row[0],
                'district': row[1],
                'state': row[2],
                'latitude': row[3],
                'longitude': row[4],
                'pincode': row[5]
            })
        
        # Search districts
        cursor.execute('''
            SELECT d.name, s.name as state, d.latitude, d.longitude
            FROM districts d
            JOIN states s ON d.state_id = s.id
            WHERE d.name LIKE ?
            LIMIT 10
        ''', (query,))
        
        for row in cursor.fetchall():
            results.append({
                'type': 'district',
                'name': row[0],
                'state': row[1],
                'latitude': row[2],
                'longitude': row[3]
            })
        
        conn.close()
        return results
    
    def save_user_location(self, user_id, location_data):
        """Save user's location to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_locations 
            (user_id, latitude, longitude, village, district, state, pincode, accuracy, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            location_data.get('latitude'),
            location_data.get('longitude'),
            location_data.get('village'),
            location_data.get('district'),
            location_data.get('state'),
            location_data.get('pincode'),
            location_data.get('accuracy'),
            location_data.get('source', 'gps')
        ))
        
        conn.commit()
        location_id = cursor.lastrowid
        conn.close()
        
        return location_id
    
    def get_states(self):
        """Get all Indian states"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM states ORDER BY name")
        states = [{'id': row[0], 'name': row[1]} for row in cursor.fetchall()]
        conn.close()
        return states
    
    def get_districts_by_state(self, state_id):
        """Get districts for a given state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM districts WHERE state_id = ? ORDER BY name", (state_id,))
        districts = [{'id': row[0], 'name': row[1]} for row in cursor.fetchall()]
        conn.close()
        return districts
    
    def validate_pincode(self, pincode):
        """Validate Indian pincode (6 digits)"""
        if len(pincode) == 6 and pincode.isdigit():
            return True
        return False
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates in km"""
        return geodesic((lat1, lon1), (lat2, lon2)).km