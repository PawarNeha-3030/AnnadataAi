import sqlite3
import json

def populate_indian_locations():
    """Populate database with Indian states and districts"""
    conn = sqlite3.connect('data/indian_locations.db')
    cursor = conn.cursor()
    
    # Sample Indian states (you can expand this)
    states = [
        (1, 'Maharashtra', 'MH'),
        (2, 'Uttar Pradesh', 'UP'),
        (3, 'Karnataka', 'KA'),
        (4, 'Gujarat', 'GJ'),
        (5, 'Rajasthan', 'RJ'),
        (6, 'Tamil Nadu', 'TN'),
        (7, 'Madhya Pradesh', 'MP'),
        (8, 'Andhra Pradesh', 'AP'),
        (9, 'West Bengal', 'WB'),
        (10, 'Punjab', 'PB'),
    ]
    
    cursor.executemany("INSERT OR IGNORE INTO states VALUES (?, ?, ?)", states)
    
    # Sample districts (you should add more from actual data)
    districts = [
        (1, 'Pune', 1, 18.5204, 73.8567),
        (2, 'Mumbai', 1, 19.0760, 72.8777),
        (3, 'Nagpur', 1, 21.1458, 79.0882),
        (4, 'Lucknow', 2, 26.8467, 80.9462),
        (5, 'Kanpur', 2, 26.4499, 80.3319),
        (6, 'Bangalore', 3, 12.9716, 77.5946),
        (7, 'Mysore', 3, 12.2958, 76.6394),
    ]
    
    cursor.executemany("INSERT OR IGNORE INTO districts VALUES (?, ?, ?, ?, ?)", districts)
    
    # Sample villages
    villages = [
        (1, 'Wagholi', 1, '412207', 18.5769, 73.9486),
        (2, 'Hinjewadi', 1, '411057', 18.5928, 73.7305),
        (3, 'Kothrud', 1, '411038', 18.5074, 73.8077),
    ]
    
    cursor.executemany("INSERT OR IGNORE INTO villages VALUES (?, ?, ?, ?, ?, ?)", villages)
    
    conn.commit()
    conn.close()
    print("Database populated with sample Indian locations")

if __name__ == "__main__":
    populate_indian_locations()