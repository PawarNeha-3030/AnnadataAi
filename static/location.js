// Location Manager with Soil Type Detection FROM DATASET
const locationManager = {
    currentLocation: null,
    map: null,
    marker: null,
    soilData: null,
    
    // Soil type database based on Indian regions
    soilDatabase: {
        // Red Soil (Southern India, Eastern Ghats)
        red: {
            name: "Red Soil",
            color: "red",
            ph: "5.5 - 6.5 (Acidic)",
            texture: "Sandy to Clay Loam",
            drainage: "Good",
            nutrients: "Low in Nitrogen, Phosphorus; Rich in Iron",
            waterRetention: "Medium",
            improvement: "Add organic compost, lime for pH adjustment, use nitrogen-fixing crops",
            crops: ["Cotton", "Wheat", "Millets", "Pulses", "Oilseeds", "Tobacco"]
        },
        // Black Soil (Deccan Plateau, Maharashtra, Gujarat)
        black: {
            name: "Black Soil (Regur)",
            color: "black",
            ph: "7.5 - 8.5 (Alkaline)",
            texture: "Clayey",
            drainage: "Poor",
            nutrients: "Rich in Iron, Lime, Alumina, Magnesia; Poor in Phosphorus",
            waterRetention: "High",
            improvement: "Add gypsum for drainage, organic matter, phosphorus fertilizers",
            crops: ["Cotton", "Sugarcane", "Wheat", "Tobacco", "Oilseeds", "Citrus"]
        },
        // Alluvial Soil (Northern Plains, River Basins)
        alluvial: {
            name: "Alluvial Soil",
            color: "alluvial",
            ph: "6.5 - 7.5 (Neutral)",
            texture: "Sandy Loam to Silty Loam",
            drainage: "Good to Moderate",
            nutrients: "Rich in Potash, Lime; Moderate in Phosphorus",
            waterRetention: "Medium to High",
            improvement: "Crop rotation, green manure, balanced fertilization",
            crops: ["Rice", "Wheat", "Sugarcane", "Jute", "Maize", "Oilseeds", "Vegetables"]
        },
        // Laterite Soil (Western Ghats, Eastern Ghats, Hills)
        laterite: {
            name: "Laterite Soil",
            color: "laterite",
            ph: "5.0 - 6.0 (Acidic)",
            texture: "Porous",
            drainage: "Excessive",
            nutrients: "Poor in Nitrogen, Phosphorus, Potash; Rich in Iron, Aluminum",
            waterRetention: "Low",
            improvement: "Add organic matter, lime, phosphorus fertilizers, terracing for water conservation",
            crops: ["Tea", "Coffee", "Rubber", "Cashew", "Coconut", "Tapioca"]
        },
        // Sandy Soil (Desert Regions, Coastal Areas)
        sandy: {
            name: "Sandy Soil",
            color: "sandy",
            ph: "6.0 - 7.0 (Slightly Acidic to Neutral)",
            texture: "Coarse",
            drainage: "Very Good",
            nutrients: "Very Low",
            waterRetention: "Very Low",
            improvement: "Add clay, organic compost, mulch heavily, use drip irrigation",
            crops: ["Groundnut", "Watermelon", "Pulses", "Millets", "Cucumber", "Carrot"]
        },
        // Clay Soil
        clay: {
            name: "Clay Soil",
            color: "clay",
            ph: "6.0 - 7.0",
            texture: "Fine",
            drainage: "Poor",
            nutrients: "Rich in Nutrients",
            waterRetention: "High",
            improvement: "Add sand, organic matter, improve drainage",
            crops: ["Rice", "Wheat", "Cabbage", "Broccoli"]
        },
        // Loam Soil
        loam: {
            name: "Loam Soil",
            color: "loam",
            ph: "6.0 - 7.0",
            texture: "Medium",
            drainage: "Good",
            nutrients: "Balanced",
            waterRetention: "Good",
            improvement: "Maintain organic matter, regular composting",
            crops: ["Most vegetables", "Fruits", "Grains", "Flowers"]
        }
    },
    
    // Initialize
    init: function() {
        console.log("Initializing location manager...");
        
        // Initialize map
        this.map = L.map('mapContainer').setView([20.5937, 78.9629], 5); // Center on India
        
        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map);
        
        // Set up GPS detection
        document.getElementById('getGpsBtn').addEventListener('click', () => {
            this.getGPSLocation();
        });
        
        // Set up manual form
        document.getElementById('manualLocationForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveManualLocation();
        });
        
        // Load saved locations
        this.loadSavedLocations();
        
        // Set up state-district dropdowns
        this.setupStateDistrictDropdowns();
        
        console.log("Location manager initialized successfully");
    },
    
    // Soil detection using dataset
    detectSoilTypeFromDataset: function(latitude, longitude, stateName = null, isManual = false) {
        console.log("Detecting soil type from dataset for:", latitude, longitude, "State:", stateName, "Manual:", isManual);
        
        // Show progress bar
        this.showSoilDetectionProgress();
        
        // Call backend API to get soil data from dataset
        fetch('/api/get-soil-from-dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                latitude: latitude,
                longitude: longitude,
                state: stateName,
                is_manual: isManual
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success && data.soil_data) {
                // Update progress
                this.updateSoilProgress(100);
                
                // Add source info to soil data
                data.soil_data.source = data.source || 'dataset';
                data.soil_data.dataset_used = data.dataset_used || false;
                data.soil_data.distance_km = data.distance_km;
                
                // Display soil analysis
                this.displaySoilAnalysis(data.soil_data);
                
                // Enable save button
                document.getElementById('saveLocationBtn').disabled = false;
            } else {
                // Fallback to coordinate-based detection
                this.fallbackSoilDetection(latitude, longitude, stateName, isManual);
            }
        })
        .catch(error => {
            console.error('Error fetching soil data from dataset:', error);
            // Fallback to coordinate-based detection
            this.fallbackSoilDetection(latitude, longitude, stateName, isManual);
        });
    },
    
    // Fallback soil detection when dataset is not available
    fallbackSoilDetection: function(latitude, longitude, stateName = null, isManual = false) {
        console.log("Using fallback soil detection for:", latitude, longitude);
        
        let soilData = null;
        
        // If we have state name for manual location, use state-based mapping
        if (isManual && stateName) {
            soilData = this.detectSoilFromState(stateName);
        } else {
            // Otherwise use coordinate-based detection
            soilData = this.determineSoilFromCoordinates(latitude, longitude);
        }
        
        // Update progress
        this.updateSoilProgress(100);
        
        // Add source info
        soilData.source = isManual ? 'state_mapping_fallback' : 'coordinate_fallback';
        soilData.dataset_used = false;
        
        // Display soil analysis
        this.displaySoilAnalysis(soilData);
        
        // Enable save button
        document.getElementById('saveLocationBtn').disabled = false;
    },
    
    // Determine soil from coordinates (fallback method)
    determineSoilFromCoordinates: function(lat, lng) {
        // India-specific soil mapping
        // Northern Plains (Alluvial)
        if (lat > 20 && lat < 32 && lng > 70 && lng < 88) {
            return this.soilDatabase.alluvial;
        }
        // Deccan Plateau (Black Soil)
        else if (lat > 15 && lat < 22 && lng > 72 && lng < 80) {
            return this.soilDatabase.black;
        }
        // Southern Peninsula (Red Soil)
        else if (lat > 8 && lat < 20 && lng > 76 && lng < 88) {
            return this.soilDatabase.red;
        }
        // Western Ghats (Laterite)
        else if (lat > 8 && lat < 20 && lng > 73 && lng < 77) {
            return this.soilDatabase.laterite;
        }
        // Rajasthan/Desert (Sandy)
        else if (lat > 24 && lat < 30 && lng > 69 && lng < 76) {
            return this.soilDatabase.sandy;
        }
        // Coastal Regions (Sandy)
        else if (lat < 15 || (lat > 8 && lng > 78 && lng < 82)) {
            return this.soilDatabase.sandy;
        }
        // Default to alluvial for most of India
        else {
            return this.soilDatabase.alluvial;
        }
    },
    
    // Detect soil from state name (for manual locations)
    detectSoilFromState: function(stateName) {
        // Map states to soil types (India-specific)
        const stateSoilMap = {
            // Red Soil States
            'Karnataka': 'red',
            'Tamil Nadu': 'red',
            'Andhra Pradesh': 'red',
            'Telangana': 'red',
            'Odisha': 'red',
            
            // Black Soil States
            'Maharashtra': 'black',
            'Gujarat': 'black',
            'Madhya Pradesh': 'black',
            
            // Alluvial Soil States
            'Uttar Pradesh': 'alluvial',
            'Punjab': 'alluvial',
            'Haryana': 'alluvial',
            'Bihar': 'alluvial',
            'West Bengal': 'alluvial',
            
            // Laterite Soil States
            'Kerala': 'laterite',
            'Goa': 'laterite',
            
            // Sandy Soil States
            'Rajasthan': 'sandy',
            
            // Clay Soil States
            'Assam': 'clay',
            
            // Loam Soil States
            'Himachal Pradesh': 'loam',
            'Uttarakhand': 'loam'
        };
        
        const soilType = stateSoilMap[stateName] || 'alluvial';
        return this.soilDatabase[soilType];
    },
    
    showSoilDetectionProgress: function() {
        document.getElementById('soilDetectionProgress').style.display = 'block';
        this.updateSoilProgress(10);
    },
    
    updateSoilProgress: function(percent) {
        const progressBar = document.getElementById('soilProgressBar');
        const percentText = document.getElementById('soilProgressPercent');
        
        progressBar.style.width = percent + '%';
        percentText.textContent = percent + '%';
        
        if (percent >= 100) {
            setTimeout(() => {
                document.getElementById('soilDetectionProgress').style.display = 'none';
            }, 500);
        }
    },
    
    displaySoilAnalysis: function(soilData) {
        this.soilData = soilData;
        
        // Show soil card
        document.getElementById('soilAnalysisCard').style.display = 'block';
        
        // Update soil type with color coding
        const soilTypeElement = document.getElementById('soilTypeValue');
        soilTypeElement.textContent = soilData.name;
        soilTypeElement.className = 'property-value soil-type-indicator soil-type-' + soilData.color;
        
        // Update other properties
        document.getElementById('soilPH').textContent = soilData.ph;
        document.getElementById('soilTexture').textContent = soilData.texture;
        document.getElementById('soilDrainage').textContent = soilData.drainage;
        document.getElementById('soilNutrients').textContent = soilData.nutrients;
        document.getElementById('soilWaterRetention').textContent = soilData.waterRetention;
        
        // Update improvement tips
        const improvementTips = document.getElementById('improvementTips');
        improvementTips.textContent = soilData.improvement;
        document.getElementById('soilImprovementTips').style.display = 'block';
        
        // Update recommended crops
        const cropsContainer = document.getElementById('recommendedCrops');
        cropsContainer.innerHTML = '';
        soilData.crops.forEach(crop => {
            const badge = document.createElement('span');
            badge.className = 'crop-badge';
            badge.textContent = crop;
            badge.title = `Grows well in ${soilData.name}`;
            cropsContainer.appendChild(badge);
        });
        document.getElementById('soilCropsSection').style.display = 'block';
        
        // Update soil source info
        const soilSourceElement = document.getElementById('soilSource');
        if (soilData.source === 'dataset') {
            soilSourceElement.innerHTML = `<i class="fas fa-database text-success"></i> Source: Agricultural Dataset`;
            soilSourceElement.className = 'soil-source dataset';
        } else if (soilData.dataset_used) {
            soilSourceElement.innerHTML = `<i class="fas fa-database text-success"></i> Source: Dataset + Fallback`;
            soilSourceElement.className = 'soil-source dataset';
        } else {
            soilSourceElement.innerHTML = `<i class="fas fa-map-marker-alt text-warning"></i> Source: Geographic Estimation`;
            soilSourceElement.className = 'soil-source fallback';
        }
        
        // Scroll to soil analysis
        document.getElementById('soilAnalysisCard').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });
    },
    
    getGPSLocation: function() {
        const status = document.getElementById('gpsStatus');
        const coordinates = document.getElementById('coordinates');
        
        status.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Detecting location...';
        coordinates.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin fa-2x text-warning mb-3"></i><p class="mb-1"><strong>Searching for GPS signal...</strong></p></div>';
        
        if (!navigator.geolocation) {
            status.innerHTML = '<span class="text-danger"><i class="fas fa-exclamation-triangle"></i> GPS not supported by your browser</span>';
            return;
        }
        
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;
                const accuracy = position.coords.accuracy;
                
                // Update status
                status.innerHTML = `<span class="text-success"><i class="fas fa-check-circle"></i> Location detected!</span>`;
                
                // Update coordinates display
                coordinates.innerHTML = `
                    <div class="row">
                        <div class="col-6">
                            <div class="text-center">
                                <i class="fas fa-latitude fa-lg text-primary mb-2"></i>
                                <p class="mb-0"><strong>Latitude</strong></p>
                                <p class="text-muted">${lat.toFixed(6)}°</p>
                            </div>
                        </div>
                        <div class="col-6">
                            <div class="text-center">
                                <i class="fas fa-longitude fa-lg text-primary mb-2"></i>
                                <p class="mb-0"><strong>Longitude</strong></p>
                                <p class="text-muted">${lng.toFixed(6)}°</p>
                            </div>
                        </div>
                    </div>
                    <div class="text-center mt-2">
                        <small class="text-muted">Accuracy: ±${Math.round(accuracy)} meters</small>
                    </div>
                `;
                
                // Update accuracy dot
                const accuracyPercent = Math.min(100, Math.max(0, (100 - accuracy/2)));
                document.getElementById('accuracyDot').style.left = `${accuracyPercent}%`;
                
                // Show map
                document.getElementById('mapContainer').style.display = 'block';
                this.map.setView([lat, lng], 15);
                
                // Clear existing marker
                if (this.marker) {
                    this.map.removeLayer(this.marker);
                }
                
                // Add new marker
                this.marker = L.marker([lat, lng]).addTo(this.map)
                    .bindPopup('Your Farm Location<br>Lat: ' + lat.toFixed(6) + '<br>Lng: ' + lng.toFixed(6))
                    .openPopup();
                
                // Get address from coordinates (reverse geocoding)
                this.getAddressFromCoordinates(lat, lng);
                
                // Store current location
                this.currentLocation = {
                    latitude: lat,
                    longitude: lng,
                    accuracy: accuracy,
                    type: 'gps'
                };
                
                // Detect soil type FROM DATASET
                this.detectSoilTypeFromDataset(lat, lng);
            },
            (error) => {
                let errorMessage = 'Unable to retrieve location';
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        errorMessage = 'Location permission denied. Please allow location access.';
                        break;
                    case error.POSITION_UNAVAILABLE:
                        errorMessage = 'Location information unavailable.';
                        break;
                    case error.TIMEOUT:
                        errorMessage = 'Location request timed out.';
                        break;
                }
                
                status.innerHTML = `<span class="text-danger"><i class="fas fa-exclamation-triangle"></i> ${errorMessage}</span>`;
                coordinates.innerHTML = `<div class="text-center"><i class="fas fa-exclamation-circle fa-2x text-danger mb-3"></i><p class="mb-1"><strong>Location detection failed</strong></p><p class="text-muted">${errorMessage}</p></div>`;
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            }
        );
    },
    
    getAddressFromCoordinates: function(lat, lng) {
        // Using OpenStreetMap Nominatim for reverse geocoding
        const url = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&zoom=10`;
        
        fetch(url)
            .then(response => response.json())
            .then(data => {
                if (data && data.address) {
                    this.displayLocationDetails(data.address, lat, lng);
                }
            })
            .catch(error => {
                console.error('Reverse geocoding failed:', error);
                // Show basic location info anyway
                this.displayLocationDetails({}, lat, lng);
            });
    },
    
    displayLocationDetails: function(address, lat, lng) {
        const locationNameBox = document.getElementById('locationNameBox');
        const locationDetails = document.getElementById('locationDetails');
        
        let detailsHTML = '';
        
        if (address.village || address.town || address.city) {
            detailsHTML += `<p><i class="fas fa-home"></i> <strong>${address.village || address.town || address.city}</strong></p>`;
        }
        
        if (address.county || address.state) {
            detailsHTML += `<p><i class="fas fa-map"></i> ${address.county || ''} ${address.state || ''}</p>`;
        }
        
        if (address.country) {
            detailsHTML += `<p><i class="fas fa-globe"></i> ${address.country}</p>`;
        }
        
        // Store address info
        if (this.currentLocation) {
            this.currentLocation.address = address;
            this.currentLocation.displayName = address.village || address.town || address.city || 'Unknown Location';
        }
        
        locationDetails.innerHTML = detailsHTML;
        locationNameBox.style.display = 'block';
    },
    
    saveGPSLocation: function() {
        if (!this.currentLocation || !this.soilData) {
            alert('Please detect location first and wait for soil analysis to complete.');
            return;
        }
        
        // Create location data object with soil info
        const locationData = {
            ...this.currentLocation,
            soilType: this.soilData.name,
            soilProperties: {
                ph: this.soilData.ph,
                texture: this.soilData.texture,
                drainage: this.soilData.drainage,
                nutrients: this.soilData.nutrients,
                waterRetention: this.soilData.waterRetention
            },
            recommendedCrops: this.soilData.crops,
            soilSource: this.soilData.source || 'unknown',
            datasetUsed: this.soilData.dataset_used || false,
            timestamp: new Date().toISOString(),
            source: 'gps'
        };
        
        this.saveLocationToStorage(locationData);
    },
    
    saveManualLocation: function() {
        const state = document.getElementById('stateSelect').value;
        const district = document.getElementById('districtSelect').value;
        const village = document.getElementById('villageInput').value;
        const pincode = document.getElementById('pincodeInput').value;
        
        if (!state || !district) {
            alert('Please select state and district');
            return;
        }
        
        // For manual location, we'll use approximate coordinates based on state
        const coordinates = this.getApproximateCoordinates(state);
        
        // Show map
        document.getElementById('mapContainer').style.display = 'block';
        this.map.setView([coordinates.lat, coordinates.lng], 10);
        
        // Clear existing marker
        if (this.marker) {
            this.map.removeLayer(this.marker);
        }
        
        // Add new marker
        this.marker = L.marker([coordinates.lat, coordinates.lng]).addTo(this.map)
            .bindPopup('Manual Location<br>Approx Lat: ' + coordinates.lat.toFixed(4) + '<br>Approx Lng: ' + coordinates.lng.toFixed(4))
            .openPopup();
        
        // Display manual location details
        this.displayManualLocationDetails(state, district, village, coordinates);
        
        // Store current location
        this.currentLocation = {
            state: document.getElementById('stateSelect').selectedOptions[0].text,
            district: document.getElementById('districtSelect').selectedOptions[0].text,
            village: village,
            pincode: pincode,
            latitude: coordinates.lat,
            longitude: coordinates.lng,
            type: 'manual',
            displayName: village || document.getElementById('districtSelect').selectedOptions[0].text || document.getElementById('stateSelect').selectedOptions[0].text
        };
        
        // Detect soil type FROM DATASET with state info
        this.detectSoilTypeFromDataset(coordinates.lat, coordinates.lng, this.currentLocation.state, true);
    },
    
    displayManualLocationDetails: function(state, district, village, coordinates) {
        const locationNameBox = document.getElementById('locationNameBox');
        const locationDetails = document.getElementById('locationDetails');
        
        let detailsHTML = '';
        
        if (village) {
            detailsHTML += `<p><i class="fas fa-home"></i> <strong>${village}</strong> (Village)</p>`;
        }
        
        if (district) {
            detailsHTML += `<p><i class="fas fa-map-pin"></i> ${district} (District)</p>`;
        }
        
        if (state) {
            detailsHTML += `<p><i class="fas fa-map-marked-alt"></i> ${state} (State)</p>`;
        }
        
        locationDetails.innerHTML = detailsHTML;
        locationNameBox.style.display = 'block';
        
        // Update coordinates display
        const coordinatesDiv = document.getElementById('coordinates');
        coordinatesDiv.innerHTML = `
            <div class="row">
                <div class="col-6">
                    <div class="text-center">
                        <i class="fas fa-latitude fa-lg text-primary mb-2"></i>
                        <p class="mb-0"><strong>Approx Latitude</strong></p>
                        <p class="text-muted">${coordinates.lat.toFixed(4)}°</p>
                    </div>
                </div>
                <div class="col-6">
                    <div class="text-center">
                        <i class="fas fa-longitude fa-lg text-primary mb-2"></i>
                        <p class="mb-0"><strong>Approx Longitude</strong></p>
                        <p class="text-muted">${coordinates.lng.toFixed(4)}°</p>
                    </div>
                </div>
            </div>
            <div class="text-center mt-2">
                <small class="text-muted">Approximate location based on ${state}</small>
            </div>
        `;
    },
    
    getApproximateCoordinates: function(state) {
        // Approximate coordinates for Indian states
        const stateCoordinates = {
            'Maharashtra': { lat: 19.7515, lng: 75.7139 },
            'Uttar Pradesh': { lat: 26.8467, lng: 80.9462 },
            'Karnataka': { lat: 15.3173, lng: 75.7139 },
            'Gujarat': { lat: 22.2587, lng: 71.1924 },
            'Tamil Nadu': { lat: 11.1271, lng: 78.6569 },
            'Madhya Pradesh': { lat: 22.9734, lng: 78.6569 },
            'Rajasthan': { lat: 27.0238, lng: 74.2179 },
            'Punjab': { lat: 31.1471, lng: 75.3412 },
            'Haryana': { lat: 29.0588, lng: 76.0856 },
            'Bihar': { lat: 25.0961, lng: 85.3131 },
            'West Bengal': { lat: 22.9868, lng: 87.8550 },
            'Andhra Pradesh': { lat: 15.9129, lng: 79.7400 },
            'Telangana': { lat: 18.1124, lng: 79.0193 },
            'Kerala': { lat: 10.8505, lng: 76.2711 },
            'Odisha': { lat: 20.9517, lng: 85.0985 },
            'Goa': { lat: 15.2993, lng: 74.1240 },
            'Assam': { lat: 26.2006, lng: 92.9376 },
            'Jharkhand': { lat: 23.6102, lng: 85.2799 },
            'Chhattisgarh': { lat: 21.2787, lng: 81.8661 },
            'Uttarakhand': { lat: 30.0668, lng: 79.0193 },
            'Himachal Pradesh': { lat: 31.1048, lng: 77.1734 }
        };
        
        return stateCoordinates[state] || { lat: 20.5937, lng: 78.9629 }; // Default to India center
    },
    
    saveLocationToStorage: function(locationData) {
        // Get existing locations
        let savedLocations = JSON.parse(localStorage.getItem('annadatai_locations') || '[]');
        
        // Add new location
        savedLocations.unshift(locationData);
        
        // Keep only last 10 locations
        if (savedLocations.length > 10) {
            savedLocations = savedLocations.slice(0, 10);
        }
        
        // Save to localStorage
        localStorage.setItem('annadatai_locations', JSON.stringify(savedLocations));
        
        // Save to session for current use
        sessionStorage.setItem('current_location', JSON.stringify(locationData));
        
        // Show success message
        let sourceInfo = '';
        if (locationData.soilSource === 'dataset') {
            sourceInfo = ' (from agricultural dataset)';
        } else if (locationData.datasetUsed) {
            sourceInfo = ' (using dataset + estimation)';
        } else {
            sourceInfo = ' (using geographic estimation)';
        }
        
        alert(`✓ Location saved successfully!\n\n📍 ${locationData.displayName}\n🗺️ ${locationData.district || ''}, ${locationData.state || ''}\n🌱 Soil: ${locationData.soilType}${sourceInfo}\n🌾 Crops: ${locationData.recommendedCrops.slice(0, 3).join(', ')}...`);
        
        // Refresh saved locations list
        this.refreshSavedLocations();
    },
    
    loadSavedLocations: function() {
        this.refreshSavedLocations();
    },
    
    refreshSavedLocations: function() {
        const savedLocations = JSON.parse(localStorage.getItem('annadatai_locations') || '[]');
        const container = document.getElementById('savedLocationsList');
        
        if (savedLocations.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-map-marked-alt"></i>
                    <h4>No Locations Saved Yet</h4>
                    <p class="mb-4">Save your first location using GPS or manual entry above.</p>
                    <p class="text-muted">Saved locations will appear here. You can select them on the Crops page for recommendations.</p>
                </div>
            `;
            return;
        }
        
        let html = '';
        savedLocations.forEach((loc, index) => {
            const isCurrent = index === 0;
            const locationName = loc.displayName || loc.village || loc.district || 'Unknown Location';
            
            // Soil source badge
            let soilSourceBadge = '';
            if (loc.soilSource === 'dataset') {
                soilSourceBadge = '<span class="badge bg-success dataset-badge">Dataset</span>';
            } else if (loc.datasetUsed) {
                soilSourceBadge = '<span class="badge bg-info dataset-badge">Mixed</span>';
            } else {
                soilSourceBadge = '<span class="badge bg-warning dataset-badge">Estimate</span>';
            }
            
            html += `
                <div class="saved-location-item ${isCurrent ? 'current' : ''}">
                    <div class="row align-items-center">
                        <div class="col-md-8">
                            <h5 class="mb-1">
                                <i class="fas fa-map-marker-alt ${isCurrent ? 'text-success' : 'text-secondary'}"></i>
                                ${locationName}
                                ${isCurrent ? '<span class="badge bg-success location-badge ms-2">Current</span>' : ''}
                            </h5>
                            <p class="text-muted mb-1">
                                <i class="fas fa-map"></i> ${loc.district || 'District'}, ${loc.state || 'State'}
                                ${loc.village ? ` • ${loc.village}` : ''}
                            </p>
                            <p class="mb-0">
                                <span class="badge ${this.getSoilBadgeClass(loc.soilType)}">${loc.soilType || 'Soil: Unknown'}</span>
                                ${soilSourceBadge}
                                <span class="badge bg-info">${loc.source === 'gps' ? 'GPS' : 'Manual'}</span>
                            </p>
                        </div>
                        <div class="col-md-4 text-end">
                            <div class="location-actions">
                                <button class="btn btn-sm btn-outline-success" onclick="locationManager.useLocation(${index})">
                                    <i class="fas fa-check"></i> Use
                                </button>
                                <button class="btn btn-sm btn-outline-danger" onclick="locationManager.deleteLocation(${index})">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                            <small class="text-muted">
                                ${new Date(loc.timestamp).toLocaleDateString()}
                            </small>
                        </div>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    },
    
    getSoilBadgeClass: function(soilType) {
        const soilClasses = {
            'Red Soil': 'bg-danger',
            'Black Soil (Regur)': 'bg-dark',
            'Alluvial Soil': 'bg-primary',
            'Laterite Soil': 'bg-warning text-dark',
            'Sandy Soil': 'bg-secondary',
            'Clay Soil': 'bg-success',
            'Loam Soil': 'bg-info'
        };
        return soilClasses[soilType] || 'bg-info';
    },
    
    useLocation: function(index) {
        const savedLocations = JSON.parse(localStorage.getItem('annadatai_locations') || '[]');
        if (savedLocations[index]) {
            // Move to top
            const location = savedLocations.splice(index, 1)[0];
            savedLocations.unshift(location);
            localStorage.setItem('annadatai_locations', JSON.stringify(savedLocations));
            sessionStorage.setItem('current_location', JSON.stringify(location));
            
            let sourceInfo = '';
            if (location.soilSource === 'dataset') {
                sourceInfo = ' (from agricultural dataset)';
            } else if (location.datasetUsed) {
                sourceInfo = ' (using dataset + estimation)';
            } else {
                sourceInfo = ' (using geographic estimation)';
            }
            
            alert(`✓ Now using location:\n📍 ${location.displayName || location.village || location.district}\n🗺️ ${location.district || ''}, ${location.state || ''}\n🌱 Soil: ${location.soilType}${sourceInfo}`);
            
            this.refreshSavedLocations();
        }
    },
    
    deleteLocation: function(index) {
        if (confirm('Are you sure you want to delete this location?')) {
            const savedLocations = JSON.parse(localStorage.getItem('annadatai_locations') || '[]');
            savedLocations.splice(index, 1);
            localStorage.setItem('annadatai_locations', JSON.stringify(savedLocations));
            this.refreshSavedLocations();
        }
    },
    
    setupStateDistrictDropdowns: function() {
        // This would typically come from a backend API
        // For now, we'll create a simple state-district mapping
        const stateDistrictMap = {
            'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Nashik', 'Aurangabad'],
            'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Varanasi', 'Agra', 'Prayagraj'],
            'Karnataka': ['Bangalore', 'Mysore', 'Hubli', 'Mangalore', 'Belgaum'],
            'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Bhavnagar'],
            'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli', 'Salem'],
            'Madhya Pradesh': ['Bhopal', 'Indore', 'Jabalpur', 'Gwalior', 'Ujjain'],
            'Rajasthan': ['Jaipur', 'Jodhpur', 'Udaipur', 'Kota', 'Bikaner'],
            'Punjab': ['Chandigarh', 'Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala'],
            'Haryana': ['Chandigarh', 'Faridabad', 'Gurgaon', 'Panipat', 'Ambala'],
            'Bihar': ['Patna', 'Gaya', 'Bhagalpur', 'Muzaffarpur', 'Purnia'],
            'West Bengal': ['Kolkata', 'Howrah', 'Durgapur', 'Asansol', 'Siliguri'],
            'Andhra Pradesh': ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Nellore', 'Kurnool'],
            'Telangana': ['Hyderabad', 'Warangal', 'Nizamabad', 'Karimnagar', 'Khammam'],
            'Kerala': ['Thiruvananthapuram', 'Kochi', 'Kozhikode', 'Thrissur', 'Kollam'],
            'Odisha': ['Bhubaneswar', 'Cuttack', 'Rourkela', 'Brahmapur', 'Sambalpur'],
            'Goa': ['North Goa', 'South Goa']
        };
        
        const stateSelect = document.getElementById('stateSelect');
        const districtSelect = document.getElementById('districtSelect');
        
        stateSelect.addEventListener('change', function() {
            const selectedState = this.options[this.selectedIndex].text;
            const districts = stateDistrictMap[selectedState] || [];
            
            districtSelect.innerHTML = '<option value="">Select District</option>';
            districts.forEach(district => {
                const option = document.createElement('option');
                option.value = district.toLowerCase().replace(/\s+/g, '_');
                option.textContent = district;
                districtSelect.appendChild(option);
            });
        });
    }
};

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    locationManager.init();
});