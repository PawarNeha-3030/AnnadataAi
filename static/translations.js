// Translation system for FarmAI website
const translations = {
    en: {
        // Navigation
        nav_home: "🏠 Home",
        nav_location: "📍 Location", 
        nav_crops: "🌱 Crops",
        nav_water: "💧 Water",
        nav_diseases: "⚠️ Diseases",
        nav_calendar: "📅 Calendar",
        nav_profit_prediction: "📊 Profit Prediction",
        
        // Hero Section
        hero_title: "Smart Farming for a Sustainable Future",
        hero_description: "AnnadataAI uses artificial intelligence to empower farmers with personalized crop guidance, real-time disease detection, smart irrigation planning, and profit prediction. It helps farmers increase crop yield, reduce losses, and use resources efficiently. By combining modern AI with local farming knowledge and multilingual support, AnnadataAI makes advanced agricultural technology accessible to every farmer.",
        btn_get_started: "Get Started Free",
        
        // Stats
        stat_farmers: "Active Farmers",
        stat_yield: "Avg. Yield Increase", 
        stat_water: "Water Saved",
        stat_income: "Farmer Income Generated",
        
        // Features Section
        features_title: "Smart Features for Smarter Farming",
        features_subtitle: "Our platform combines AI with agricultural expertise to deliver actionable insights",
        
        feature_location_title: "Precision Location Mapping",
        feature_location_desc: "GPS-based farm mapping with soil analysis and regional climate adaptation.",
        feature_location_btn: "Try It",
        
        feature_ai_title: "AI Crop Advisor",
        feature_ai_desc: "Intelligent crop recommendations based on 50+ factors including soil, weather, and market trends.",
        feature_ai_btn: "Get Recommendations",
        
        feature_irrigation_title: "Smart Irrigation",
        feature_irrigation_desc: "Precision water management with automated scheduling and rainwater optimization.",
        feature_irrigation_btn: "Calculate Water",
        
        feature_disease_title: "Disease Protection",
        feature_disease_desc: "Early detection of 100+ crop diseases with AI image recognition and organic treatments.",
        feature_disease_btn: "Check Health",
        
        // How It Works
        how_it_works_title: "How FarmAI Works",
        how_it_works_subtitle: "Simple 4-step process to transform your farming",
        step1_title: "Set Location",
        step1_desc: "Enter your farm location via GPS or manual input",
        step2_title: "Get Analysis", 
        step2_desc: "AI analyzes soil, weather, and regional data",
        step3_title: "Receive Plan",
        step3_desc: "Get personalized crop and resource recommendations",
        step4_title: "Implement & Track",
        step4_desc: "Follow the plan and track progress with our app",
        
        // CTA Section
        cta_title: "Ready to Transform Your Farming?",
        cta_description: "Join thousands of farmers who have increased their yield and reduced costs with FarmAI. Start your 30-day free trial today.",
        cta_button: "Start Free Trial",
        cta_note: "No credit card required • Cancel anytime",
        
        // Footer
        footer_about: "Empowering farmers with artificial intelligence for sustainable and profitable agriculture.",
        footer_product: "Product",
        footer_company: "Company",
        footer_connect: "Connect",
        copyright: "© 2024 FarmAI Assistant. All rights reserved. | Hackathon Project",
        
        // Location Page
        location_title: "Set Your Location",
        location_subtitle: "Help us provide personalized farming recommendations for your area",
        location_gps_title: "📍 GPS Location",
        location_gps_desc: "Get your exact location automatically",
        location_gps_btn: "Use GPS Location",
        location_manual_title: "📝 Manual Entry",
        location_manual_desc: "Enter your location details manually",
        location_manual_btn: "Enter Manually",
        location_accuracy: "Location Accuracy",
        location_coordinates: "Coordinates",
        location_soil_type: "Soil Type",
        location_climate: "Climate Zone",
        location_continue: "Continue to Recommendations",
        
        // Crops Page
        crops_title: "AI Crop Recommendations",
        crops_subtitle: "Get personalized crop suggestions based on your farm location, soil type, and season.",
        crops_seasonal_tab: "📅 Seasonal Analysis",
        crops_recommendations_tab: "🚀 Crop Recommendations",
        crops_season: "Season",
        crops_soil: "Soil Type",
        crops_climate: "Climate",
        crops_recommended: "Recommended Crops",
        crops_profit: "Expected Profit",
        crops_duration: "Growth Duration",
        crops_difficulty: "Difficulty Level",
        crops_water_req: "Water Requirement",
        crops_select: "Select This Crop",
        
        // Water Page
        water_title: "Smart Water Management",
        water_subtitle: "Calculate precise water requirements, plan irrigation schedules, and optimize water usage for your crops with location-based insights.",
        water_calculator: "Location-Enhanced Water Calculator",
        location_water_analysis: "Location-Based Water Analysis",
        water_crop_type: "Crop Type",
        water_area: "Farm Area (acres)",
        water_season: "Season",
        water_calculate: "Calculate Water Needs",
        water_daily_req: "Daily Water Requirement",
        water_weekly_req: "Weekly Water Requirement",
        water_monthly_req: "Monthly Water Requirement",
        water_irrigation_schedule: "Irrigation Schedule",
        water_tips: "Water Conservation Tips",
        
        // Calendar Page
        calendar_title: "Farming Calendar",
        calendar_subtitle: "Plan your farming activities throughout the year",
        calendar_current_month: "Current Month Activities",
        calendar_upcoming: "Upcoming Tasks",
        calendar_planting: "Planting Season",
        calendar_harvesting: "Harvesting Season",
        calendar_maintenance: "Maintenance Tasks",
        calendar_weather: "Weather Forecast",
        
        // Profit Prediction Page
        profit_prediction_title: "Smart Crop Profit Prediction",
        profit_prediction_subtitle: "Predict crop profitability by analyzing growth duration, weather patterns, and market conditions to make informed farming decisions.",
        profit_prediction_crop_name: "Crop Name",
        profit_prediction_planting_month: "Planting Month",
        profit_prediction_farm_area: "Farm Area (acres)",
        profit_prediction_location: "Location/State",
        profit_prediction_analyze: "Predict Profit Potential",
        profit_prediction_analyzing: "Analyzing crop profitability...",
        
        // Diseases Page
        diseases_title: "AI Disease Detection",
        diseases_subtitle: "Upload images of affected plants to detect diseases early and get organic treatment recommendations.",
        diseases_scanner_title: "Plant Disease Scanner",
        diseases_upload: "Upload Plant Image",
        diseases_camera: "Take Photo",
        diseases_analyze: "Analyze Image",
        diseases_result: "Analysis Result",
        diseases_confidence: "Confidence Level",
        diseases_treatment: "Treatment Recommendations",
        diseases_prevention: "Prevention Tips",
        diseases_common: "Common Diseases"
    },
    
    hi: {
        // Navigation
        nav_home: "🏠 होम",
        nav_location: "📍 स्थान",
        nav_crops: "🌱 फसलें",
        nav_water: "💧 पानी",
        nav_diseases: "⚠️ रोग",
        nav_calendar: "📅 कैलेंडर",
        nav_profit_prediction: "📊 लाभ पूर्वानुमान",
        
        // Hero Section
        hero_title: "आधुनिक कृषि के लिए AI-संचालित स्मार्ट फार्मिंग",
        hero_description: "FarmAI व्यक्तिगत कृषि सिफारिशें प्रदान करने, फसल की पैदावार को अनुकूलित करने और संसाधन की बर्बादी को कम करने के लिए कृत्रिम बुद्धिमत्ता का उपयोग करता है। प्रौद्योगिकी के साथ कृषि में क्रांति लाने वाले हजारों किसानों से जुड़ें।",
        btn_get_started: "मुफ्त शुरुआत करें",
        
        // Stats
        stat_farmers: "सक्रिय किसान",
        stat_yield: "औसत उत्पादन वृद्धि",
        stat_water: "पानी की बचत",
        stat_income: "किसान आय उत्पन्न",
        
        // Features Section
        features_title: "स्मार्ट फार्मिंग के लिए स्मार्ट सुविधाएं",
        features_subtitle: "हमारा प्लेटफॉर्म कार्यात्मक अंतर्दृष्टि प्रदान करने के लिए AI को कृषि विशेषज्ञता के साथ जोड़ता है",
        
        feature_location_title: "सटीक स्थान मैपिंग",
        feature_location_desc: "मिट्टी विश्लेषण और क्षेत्रीय जलवायु अनुकूलन के साथ GPS-आधारित फार्म मैपिंग।",
        feature_location_btn: "इसे आज़माएं",
        
        feature_ai_title: "AI फसल सलाहकार",
        feature_ai_desc: "मिट्टी, मौसम और बाजार के रुझान सहित 50+ कारकों के आधार पर बुद्धिमान फसल सिफारिशें।",
        feature_ai_btn: "सिफारिशें प्राप्त करें",
        
        feature_irrigation_title: "स्मार्ट सिंचाई",
        feature_irrigation_desc: "स्वचालित शेड्यूलिंग और वर्षा जल अनुकूलन के साथ सटीक जल प्रबंधन।",
        feature_irrigation_btn: "पानी की गणना करें",
        
        feature_disease_title: "रोग सुरक्षा",
        feature_disease_desc: "AI छवि पहचान और जैविक उपचार के साथ 100+ फसल रोगों का प्रारंभिक पता लगाना।",
        feature_disease_btn: "स्वास्थ्य जांचें",
        
        // How It Works
        how_it_works_title: "FarmAI कैसे काम करता है",
        how_it_works_subtitle: "आपकी खेती को बदलने के लिए सरल 4-चरणीय प्रक्रिया",
        step1_title: "स्थान सेट करें",
        step1_desc: "GPS या मैन्युअल इनपुट के माध्यम से अपने खेत का स्थान दर्ज करें",
        step2_title: "विश्लेषण प्राप्त करें",
        step2_desc: "AI मिट्टी, मौसम और क्षेत्रीय डेटा का विश्लेषण करता है",
        step3_title: "योजना प्राप्त करें",
        step3_desc: "व्यक्तिगत फसल और संसाधन सिफारिशें प्राप्त करें",
        step4_title: "लागू करें और ट्रैक करें",
        step4_desc: "योजना का पालन करें और हमारे ऐप के साथ प्रगति को ट्रैक करें",
        
        // CTA Section
        cta_title: "अपनी खेती को बदलने के लिए तैयार हैं?",
        cta_description: "हजारों किसानों से जुड़ें जिन्होंने FarmAI के साथ अपनी उपज बढ़ाई है और लागत कम की है। आज ही अपना 30-दिन का मुफ्त परीक्षण शुरू करें।",
        cta_button: "मुफ्त परीक्षण शुरू करें",
        cta_note: "कोई क्रेडिट कार्ड आवश्यक नहीं • कभी भी रद्द करें",
        
        // Footer
        footer_about: "टिकाऊ और लाभदायक कृषि के लिए कृत्रिम बुद्धिमत्ता के साथ किसानों को सशक्त बनाना।",
        footer_product: "उत्पाद",
        footer_company: "कंपनी",
        footer_connect: "जुड़ें",
        copyright: "© 2024 FarmAI सहायक। सभी अधिकार सुरक्षित। | हैकाथॉन प्रोजेक्ट",
        
        // Location Page
        location_title: "अपना स्थान सेट करें",
        location_subtitle: "आपके क्षेत्र के लिए व्यक्तिगत कृषि सिफारिशें प्रदान करने में हमारी सहायता करें",
        location_gps_title: "📍 GPS स्थान",
        location_gps_desc: "अपना सटीक स्थान स्वचालित रूप से प्राप्त करें",
        location_gps_btn: "GPS स्थान का उपयोग करें",
        location_manual_title: "📝 मैन्युअल प्रविष्टि",
        location_manual_desc: "अपने स्थान का विवरण मैन्युअल रूप से दर्ज करें",
        location_manual_btn: "मैन्युअल रूप से दर्ज करें",
        location_accuracy: "स्थान सटीकता",
        location_coordinates: "निर्देशांक",
        location_soil_type: "मिट्टी का प्रकार",
        location_climate: "जलवायु क्षेत्र",
        location_continue: "सिफारिशों के लिए जारी रखें",
        
        // Crops Page
        crops_title: "AI फसल सिफारिशें",
        crops_subtitle: "अपने खेत के स्थान, मिट्टी के प्रकार और मौसम के आधार पर व्यक्तिगत फसल सुझाव प्राप्त करें।",
        crops_seasonal_tab: "📅 मौसमी विश्लेषण",
        crops_recommendations_tab: "🚀 फसल सिफारिशें",
        crops_season: "मौसम",
        crops_soil: "मिट्टी का प्रकार",
        crops_climate: "जलवायु",
        crops_recommended: "अनुशंसित फसलें",
        crops_profit: "अपेक्षित लाभ",
        crops_duration: "वृद्धि अवधि",
        crops_difficulty: "कठिनाई स्तर",
        crops_water_req: "पानी की आवश्यकता",
        crops_select: "इस फसल का चयन करें",
        
        // Water Page
        water_title: "स्मार्ट जल प्रबंधन",
        water_subtitle: "स्थान-आधारित अंतर्दृष्टि के साथ सटीक पानी की आवश्यकताओं की गणना करें, सिंचाई कार्यक्रम की योजना बनाएं, और अपनी फसलों के लिए पानी के उपयोग को अनुकूलित करें।",
        water_calculator: "स्थान-संवर्धित जल कैलकुलेटर",
        location_water_analysis: "स्थान-आधारित जल विश्लेषण",
        water_crop_type: "फसल का प्रकार",
        water_area: "खेत का क्षेत्रफल (एकड़)",
        water_season: "मौसम",
        water_calculate: "पानी की जरूरतों की गणना करें",
        water_daily_req: "दैनिक पानी की आवश्यकता",
        water_weekly_req: "साप्ताहिक पानी की आवश्यकता",
        water_monthly_req: "मासिक पानी की आवश्यकता",
        water_irrigation_schedule: "सिंचाई कार्यक्रम",
        water_tips: "पानी संरक्षण युक्तियाँ",
        
        // Calendar Page
        calendar_title: "कृषि कैलेंडर",
        calendar_subtitle: "साल भर अपनी कृषि गतिविधियों की योजना बनाएं",
        calendar_current_month: "वर्तमान माह की गतिविधियां",
        calendar_upcoming: "आगामी कार्य",
        calendar_planting: "बुआई का मौसम",
        calendar_harvesting: "कटाई का मौसम",
        calendar_maintenance: "रखरखाव कार्य",
        calendar_weather: "मौसम पूर्वानुमान",
        
        // Crop Analysis Page
        crop_analysis_title: "स्मार्ट फसल लाभ विश्लेषण",
        crop_analysis_subtitle: "सूचित कृषि निर्णय लेने के लिए वृद्धि अवधि, मौसम पैटर्न और बाजार स्थितियों पर विचार करके फसल लाभप्रदता का विश्लेषण करें।",
        crop_analysis_crop_name: "फसल का नाम",
        crop_analysis_planting_month: "बुआई का महीना",
        crop_analysis_farm_area: "खेत का क्षेत्रफल (एकड़)",
        crop_analysis_location: "स्थान/राज्य",
        crop_analysis_analyze: "लाभ क्षमता का विश्लेषण करें",
        crop_analysis_analyzing: "फसल लाभप्रदता का विश्लेषण कर रहे हैं...",
        
        // Diseases Page
        diseases_title: "AI रोग का पता लगाना",
        diseases_subtitle: "रोगों का जल्दी पता लगाने और जैविक उपचार की सिफारिशें प्राप्त करने के लिए प्रभावित पौधों की छवियां अपलोड करें।",
        diseases_scanner_title: "पौधे की बीमारी स्कैनर",
        diseases_upload: "पौधे की छवि अपलोड करें",
        diseases_camera: "फोटो लें",
        diseases_analyze: "छवि का विश्लेषण करें",
        diseases_result: "विश्लेषण परिणाम",
        diseases_confidence: "विश्वास स्तर",
        diseases_treatment: "उपचार सिफारिशें",
        diseases_prevention: "रोकथाम युक्तियाँ",
        diseases_common: "सामान्य रोग"
    },
    
    te: {
        // Navigation
        nav_home: "🏠 హోమ్",
        nav_location: "📍 స్థానం",
        nav_crops: "🌱 పంటలు",
        nav_water: "💧 నీరు",
        nav_diseases: "⚠️ వ్యాధులు",
        nav_calendar: "📅 క్యాలెండర్",
        nav_profit_prediction: "📊 లాభ అంచనా",
        
        // Hero Section
        hero_title: "ఆధునిక వ్యవసాయం కోసం AI-శక్తితో కూడిన స్మార్ట్ ఫార్మింగ్",
        hero_description: "FarmAI వ్యక్తిగత వ్యవసాయ సిఫార్సులను అందించడానికి, పంట దిగుబడిని అనుకూలీకరించడానికి మరియు వనరుల వ్యర్థాన్ని తగ్గించడానికి కృత్రిమ మేధస్సును ఉపయోగిస్తుంది। సాంకేతికతతో వ్యవసాయంలో విప్లవం చేస్తున్న వేలాది రైతులతో చేరండి।",
        btn_get_started: "ఉచితంగా ప్రారంభించండి",
        
        // Stats
        stat_farmers: "క్రియాశీల రైతులు",
        stat_yield: "సగటు దిగుబడి పెరుగుదల",
        stat_water: "నీరు ఆదా",
        stat_income: "రైతు ఆదాయం ఉత్పత్తి",
        
        // Features Section
        features_title: "స్మార్ట్ ఫార్మింగ్ కోసం స్మార్ట్ ఫీచర్లు",
        features_subtitle: "మా ప్లాట్‌ఫారమ్ చర్యాత్మక అంతర్దృష్టులను అందించడానికి AIని వ్యవసాయ నైపుణ్యంతో కలుపుతుంది",
        
        feature_location_title: "ఖచ్చితమైన స్థాన మ్యాపింగ్",
        feature_location_desc: "మట్టి విశ్లేషణ మరియు ప్రాంతీయ వాతావరణ అనుకూలనతో GPS-ఆధారిత వ్యవసాయ మ్యాపింగ్।",
        feature_location_btn: "దీన్ని ప్రయత్నించండి",
        
        feature_ai_title: "AI పంట సలహాదారు",
        feature_ai_desc: "మట్టి, వాతావరణం మరియు మార్కెట్ ట్రెండ్‌లతో సహా 50+ కారకాల ఆధారంగా తెలివైన పంట సిఫార్సులు।",
        feature_ai_btn: "సిఫార్సులు పొందండి",
        
        feature_irrigation_title: "స్మార్ట్ నీటిపారుదల",
        feature_irrigation_desc: "స్వయంచాలక షెడ్యూలింగ్ మరియు వర్షపు నీటి అనుకూలీకరణతో ఖచ్చితమైన నీటి నిర్వహణ।",
        feature_irrigation_btn: "నీటిని లెక్కించండి",
        
        feature_disease_title: "వ్యాధి రక్షణ",
        feature_disease_desc: "AI చిత్ర గుర్తింపు మరియు సేంద్రీయ చికిత్సలతో 100+ పంట వ్యాధుల ముందస్తు గుర్తింపు।",
        feature_disease_btn: "ఆరోగ్యాన్ని తనిఖీ చేయండి",
        
        // How It Works
        how_it_works_title: "FarmAI ఎలా పని చేస్తుంది",
        how_it_works_subtitle: "మీ వ్యవసాయాన్ని మార్చడానికి సరళమైన 4-దశల ప్రక్రిया",
        step1_title: "స్థానం సెట్ చేయండి",
        step1_desc: "GPS లేదా మాన్యువల్ ఇన్‌పుట్ ద్వారా మీ వ్యవసాయ స్థానాన్ని నమోదు చేయండి",
        step2_title: "విశ్లేషణ పొందండి",
        step2_desc: "AI మట్టి, వాతావరణం మరియు ప్రాంతీయ డేటాను విశ్లేషిస్తుంది",
        step3_title: "ప్రణాళిక పొందండి",
        step3_desc: "వ్యక్తిగత పంట మరియు వనరుల సిఫార్సులను పొందండి",
        step4_title: "అమలు చేయండి & ట్రాక్ చేయండి",
        step4_desc: "ప్రణాళికను అనుసరించండి మరియు మా యాప్‌తో పురోగతిని ట్రాక్ చేయండి",
        
        // CTA Section
        cta_title: "మీ వ్యవసాయాన్ని మార్చడానికి సిద్ధంగా ఉన్నారా?",
        cta_description: "FarmAI తో తమ దిగుబడిని పెంచుకున్న మరియు ఖర్చులను తగ్గించుకున్న వేలాది రైతులతో చేరండి. ఈరోజే మీ 30-రోజుల ఉచిత ట్రయల్‌ను ప్రారంభించండి।",
        cta_button: "ఉచిత ట్రయల్ ప్రారంభించండి",
        cta_note: "క్రెడిట్ కార్డ్ అవసరం లేదు • ఎప్పుడైనా రద్దు చేయండి",
        
        // Footer
        footer_about: "స్థిరమైన మరియు లాభదాయకమైన వ్యవసాయం కోసం కృత్రిమ మేధస్సుతో రైతులను శక్తివంతం చేయడం।",
        footer_product: "ఉత్పత్తి",
        footer_company: "కంపెనీ",
        footer_connect: "కనెక్ట్",
        copyright: "© 2024 FarmAI అసిస్టెంట్. అన్ని హక్కులు రక్షించబడ్డాయి. | హ్యాకథాన్ ప్రాజెక్ట్",
        
        // Location Page
        location_title: "మీ స్థానాన్ని సెట్ చేయండి",
        location_subtitle: "మీ ప్రాంతానికి వ్యక్తిగత వ్యవసాయ సిఫార్సులను అందించడంలో మాకు సహాయపడండి",
        location_gps_title: "📍 GPS స్థానం",
        location_gps_desc: "మీ ఖచ్చితమైన స్థానాన్ని స్వయంచాలకంగా పొందండి",
        location_gps_btn: "GPS స్థానాన్ని ఉపయోగించండి",
        location_manual_title: "📝 మాన్యువల్ ఎంట్రీ",
        location_manual_desc: "మీ స్థాన వివరాలను మాన్యువల్‌గా నమోదు చేయండి",
        location_manual_btn: "మాన్యువల్‌గా నమోదు చేయండి",
        location_accuracy: "స్థాన ఖచ్చితత్వం",
        location_coordinates: "కోఆర్డినేట్స్",
        location_soil_type: "మట్టి రకం",
        location_climate: "వాతావరణ మండలం",
        location_continue: "సిఫార్సుల కోసం కొనసాగించండి",
        
        // Crops Page
        crops_title: "AI పంట సిఫార్సులు",
        crops_subtitle: "మీ వ్యవసాయ స్థానం, మట్టి రకం మరియు సీజన్ ఆధారంగా వ్యక్తిగత పంట సూచనలను పొందండి।",
        crops_seasonal_tab: "📅 కాలానుగుణ విశ్లేషణ",
        crops_recommendations_tab: "🚀 పంట సిఫార్సులు",
        crops_season: "సీజన్",
        crops_soil: "మట్టి రకం",
        crops_climate: "వాతావరణం",
        crops_recommended: "సిఫార్సు చేయబడిన పంటలు",
        crops_profit: "ఆశించిన లాభం",
        crops_duration: "వృద్ధి వ్యవధి",
        crops_difficulty: "కష్టం స్థాయి",
        crops_water_req: "నీటి అవసరం",
        crops_select: "ఈ పంటను ఎంచుకోండి",
        
        // Water Page
        water_title: "నీటి నిర్వహణ",
        water_subtitle: "సరైన పంట వృద్ధి కోసం స్మార్ట్ నీటిపారుదల ప్రణాళిక",
        water_calculator: "నీటి కాలిక్యులేటర్",
        water_crop_type: "పంట రకం",
        water_area: "వ్యవసాయ ప్రాంతం (ఎకరాలు)",
        water_season: "సీజన్",
        water_calculate: "నీటి అవసరాలను లెక్కించండి",
        water_daily_req: "రోజువారీ నీటి అవసరం",
        water_weekly_req: "వారపు నీటి అవసరం",
        water_monthly_req: "నెలవారీ నీటి అవసరం",
        water_irrigation_schedule: "నీటిపారుదల షెడ్యూల్",
        water_tips: "నీటి పరిరక్షణ చిట్కాలు",
        
        // Calendar Page
        calendar_title: "వ్యవసాయ క్యాలెండర్",
        calendar_subtitle: "సంవత్సరం పొడవునా మీ వ్యవసాయ కార్యకలాపాలను ప్రణాళిక చేయండి",
        calendar_current_month: "ప్రస్తుత నెల కార్యకలాపాలు",
        calendar_upcoming: "రాబోయే పనులు",
        calendar_planting: "నాటడం సీజన్",
        calendar_harvesting: "కోత సీజన్",
        calendar_maintenance: "నిర్వహణ పనులు",
        calendar_weather: "వాతావరణ అంచనా",
        
        // Diseases Page
        diseases_title: "AI వ్యాధి గుర్తింపు",
        diseases_subtitle: "వ్యాధులను త్వరగా గుర్తించడానికి మరియు సేంద్రీయ చికిత్స సిఫార్సులను పొందడానికి ప్రభావిత మొక్కల చిత్రాలను అప్‌లోడ్ చేయండి।",
        diseases_scanner_title: "మొక్కల వ్యాధి స్కానర్",
        diseases_upload: "మొక్క చిత్రాన్ని అప్‌లోడ్ చేయండి",
        diseases_camera: "ఫోటో తీయండి",
        diseases_analyze: "చిత్రాన్ని విశ్లేషించండి",
        diseases_result: "విశ్లేషణ ఫలితం",
        diseases_confidence: "విశ్వాస స్థాయి",
        diseases_treatment: "చికిత్స సిఫార్సులు",
        diseases_prevention: "నివారణ చిట్కాలు",
        diseases_common: "సాధారణ వ్యాధులు"
    },
    
    ta: {
        // Navigation
        nav_home: "🏠 முகப்பு",
        nav_location: "📍 இடம்",
        nav_crops: "🌱 பயிர்கள்",
        nav_water: "💧 நீர்",
        nav_diseases: "⚠️ நோய்கள்",
        nav_calendar: "📅 நாட்காட்டி",
        nav_profit_prediction: "📊 லாப முன்னறிவிப்பு",
        
        // Hero Section
        hero_title: "நவீன விவசாயத்திற்கான AI-இயங்கும் ஸ்மார்ட் பண்ணை",
        hero_description: "FarmAI தனிப்பட்ட விவசாய பரிந்துரைகளை வழங்க, பயிர் விளைச்சலை மேம்படுத்த மற்றும் வள விரயத்தை குறைக்க செயற்கை நுண்ணறிவைப் பயன்படுத்துகிறது. தொழில்நுட்பத்துடன் விவசாயத்தில் புரட்சி செய்யும் ஆயிரக்கணக்கான விவசாயிகளுடன் சேரவும்.",
        btn_get_started: "இலவசமாக தொடங்கவும்",
        
        // Stats
        stat_farmers: "செயலில் உள்ள விவசாயிகள்",
        stat_yield: "சராசரி விளைச்சல் அதிகரிப்பு",
        stat_water: "நீர் சேமிப்பு",
        stat_income: "விவசாயி வருமானம் உருவாக்கம்",
        
        // Features Section
        features_title: "ஸ்மார்ட் பண்ணைக்கான ஸ்மார்ட் அம்சங்கள்",
        features_subtitle: "எங்கள் தளம் செயல்படக்கூடிய நுண்ணறிவுகளை வழங்க AI ஐ விவசாய நிபுணத்துவத்துடன் இணைக்கிறது",
        
        feature_location_title: "துல்லியமான இட வரைபடம்",
        feature_location_desc: "மண் பகுப்பாய்வு மற்றும் பிராந்திய காலநிலை தழுவலுடன் GPS-அடிப்படையிலான பண்ணை வரைபடம்.",
        feature_location_btn: "இதை முயற்சிக்கவும்",
        
        feature_ai_title: "AI பயிர் ஆலோசகர்",
        feature_ai_desc: "மண், வானிலை மற்றும் சந்தை போக்குகள் உட்பட 50+ காரணிகளின் அடிப்படையில் அறிவார்ந்த பயிர் பரிந்துரைகள்.",
        feature_ai_btn: "பரிந்துரைகளைப் பெறவும்",
        
        feature_irrigation_title: "ஸ்மார்ட் நீர்ப்பாசனம்",
        feature_irrigation_desc: "தானியங்கி திட்டமிடல் மற்றும் மழைநீர் மேம்படுத்தலுடன் துல்லியமான நீர் மேலாண்மை.",
        feature_irrigation_btn: "நீரைக் கணக்கிடவும்",
        
        feature_disease_title: "நோய் பாதுகாப்பு",
        feature_disease_desc: "AI படம் அங்கீகாரம் மற்றும் இயற்கை சிகிச்சைகளுடன் 100+ பயிர் நோய்களின் ஆரம்ப கண்டறிதல்.",
        feature_disease_btn: "ஆரோக்கியத்தை சரிபார்க்கவும்",
        
        // How It Works
        how_it_works_title: "FarmAI எப்படி வேலை செய்கிறது",
        how_it_works_subtitle: "உங்கள் விவசாயத்தை மாற்ற எளிய 4-படி செயல்முறை",
        step1_title: "இடத்தை அமைக்கவும்",
        step1_desc: "GPS அல்லது கைமுறை உள்ளீடு மூலம் உங்கள் பண்ணை இடத்தை உள்ளிடவும்",
        step2_title: "பகுப்பாய்வு பெறவும்",
        step2_desc: "AI மண், வானிலை மற்றும் பிராந்திய தரவை பகுப்பாய்வு செய்கிறது",
        step3_title: "திட்டத்தைப் பெறவும்",
        step3_desc: "தனிப்பட்ட பயிர் மற்றும் வள பரிந்துரைகளைப் பெறவும்",
        step4_title: "செயல்படுத்து & கண்காணிக்கவும்",
        step4_desc: "திட்டத்தைப் பின்பற்றி எங்கள் ஆப்ஸுடன் முன்னேற்றத்தைக் கண்காணிக்கவும்",
        
        // CTA Section
        cta_title: "உங்கள் விவசாயத்தை மாற்ற தயாரா?",
        cta_description: "FarmAI உடன் தங்கள் விளைச்சலை அதிகரித்து செலவுகளைக் குறைத்த ஆயிரக்கணக்கான விவசாயிகளுடன் சேரவும். இன்றே உங்கள் 30-நாள் இலவச சோதனையைத் தொடங்கவும்.",
        cta_button: "இலவச சோதனையைத் தொடங்கவும்",
        cta_note: "கிரெடிட் கார்டு தேவையில்லை • எப்போது வேண்டுமானாலும் ரத்து செய்யவும்",
        
        // Footer
        footer_about: "நிலையான மற்றும் லாபகரமான விவசாயத்திற்கான செயற்கை நுண்ணறிவுடன் விவசாயிகளை வலுப்படுத்துதல்.",
        footer_product: "தயாரிப்பு",
        footer_company: "நிறுவனம்",
        footer_connect: "இணைக்கவும்",
        copyright: "© 2024 FarmAI உதவியாளர். அனைத்து உரிமைகளும் பாதுகாக்கப்பட்டவை. | ஹேக்கத்தான் திட்டம்"
    },
    
    bn: {
        // Navigation
        nav_home: "🏠 হোম",
        nav_location: "📍 অবস্থান",
        nav_crops: "🌱 ফসল",
        nav_water: "💧 পানি",
        nav_diseases: "⚠️ রোগ",
        nav_calendar: "📅 ক্যালেন্ডার",
        nav_profit_prediction: "📊 লাভের পূর্বাভাস",
        
        // Hero Section
        hero_title: "আধুনিক কৃষির জন্য AI-চালিত স্মার্ট ফার্মিং",
        hero_description: "FarmAI ব্যক্তিগত কৃষি সুপারিশ প্রদান, ফসলের ফলন অপ্টিমাইজ এবং সম্পদের অপচয় কমাতে কৃত্রিম বুদ্ধিমত্তা ব্যবহার করে। প্রযুক্তির সাথে কৃষিতে বিপ্লব আনা হাজার হাজার কৃষকের সাথে যোগ দিন।",
        btn_get_started: "বিনামূল্যে শুরু করুন",
        
        // Stats
        stat_farmers: "সক্রিয় কৃষক",
        stat_yield: "গড় ফলন বৃদ্ধি",
        stat_water: "পানি সাশ্রয়",
        stat_income: "কৃষক আয় উৎপাদন",
        
        // Features Section
        features_title: "স্মার্ট ফার্মিংয়ের জন্য স্মার্ট বৈশিষ্ট্য",
        features_subtitle: "আমাদের প্ল্যাটফর্ম কার্যকর অন্তর্দৃষ্টি প্রদানের জন্য AI কে কৃষি দক্ষতার সাথে একত্রিত করে",
        
        feature_location_title: "নির্ভুল অবস্থান ম্যাপিং",
        feature_location_desc: "মাটি বিশ্লেষণ এবং আঞ্চলিক জলবায়ু অভিযোজন সহ GPS-ভিত্তিক খামার ম্যাপিং।",
        feature_location_btn: "এটি চেষ্টা করুন",
        
        feature_ai_title: "AI ফসল উপদেষ্টা",
        feature_ai_desc: "মাটি, আবহাওয়া এবং বাজারের প্রবণতা সহ ৫০+ কারণের উপর ভিত্তি করে বুদ্ধিমান ফসল সুপারিশ।",
        feature_ai_btn: "সুপারিশ পান",
        
        feature_irrigation_title: "স্মার্ট সেচ",
        feature_irrigation_desc: "স্বয়ংক্রিয় সময়সূচী এবং বৃষ্টির পানি অপ্টিমাইজেশন সহ নির্ভুল পানি ব্যবস্থাপনা।",
        feature_irrigation_btn: "পানি গণনা করুন",
        
        feature_disease_title: "রোগ সুরক্ষা",
        feature_disease_desc: "AI ছবি স্বীকৃতি এবং জৈব চিকিৎসা সহ ১০০+ ফসলের রোগের প্রাথমিক সনাক্তকরণ।",
        feature_disease_btn: "স্বাস্থ্য পরীক্ষা করুন",
        
        // How It Works
        how_it_works_title: "FarmAI কিভাবে কাজ করে",
        how_it_works_subtitle: "আপনার কৃষিকাজ রূপান্তরিত করার জন্য সহজ ৪-ধাপের প্রক্রিয়া",
        step1_title: "অবস্থান সেট করুন",
        step1_desc: "GPS বা ম্যানুয়াল ইনপুটের মাধ্যমে আপনার খামারের অবস্থান প্রবেশ করান",
        step2_title: "বিশ্লেষণ পান",
        step2_desc: "AI মাটি, আবহাওয়া এবং আঞ্চলিক ডেটা বিশ্লেষণ করে",
        step3_title: "পরিকল্পনা গ্রহণ করুন",
        step3_desc: "ব্যক্তিগত ফসল এবং সম্পদ সুপারিশ পান",
        step4_title: "বাস্তবায়ন ও ট্র্যাক করুন",
        step4_desc: "পরিকল্পনা অনুসরণ করুন এবং আমাদের অ্যাপের সাথে অগ্রগতি ট্র্যাক করুন",
        
        // CTA Section
        cta_title: "আপনার কৃষিকাজ রূপান্তরিত করতে প্রস্তুত?",
        cta_description: "FarmAI দিয়ে তাদের ফলন বৃদ্ধি এবং খরচ কমিয়েছেন এমন হাজার হাজার কৃষকের সাথে যোগ দিন। আজই আপনার ৩০-দিনের বিনামূল্যে ট্রায়াল শুরু করুন।",
        cta_button: "বিনামূল্যে ট্রায়াল শুরু করুন",
        cta_note: "কোন ক্রেডিট কার্ডের প্রয়োজন নেই • যেকোনো সময় বাতিল করুন",
        
        // Footer
        footer_about: "টেকসই এবং লাভজনক কৃষির জন্য কৃত্রিম বুদ্ধিমত্তার সাথে কৃষকদের ক্ষমতায়ন।",
        footer_product: "পণ্য",
        footer_company: "কোম্পানি",
        footer_connect: "সংযোগ",
        copyright: "© ২০২৪ FarmAI সহায়ক। সমস্ত অধিকার সংরক্ষিত। | হ্যাকাথন প্রকল্প"
    },
    
    mr: {
        // Navigation
        nav_home: "🏠 होम",
        nav_location: "📍 स्थान",
        nav_crops: "🌱 पिके",
        nav_water: "💧 पाणी",
        nav_diseases: "⚠️ रोग",
        nav_calendar: "📅 दिनदर्शिका",
        nav_profit_prediction: "📊 नफा अंदाज",
        
        // Hero Section
        hero_title: "आधुनिक शेतीसाठी AI-चालित स्मार्ट फार्मिंग",
        hero_description: "FarmAI वैयक्तिक शेती शिफारसी प्रदान करण्यासाठी, पीक उत्पादन अनुकूल करण्यासाठी आणि संसाधन अपव्यय कमी करण्यासाठी कृत्रिम बुद्धिमत्ता वापरते. तंत्रज्ञानासह शेतीमध्ये क्रांती घडवून आणणाऱ्या हजारो शेतकऱ्यांसोबत सामील व्हा.",
        btn_get_started: "मोफत सुरुवात करा",
        
        // Stats
        stat_farmers: "सक्रिय शेतकरी",
        stat_yield: "सरासरी उत्पादन वाढ",
        stat_water: "पाणी बचत",
        stat_income: "शेतकरी उत्पन्न निर्मिती",
        
        // Features Section
        features_title: "स्मार्ट फार्मिंगसाठी स्मार्ट वैशिष्ट्ये",
        features_subtitle: "आमचे प्लॅटफॉर्म कार्यात्मक अंतर्दृष्टी प्रदान करण्यासाठी AI ला कृषी तज्ञतेसह एकत्र करते",
        
        feature_location_title: "अचूक स्थान मॅपिंग",
        feature_location_desc: "मातीचे विश्लेषण आणि प्रादेशिक हवामान अनुकूलनासह GPS-आधारित शेत मॅपिंग.",
        feature_location_btn: "हे करून पहा",
        
        feature_ai_title: "AI पीक सल्लागार",
        feature_ai_desc: "माती, हवामान आणि बाजार ट्रेंडसह ५०+ घटकांवर आधारित बुद्धिमान पीक शिफारसी.",
        feature_ai_btn: "शिफारसी मिळवा",
        
        feature_irrigation_title: "स्मार्ट सिंचन",
        feature_irrigation_desc: "स्वयंचलित वेळापत्रक आणि पावसाचे पाणी अनुकूलनासह अचूक पाणी व्यवस्थापन.",
        feature_irrigation_btn: "पाणी मोजा",
        
        feature_disease_title: "रोग संरक्षण",
        feature_disease_desc: "AI प्रतिमा ओळख आणि सेंद्रिय उपचारांसह १००+ पीक रोगांची लवकर ओळख.",
        feature_disease_btn: "आरोग्य तपासा",
        
        // How It Works
        how_it_works_title: "FarmAI कसे काम करते",
        how_it_works_subtitle: "तुमची शेती बदलण्यासाठी सोपी ४-चरण प्रक्रिया",
        step1_title: "स्थान सेट करा",
        step1_desc: "GPS किंवा मॅन्युअल इनपुटद्वारे तुमच्या शेताचे स्थान प्रविष्ट करा",
        step2_title: "विश्लेषण मिळवा",
        step2_desc: "AI माती, हवामान आणि प्रादेशिक डेटाचे विश्लेषण करते",
        step3_title: "योजना प्राप्त करा",
        step3_desc: "वैयक्तिक पीक आणि संसाधन शिफारसी मिळवा",
        step4_title: "अंमलबजावणी आणि ट्रॅक करा",
        step4_desc: "योजनेचे पालन करा आणि आमच्या अॅपसह प्रगती ट्रॅक करा",
        
        // CTA Section
        cta_title: "तुमची शेती बदलण्यास तयार आहात?",
        cta_description: "FarmAI सह त्यांचे उत्पादन वाढवलेल्या आणि खर्च कमी केलेल्या हजारो शेतकऱ्यांसोबत सामील व्हा। आजच तुमची ३०-दिवसांची मोफत चाचणी सुरू करा।",
        cta_button: "मोफत चाचणी सुरू करा",
        cta_note: "कोणत्याही क्रेडिट कार्डची गरज नाही • कधीही रद्द करा",
        
        // Footer
        footer_about: "शाश्वत आणि फायदेशीर शेतीसाठी कृत्रिम बुद्धिमत्तेसह शेतकऱ्यांना सक्षम करणे।",
        footer_product: "उत्पादन",
        footer_company: "कंपनी",
        footer_connect: "जोडा",
        copyright: "© २०२४ FarmAI सहाय्यक। सर्व हक्क राखीव। | हॅकाथॉन प्रकल्प"
    },
    
    gu: {
        // Navigation
        nav_home: "🏠 હોમ",
        nav_location: "📍 સ્થાન",
        nav_crops: "🌱 પાક",
        nav_water: "💧 પાણી",
        nav_diseases: "⚠️ રોગ",
        nav_calendar: "📅 કૅલેન્ડર",
        nav_profit_prediction: "📊 નફાની આગાહી",
        
        // Hero Section
        hero_title: "આધુનિક ખેતી માટે AI-સંચાલિત સ્માર્ટ ફાર્મિંગ",
        hero_description: "FarmAI વ્યક્તિગત ખેતી ભલામણો પ્રદાન કરવા, પાકની ઉપજને અનુકૂલિત કરવા અને સંસાધનોનો બગાડ ઘટાડવા માટે કૃત્રિમ બુદ્ધિનો ઉપયોગ કરે છે. ટેકનોલોજી સાથે ખેતીમાં ક્રાંતિ લાવતા હજારો ખેડૂતો સાથે જોડાઓ.",
        btn_get_started: "મફત શરૂઆત કરો",
        
        // Stats
        stat_farmers: "સક્રિય ખેડૂતો",
        stat_yield: "સરેરાશ ઉપજ વધારો",
        stat_water: "પાણીની બચત",
        stat_income: "ખેડૂત આવક ઉત્પાદન",
        
        // Features Section
        features_title: "સ્માર્ટ ફાર્મિંગ માટે સ્માર્ટ વિશેષતાઓ",
        features_subtitle: "અમારું પ્લેટફોર્મ કાર્યાત્મક અંતર્દૃષ્ટિ પ્રદાન કરવા માટે AI ને કૃષિ નિપુણતા સાથે જોડે છે",
        
        feature_location_title: "ચોક્કસ સ્થાન મેપિંગ",
        feature_location_desc: "માટીનું વિશ્લેષણ અને પ્રાદેશિક આબોહવા અનુકૂલન સાથે GPS-આધારિત ખેત મેપિંગ.",
        feature_location_btn: "આ અજમાવો",
        
        feature_ai_title: "AI પાક સલાહકાર",
        feature_ai_desc: "માટી, હવામાન અને બજાર વલણો સહિત ૫૦+ પરિબળોના આધારે બુદ્ધિશાળી પાક ભલામણો.",
        feature_ai_btn: "ભલામણો મેળવો",
        
        feature_irrigation_title: "સ્માર્ટ સિંચાઈ",
        feature_irrigation_desc: "સ્વચાલિત સમયપત્રક અને વરસાદી પાણીના અનુકૂલન સાથે ચોક્કસ પાણી વ્યવસ્થાપન.",
        feature_irrigation_btn: "પાણીની ગણતરી કરો",
        
        feature_disease_title: "રોગ સંરક્ષણ",
        feature_disease_desc: "AI છબી ઓળખ અને કાર્બનિક સારવાર સાથે ૧૦૦+ પાક રોગોની પ્રારંભિક શોધ.",
        feature_disease_btn: "આરોગ્ય તપાસો",
        
        // How It Works
        how_it_works_title: "FarmAI કેવી રીતે કામ કરે છે",
        how_it_works_subtitle: "તમારી ખેતીને બદલવા માટે સરળ 4-પગલાની પ્રક્રિયા",
        step1_title: "સ્થાન સેટ કરો",
        step1_desc: "GPS અથવા મેન્યુઅલ ઇનપુટ દ્વારા તમારા ખેતરનું સ્થાન દાખલ કરો",
        step2_title: "વિશ્લેષણ મેળવો",
        step2_desc: "AI માટી, હવામાન અને પ્રાદેશિક ડેટાનું વિશ્લેષણ કરે છે",
        step3_title: "યોજના પ્રાપ્ત કરો",
        step3_desc: "વ્યક્તિગત પાક અને સંસાધન ભલામણો મેળવો",
        step4_title: "અમલીકરણ અને ટ્રેક કરો",
        step4_desc: "યોજનાને અનુસરો અને અમારી એપ સાથે પ્રગતિને ટ્રેક કરો",
        
        // CTA Section
        cta_title: "તમારી ખેતીને બદલવા માટે તૈયાર છો?",
        cta_description: "FarmAI સાથે તેમની ઉપજ વધારી અને ખર્ચ ઘટાડ્યો છે તેવા હજારો ખેડૂતો સાથે જોડાઓ. આજે જ તમારી 30-દિવસની મફત ટ્રાયલ શરૂ કરો.",
        cta_button: "મફત ટ્રાયલ શરૂ કરો",
        cta_note: "કોઈ ક્રેડિટ કાર્ડની જરૂર નથી • કોઈપણ સમયે રદ કરો",
        
        // Footer
        footer_about: "ટકાઉ અને લાભદાયક ખેતી માટે કૃત્રિમ બુદ્ધિ સાથે ખેડૂતોને સશક્ત બનાવવું.",
        footer_product: "ઉત્પાદન",
        footer_company: "કંપની",
        footer_connect: "જોડાઓ",
        copyright: "© 2024 FarmAI સહાયક. બધા અધિકારો સુરક્ષિત. | હેકાથોન પ્રોજેક્ટ",
        
        // Location Page
        location_title: "તમારું સ્થાન સેટ કરો",
        location_subtitle: "તમારા વિસ્તાર માટે વ્યક્તિગત ખેતી ભલામણો પ્રદાન કરવામાં અમારી સહાય કરો",
        location_gps_title: "📍 GPS સ્થાન",
        location_gps_desc: "તમારું ચોક્કસ સ્થાન આપોઆપ મેળવો",
        location_gps_btn: "GPS સ્થાનનો ઉપયોગ કરો",
        location_manual_title: "📝 મેન્યુઅલ એન્ટ્રી",
        location_manual_desc: "તમારા સ્થાનની વિગતો મેન્યુઅલી દાખલ કરો",
        location_manual_btn: "મેન્યુઅલી દાખલ કરો",
        location_accuracy: "સ્થાન ચોકસાઈ",
        location_coordinates: "કોઓર્ડિનેટ્સ",
        location_soil_type: "માટીનો પ્રકાર",
        location_climate: "આબોહવા ક્ષેત્ર",
        location_continue: "ભલામણો માટે આગળ વધો",
        
        // Crops Page
        crops_title: "AI પાક ભલામણો",
        crops_subtitle: "તમારા ખેતરના સ્થાન, માટીના પ્રકાર અને મોસમના આધારે વ્યક્તિગત પાક સૂચનો મેળવો.",
        crops_seasonal_tab: "📅 મોસમી વિશ્લેષણ",
        crops_recommendations_tab: "🚀 પાક ભલામણો",
        crops_season: "મોસમ",
        crops_soil: "માટીનો પ્રકાર",
        crops_climate: "આબોહવા",
        crops_recommended: "ભલામણ કરેલ પાક",
        crops_profit: "અપેક્ષિત નફો",
        crops_duration: "વૃદ્ધિ અવધિ",
        crops_difficulty: "મુશ્કેલી સ્તર",
        crops_water_req: "પાણીની જરૂરિયાત",
        crops_select: "આ પાક પસંદ કરો",
        
        // Water Page
        water_title: "પાણી વ્યવસ્થાપન",
        water_subtitle: "શ્રેષ્ઠ પાક વૃદ્ધિ માટે સ્માર્ટ સિંચાઈ આયોજન",
        water_calculator: "પાણી કેલ્ક્યુલેટર",
        water_crop_type: "પાકનો પ્રકાર",
        water_area: "ખેતનો વિસ્તાર (એકર)",
        water_season: "મોસમ",
        water_calculate: "પાણીની જરૂરિયાતોની ગણતરી કરો",
        water_daily_req: "દૈનિક પાણીની જરૂરિયાત",
        water_weekly_req: "સાપ્તાહિક પાણીની જરૂરિયાત",
        water_monthly_req: "માસિક પાણીની જરૂરિયાત",
        water_irrigation_schedule: "સિંચાઈ સમયપત્રક",
        water_tips: "પાણી બચાવવાની ટિપ્સ",
        
        // Calendar Page
        calendar_title: "ખેતી કેલેન્ડર",
        calendar_subtitle: "વર્ષભર તમારી ખેતીની પ્રવૃત્તિઓનું આયોજન કરો",
        calendar_current_month: "વર્તમાન મહિનાની પ્રવૃત્તિઓ",
        calendar_upcoming: "આગામી કાર્યો",
        calendar_planting: "વાવેતરનો મોસમ",
        calendar_harvesting: "કાપણીનો મોસમ",
        calendar_maintenance: "જાળવણીના કાર્યો",
        calendar_weather: "હવામાન આગાહી",
        
        // Diseases Page
        diseases_title: "AI રોગ શોધ",
        diseases_subtitle: "રોગોની વહેલી શોધ અને કાર્બનિક સારવારની ભલામણો મેળવવા માટે અસરગ્રસ્ત છોડની છબીઓ અપલોડ કરો.",
        diseases_scanner_title: "છોડના રોગ સ્કેનર",
        diseases_upload: "છોડની છબી અપલોડ કરો",
        diseases_camera: "ફોટો લો",
        diseases_analyze: "છબીનું વિશ્લેષણ કરો",
        diseases_result: "વિશ્લેષણ પરિણામ",
        diseases_confidence: "વિશ્વાસ સ્તર",
        diseases_treatment: "સારવાર ભલામણો",
        diseases_prevention: "નિવારણ ટિપ્સ",
        diseases_common: "સામાન્ય રોગો"
    },
    
    kn: {
        // Navigation
        nav_home: "🏠 ಮುಖ್ಯಪುಟ",
        nav_location: "📍 ಸ್ಥಳ",
        nav_crops: "🌱 ಬೆಳೆಗಳು",
        nav_water: "💧 ನೀರು",
        nav_diseases: "⚠️ ರೋಗಗಳು",
        nav_calendar: "📅 ಕ್ಯಾಲೆಂಡರ್",
        nav_profit_prediction: "📊 ಲಾಭ ಮುನ್ಸೂಚನೆ",
        
        // Hero Section
        hero_title: "ಆಧುನಿಕ ಕೃಷಿಗಾಗಿ AI-ಚಾಲಿತ ಸ್ಮಾರ್ಟ್ ಫಾರ್ಮಿಂಗ್",
        hero_description: "FarmAI ವೈಯಕ್ತಿಕ ಕೃಷಿ ಶಿಫಾರಸುಗಳನ್ನು ಒದಗಿಸಲು, ಬೆಳೆ ಇಳುವರಿಯನ್ನು ಅತ್ಯುತ್ತಮಗೊಳಿಸಲು ಮತ್ತು ಸಂಪನ್ಮೂಲ ವ್ಯರ್ಥವನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಕೃತ್ರಿಮ ಬುದ್ಧಿಮತ್ತೆಯನ್ನು ಬಳಸುತ್ತದೆ. ತಂತ್ರಜ್ಞಾನದೊಂದಿಗೆ ಕೃಷಿಯಲ್ಲಿ ಕ್ರಾಂತಿ ತರುತ್ತಿರುವ ಸಾವಿರಾರು ರೈತರೊಂದಿಗೆ ಸೇರಿಕೊಳ್ಳಿ.",
        btn_get_started: "ಉಚಿತವಾಗಿ ಪ್ರಾರಂಭಿಸಿ",
        
        // Stats
        stat_farmers: "ಸಕ್ರಿಯ ರೈತರು",
        stat_yield: "ಸರಾಸರಿ ಇಳುವರಿ ಹೆಚ್ಚಳ",
        stat_water: "ನೀರಿನ ಉಳಿತಾಯ",
        stat_income: "ರೈತ ಆದಾಯ ಉತ್ಪಾದನೆ",
        
        // Features Section
        features_title: "ಸ್ಮಾರ್ಟ್ ಫಾರ್ಮಿಂಗ್‌ಗಾಗಿ ಸ್ಮಾರ್ಟ್ ವೈಶಿಷ್ಟ್ಯಗಳು",
        features_subtitle: "ನಮ್ಮ ಪ್ಲಾಟ್‌ಫಾರ್ಮ್ ಕ್ರಿಯಾತ್ಮಕ ಒಳನೋಟಗಳನ್ನು ಒದಗಿಸಲು AI ಅನ್ನು ಕೃಷಿ ಪರಿಣತಿಯೊಂದಿಗೆ ಸಂಯೋಜಿಸುತ್ತದೆ",
        
        feature_location_title: "ನಿಖರವಾದ ಸ್ಥಳ ಮ್ಯಾಪಿಂಗ್",
        feature_location_desc: "ಮಣ್ಣಿನ ವಿಶ್ಲೇಷಣೆ ಮತ್ತು ಪ್ರಾದೇಶಿಕ ಹವಾಮಾನ ಅನುಕೂಲನೆಯೊಂದಿಗೆ GPS-ಆಧಾರಿತ ಫಾರ್ಮ್ ಮ್ಯಾಪಿಂಗ್.",
        feature_location_btn: "ಇದನ್ನು ಪ್ರಯತ್ನಿಸಿ",
        
        feature_ai_title: "AI ಬೆಳೆ ಸಲಹೆಗಾರ",
        feature_ai_desc: "ಮಣ್ಣು, ಹವಾಮಾನ ಮತ್ತು ಮಾರುಕಟ್ಟೆ ಪ್ರವೃತ್ತಿಗಳು ಸೇರಿದಂತೆ ೫೦+ ಅಂಶಗಳ ಆಧಾರದ ಮೇಲೆ ಬುದ್ಧಿವಂತ ಬೆಳೆ ಶಿಫಾರಸುಗಳು.",
        feature_ai_btn: "ಶಿಫಾರಸುಗಳನ್ನು ಪಡೆಯಿರಿ",
        
        feature_irrigation_title: "ಸ್ಮಾರ್ಟ್ ನೀರಾವರಿ",
        feature_irrigation_desc: "ಸ್ವಯಂಚಾಲಿತ ವೇಳಾಪಟ್ಟಿ ಮತ್ತು ಮಳೆನೀರಿನ ಅತ್ಯುತ್ತಮೀಕರಣದೊಂದಿಗೆ ನಿಖರವಾದ ನೀರಿನ ನಿರ್ವಹಣೆ.",
        feature_irrigation_btn: "ನೀರನ್ನು ಲೆಕ್ಕಾಚಾರ ಮಾಡಿ",
        
        feature_disease_title: "ರೋಗ ರಕ್ಷಣೆ",
        feature_disease_desc: "AI ಚಿತ್ರ ಗುರುತಿಸುವಿಕೆ ಮತ್ತು ಸಾವಯವ ಚಿಕಿತ್ಸೆಗಳೊಂದಿಗೆ ೧೦೦+ ಬೆಳೆ ರೋಗಗಳ ಆರಂಭಿಕ ಪತ್ತೆ.",
        feature_disease_btn: "ಆರೋಗ್ಯವನ್ನು ಪರಿಶೀಲಿಸಿ",
        
        // How It Works
        how_it_works_title: "FarmAI ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ",
        how_it_works_subtitle: "ನಿಮ್ಮ ಕೃಷಿಯನ್ನು ಪರಿವರ್ತಿಸಲು ಸರಳ 4-ಹಂತದ ಪ್ರಕ್ರಿಯೆ",
        step1_title: "ಸ್ಥಳವನ್ನು ಹೊಂದಿಸಿ",
        step1_desc: "GPS ಅಥವಾ ಮ್ಯಾನುಯಲ್ ಇನ್‌ಪುಟ್ ಮೂಲಕ ನಿಮ್ಮ ಫಾರ್ಮ್ ಸ್ಥಳವನ್ನು ನಮೂದಿಸಿ",
        step2_title: "ವಿಶ್ಲೇಷಣೆ ಪಡೆಯಿರಿ",
        step2_desc: "AI ಮಣ್ಣು, ಹವಾಮಾನ ಮತ್ತು ಪ್ರಾದೇಶಿಕ ಡೇಟಾವನ್ನು ವಿಶ್ಲೇಷಿಸುತ್ತದೆ",
        step3_title: "ಯೋಜನೆ ಸ್ವೀಕರಿಸಿ",
        step3_desc: "ವೈಯಕ್ತಿಕ ಬೆಳೆ ಮತ್ತು ಸಂಪನ್ಮೂಲ ಶಿಫಾರಸುಗಳನ್ನು ಪಡೆಯಿರಿ",
        step4_title: "ಅನುಷ್ಠಾನ ಮತ್ತು ಟ್ರ್ಯಾಕ್ ಮಾಡಿ",
        step4_desc: "ಯೋಜನೆಯನ್ನು ಅನುಸರಿಸಿ ಮತ್ತು ನಮ್ಮ ಅಪ್ಲಿಕೇಶನ್‌ನೊಂದಿಗೆ ಪ್ರಗತಿಯನ್ನು ಟ್ರ್ಯಾಕ್ ಮಾಡಿ",
        
        // CTA Section
        cta_title: "ನಿಮ್ಮ ಕೃಷಿಯನ್ನು ಪರಿವರ್ತಿಸಲು ಸಿದ್ಧರಿದ್ದೀರಾ?",
        cta_description: "FarmAI ಯೊಂದಿಗೆ ತಮ್ಮ ಇಳುವರಿಯನ್ನು ಹೆಚ್ಚಿಸಿದ ಮತ್ತು ವೆಚ್ಚಗಳನ್ನು ಕಡಿಮೆ ಮಾಡಿದ ಸಾವಿರಾರು ರೈತರೊಂದಿಗೆ ಸೇರಿಕೊಳ್ಳಿ. ಇಂದೇ ನಿಮ್ಮ 30-ದಿನಗಳ ಉಚಿತ ಪ್ರಯೋಗವನ್ನು ಪ್ರಾರಂಭಿಸಿ.",
        cta_button: "ಉಚಿತ ಪ್ರಯೋಗವನ್ನು ಪ್ರಾರಂಭಿಸಿ",
        cta_note: "ಯಾವುದೇ ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್ ಅಗತ್ಯವಿಲ್ಲ • ಯಾವಾಗ ಬೇಕಾದರೂ ರದ್ದುಗೊಳಿಸಿ",
        
        // Footer
        footer_about: "ಸುಸ್ಥಿರ ಮತ್ತು ಲಾಭದಾಯಕ ಕೃಷಿಗಾಗಿ ಕೃತ್ರಿಮ ಬುದ್ಧಿಮತ್ತೆಯೊಂದಿಗೆ ರೈತರನ್ನು ಸಶಕ್ತಗೊಳಿಸುವುದು.",
        footer_product: "ಉತ್ಪನ್ನ",
        footer_company: "ಕಂಪನಿ",
        footer_connect: "ಸಂಪರ್ಕಿಸಿ",
        copyright: "© 2024 FarmAI ಸಹಾಯಕ. ಎಲ್ಲಾ ಹಕ್ಕುಗಳನ್ನು ಕಾಯ್ದಿರಿಸಲಾಗಿದೆ. | ಹ್ಯಾಕಥಾನ್ ಪ್ರಾಜೆಕ್ಟ್",
        
        // Location Page
        location_title: "ನಿಮ್ಮ ಸ್ಥಳವನ್ನು ಹೊಂದಿಸಿ",
        location_subtitle: "ನಿಮ್ಮ ಪ್ರದೇಶಕ್ಕೆ ವೈಯಕ್ತಿಕ ಕೃಷಿ ಶಿಫಾರಸುಗಳನ್ನು ಒದಗಿಸಲು ನಮಗೆ ಸಹಾಯ ಮಾಡಿ",
        location_gps_title: "📍 GPS ಸ್ಥಳ",
        location_gps_desc: "ನಿಮ್ಮ ನಿಖರವಾದ ಸ್ಥಳವನ್ನು ಸ್ವಯಂಚಾಲಿತವಾಗಿ ಪಡೆಯಿರಿ",
        location_gps_btn: "GPS ಸ್ಥಳವನ್ನು ಬಳಸಿ",
        location_manual_title: "📝 ಮ್ಯಾನುಯಲ್ ಎಂಟ್ರಿ",
        location_manual_desc: "ನಿಮ್ಮ ಸ್ಥಳದ ವಿವರಗಳನ್ನು ಹಸ್ತಚಾಲಿತವಾಗಿ ನಮೂದಿಸಿ",
        location_manual_btn: "ಹಸ್ತಚಾಲಿತವಾಗಿ ನಮೂದಿಸಿ",
        location_accuracy: "ಸ್ಥಳ ನಿಖರತೆ",
        location_coordinates: "ನಿರ್ದೇಶಾಂಕಗಳು",
        location_soil_type: "ಮಣ್ಣಿನ ಪ್ರಕಾರ",
        location_climate: "ಹವಾಮಾನ ವಲಯ",
        location_continue: "ಶಿಫಾರಸುಗಳಿಗೆ ಮುಂದುವರಿಯಿರಿ",
        
        // Crops Page
        crops_title: "AI ಬೆಳೆ ಶಿಫಾರಸುಗಳು",
        crops_subtitle: "ನಿಮ್ಮ ಫಾರ್ಮ್ ಸ್ಥಳ, ಮಣ್ಣಿನ ಪ್ರಕಾರ ಮತ್ತು ಋತುವಿನ ಆಧಾರದ ಮೇಲೆ ವೈಯಕ್ತಿಕ ಬೆಳೆ ಸಲಹೆಗಳನ್ನು ಪಡೆಯಿರಿ.",
        crops_seasonal_tab: "📅 ಋತುಮಾನದ ವಿಶ್ಲೇಷಣೆ",
        crops_recommendations_tab: "🚀 ಬೆಳೆ ಶಿಫಾರಸುಗಳು",
        crops_season: "ಋತು",
        crops_soil: "ಮಣ್ಣಿನ ಪ್ರಕಾರ",
        crops_climate: "ಹವಾಮಾನ",
        crops_recommended: "ಶಿಫಾರಸು ಮಾಡಿದ ಬೆಳೆಗಳು",
        crops_profit: "ನಿರೀಕ್ಷಿತ ಲಾಭ",
        crops_duration: "ಬೆಳವಣಿಗೆಯ ಅವಧಿ",
        crops_difficulty: "ಕಷ್ಟದ ಮಟ್ಟ",
        crops_water_req: "ನೀರಿನ ಅವಶ್ಯಕತೆ",
        crops_select: "ಈ ಬೆಳೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        
        // Water Page
        water_title: "ನೀರಿನ ನಿರ್ವಹಣೆ",
        water_subtitle: "ಅತ್ಯುತ್ತಮ ಬೆಳೆ ಬೆಳವಣಿಗೆಗಾಗಿ ಸ್ಮಾರ್ಟ್ ನೀರಾವರಿ ಯೋಜನೆ",
        water_calculator: "ನೀರಿನ ಕ್ಯಾಲ್ಕುಲೇಟರ್",
        water_crop_type: "ಬೆಳೆಯ ಪ್ರಕಾರ",
        water_area: "ಫಾರ್ಮ್ ಪ್ರದೇಶ (ಎಕರೆಗಳು)",
        water_season: "ಋತು",
        water_calculate: "ನೀರಿನ ಅಗತ್ಯಗಳನ್ನು ಲೆಕ್ಕಾಚಾರ ಮಾಡಿ",
        water_daily_req: "ದೈನಂದಿನ ನೀರಿನ ಅವಶ್ಯಕತೆ",
        water_weekly_req: "ಸಾಪ್ತಾಹಿಕ ನೀರಿನ ಅವಶ್ಯಕತೆ",
        water_monthly_req: "ಮಾಸಿಕ ನೀರಿನ ಅವಶ್ಯಕತೆ",
        water_irrigation_schedule: "ನೀರಾವರಿ ವೇಳಾಪಟ್ಟಿ",
        water_tips: "ನೀರಿನ ಸಂರಕ್ಷಣೆ ಸಲಹೆಗಳು",
        
        // Calendar Page
        calendar_title: "ಕೃಷಿ ಕ್ಯಾಲೆಂಡರ್",
        calendar_subtitle: "ವರ್ಷಪೂರ್ತಿ ನಿಮ್ಮ ಕೃಷಿ ಚಟುವಟಿಕೆಗಳನ್ನು ಯೋಜಿಸಿ",
        calendar_current_month: "ಪ್ರಸ್ತುತ ತಿಂಗಳ ಚಟುವಟಿಕೆಗಳು",
        calendar_upcoming: "ಮುಂಬರುವ ಕಾರ್ಯಗಳು",
        calendar_planting: "ನೆಟ್ಟ ಋತು",
        calendar_harvesting: "ಕೊಯ್ಲು ಋತು",
        calendar_maintenance: "ನಿರ್ವಹಣೆ ಕಾರ್ಯಗಳು",
        calendar_weather: "ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ",
        
        // Diseases Page
        diseases_title: "AI ರೋಗ ಪತ್ತೆ",
        diseases_subtitle: "ರೋಗಗಳನ್ನು ಬೇಗನೆ ಪತ್ತೆಹಚ್ಚಲು ಮತ್ತು ಸಾವಯವ ಚಿಕಿತ್ಸೆ ಶಿಫಾರಸುಗಳನ್ನು ಪಡೆಯಲು ಪೀಡಿತ ಸಸ್ಯಗಳ ಚಿತ್ರಗಳನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        diseases_scanner_title: "ಸಸ್ಯ ರೋಗ ಸ್ಕ್ಯಾನರ್",
        diseases_upload: "ಸಸ್ಯದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        diseases_camera: "ಫೋಟೋ ತೆಗೆಯಿರಿ",
        diseases_analyze: "ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
        diseases_result: "ವಿಶ್ಲೇಷಣೆ ಫಲಿತಾಂಶ",
        diseases_confidence: "ವಿಶ್ವಾಸ ಮಟ್ಟ",
        diseases_treatment: "ಚಿಕಿತ್ಸೆ ಶಿಫಾರಸುಗಳು",
        diseases_prevention: "ತಡೆಗಟ್ಟುವಿಕೆ ಸಲಹೆಗಳು",
        diseases_common: "ಸಾಮಾನ್ಯ ರೋಗಗಳು"
    },
    
    ml: {
        // Navigation
        nav_home: "🏠 ഹോം",
        nav_location: "📍 സ്ഥലം",
        nav_crops: "🌱 വിളകൾ",
        nav_water: "💧 വെള്ളം",
        nav_diseases: "⚠️ രോഗങ്ങൾ",
        nav_calendar: "📅 കലണ്ടർ",
        nav_profit_prediction: "📊 ലാഭ പ്രവചനം",
        
        // Hero Section
        hero_title: "ആധുനിക കൃഷിക്കായി AI-പവർഡ് സ്മാർട്ട് ഫാർമിംഗ്",
        hero_description: "FarmAI വ്യക്തിഗത കൃഷി ശുപാർശകൾ നൽകാനും വിള വിളവ് ഒപ്റ്റിമൈസ് ചെയ്യാനും വിഭവ പാഴാക്കൽ കുറയ്ക്കാനും കൃത്രിമ ബുദ്ധി ഉപയോഗിക്കുന്നു. സാങ്കേതികവിദ്യയുമായി കൃഷിയിൽ വിപ്ലവം സൃഷ്ടിക്കുന്ന ആയിരക്കണക്കിന് കർഷകരോടൊപ്പം ചേരുക.",
        btn_get_started: "സൗജന്യമായി ആരംഭിക്കുക",
        
        // Stats
        stat_farmers: "സജീവ കർഷകർ",
        stat_yield: "ശരാശരി വിള വർദ്ധനവ്",
        stat_water: "വെള്ളം ലാഭിച്ചു",
        stat_income: "കർഷക വരുമാന ഉൽപ്പാദനം",
        
        // Features Section
        features_title: "സ്മാർട്ട് ഫാർമിംഗിനായി സ്മാർട്ട് ഫീച്ചറുകൾ",
        features_subtitle: "പ്രവർത്തനക്ഷമമായ ഉൾക്കാഴ്ചകൾ നൽകാൻ ഞങ്ങളുടെ പ്ലാറ്റ്ഫോം AI-യെ കാർഷിക വൈദഗ്ധ്യവുമായി സംയോജിപ്പിക്കുന്നു",
        
        feature_location_title: "കൃത്യമായ ലൊക്കേഷൻ മാപ്പിംഗ്",
        feature_location_desc: "മണ്ണ് വിശകലനവും പ്രാദേശിക കാലാവസ്ഥാ അനുകൂലനവുമായി GPS-അടിസ്ഥാന ഫാം മാപ്പിംഗ്.",
        feature_location_btn: "ഇത് പരീക്ഷിക്കുക",
        
        feature_ai_title: "AI വിള ഉപദേഷ്ടാവ്",
        feature_ai_desc: "മണ്ണ്, കാലാവസ്ഥ, മാർക്കറ്റ് ട്രെൻഡുകൾ എന്നിവയുൾപ്പെടെ 50+ ഘടകങ്ങളെ അടിസ്ഥാനമാക്കിയുള്ള ബുദ്ധിപരമായ വിള ശുപാർശകൾ.",
        feature_ai_btn: "ശുപാർശകൾ നേടുക",
        
        feature_irrigation_title: "സ്മാർട്ട് ജലസേചനം",
        feature_irrigation_desc: "ഓട്ടോമാറ്റിക് ഷെഡ്യൂളിംഗും മഴവെള്ള ഒപ്റ്റിമൈസേഷനുമായി കൃത്യമായ ജല മാനേജ്മെന്റ്.",
        feature_irrigation_btn: "വെള്ളം കണക്കാക്കുക",
        
        feature_disease_title: "രോഗ സംരക്ഷണം",
        feature_disease_desc: "AI ഇമേജ് റെക്കഗ്നിഷനും ഓർഗാനിക് ട്രീറ്റ്മെന്റുകളുമായി 100+ വിള രോഗങ്ങളുടെ നേരത്തെയുള്ള കണ്ടെത്തൽ.",
        feature_disease_btn: "ആരോഗ്യം പരിശോധിക്കുക",
        
        // How It Works
        how_it_works_title: "FarmAI എങ്ങനെ പ്രവർത്തിക്കുന്നു",
        how_it_works_subtitle: "നിങ്ങളുടെ കൃഷി മാറ്റാൻ ലളിതമായ 4-ഘട്ട പ്രക്രിയ",
        step1_title: "സ്ഥലം സെറ്റ് ചെയ്യുക",
        step1_desc: "GPS അല്ലെങ്കിൽ മാനുവൽ ഇൻപുട്ട് വഴി നിങ്ങളുടെ ഫാം ലൊക്കേഷൻ നൽകുക",
        step2_title: "വിശകലനം നേടുക",
        step2_desc: "AI മണ്ണ്, കാലാവസ്ഥ, പ്രാദേശിക ഡാറ്റ വിശകലനം ചെയ്യുന്നു",
        step3_title: "പ്ലാൻ സ്വീകരിക്കുക",
        step3_desc: "വ്യക്തിഗത വിള, വിഭവ ശുപാർശകൾ നേടുക",
        step4_title: "നടപ്പിലാക്കുക & ട്രാക്ക് ചെയ്യുക",
        step4_desc: "പ്ലാൻ പിന്തുടരുക, ഞങ്ങളുടെ ആപ്പിൽ പുരോഗതി ട്രാക്ക് ചെയ്യുക",
        
        // CTA Section
        cta_title: "നിങ്ങളുടെ കൃഷി മാറ്റാൻ തയ്യാറാണോ?",
        cta_description: "FarmAI ഉപയോഗിച്ച് വിളവ് വർദ്ധിപ്പിക്കുകയും ചെലവ് കുറയ്ക്കുകയും ചെയ്ത ആയിരക്കണക്കിന് കർഷകരോടൊപ്പം ചേരുക. ഇന്നുതന്നെ നിങ്ങളുടെ 30-ദിവസത്തെ സൗജന്യ ട്രയൽ ആരംഭിക്കുക.",
        cta_button: "സൗജന്യ ട്രയൽ ആരംഭിക്കുക",
        cta_note: "ക്രെഡിറ്റ് കാർഡ് ആവശ്യമില്ല • എപ്പോൾ വേണമെങ്കിലും റദ്ദാക്കുക",
        
        // Footer
        footer_about: "സുസ്ഥിരവും ലാഭകരമായ കൃഷിക്കായി കൃത്രിമ ബുദ്ധിയുമായി കർഷകരെ ശാക്തീകരിക്കുന്നു.",
        footer_product: "ഉൽപ്പന്നം",
        footer_company: "കമ്പനി",
        footer_connect: "കണക്റ്റ്",
        copyright: "© 2024 FarmAI അസിസ്റ്റന്റ്. എല്ലാ അവകാശങ്ങളും സംരക്ഷിതം. | ഹാക്കത്തൺ പ്രോജക്റ്റ്",
        
        // Location Page
        location_title: "നിങ്ങളുടെ സ്ഥലം സെറ്റ് ചെയ്യുക",
        location_subtitle: "നിങ്ങളുടെ പ്രദേശത്തിനായി വ്യക്തിഗത കൃഷി ശുപാർശകൾ നൽകാൻ ഞങ്ങളെ സഹായിക്കുക",
        location_gps_title: "📍 GPS സ്ഥലം",
        location_gps_desc: "നിങ്ങളുടെ കൃത്യമായ സ്ഥലം സ്വയമേവ നേടുക",
        location_gps_btn: "GPS സ്ഥലം ഉപയോഗിക്കുക",
        location_manual_title: "📝 മാനുവൽ എൻട്രി",
        location_manual_desc: "നിങ്ങളുടെ സ്ഥല വിവരങ്ങൾ മാനുവലായി നൽകുക",
        location_manual_btn: "മാനുവലായി നൽകുക",
        location_accuracy: "സ്ഥല കൃത്യത",
        location_coordinates: "കോർഡിനേറ്റുകൾ",
        location_soil_type: "മണ്ണിന്റെ തരം",
        location_climate: "കാലാവസ്ഥാ മേഖല",
        location_continue: "ശുപാർശകൾക്കായി തുടരുക",
        
        // Crops Page
        crops_title: "AI വിള ശുപാർശകൾ",
        crops_subtitle: "നിങ്ങളുടെ ഫാം ലൊക്കേഷൻ, മണ്ണിന്റെ തരം, സീസൺ എന്നിവയെ അടിസ്ഥാനമാക്കി വ്യക്തിഗത വിള നിർദ്ദേശങ്ങൾ നേടുക.",
        crops_seasonal_tab: "📅 സീസണൽ വിശകലനം",
        crops_recommendations_tab: "🚀 വിള ശുപാർശകൾ",
        crops_season: "സീസൺ",
        crops_soil: "മണ്ണിന്റെ തരം",
        crops_climate: "കാലാവസ്ഥ",
        crops_recommended: "ശുപാർശ ചെയ്യുന്ന വിളകൾ",
        crops_profit: "പ്രതീക്ഷിക്കുന്ന ലാഭം",
        crops_duration: "വളർച്ചാ കാലയളവ്",
        crops_difficulty: "ബുദ്ധിമുട്ടിന്റെ തോത്",
        crops_water_req: "വെള്ളത്തിന്റെ ആവശ്യകത",
        crops_select: "ഈ വിള തിരഞ്ഞെടുക്കുക",
        
        // Water Page
        water_title: "ജല മാനേജ്മെന്റ്",
        water_subtitle: "ഒപ്റ്റിമൽ വിള വളർച്ചയ്ക്കായി സ്മാർട്ട് ജലസേചന പ്ലാനിംഗ്",
        water_calculator: "വാട്ടർ കാൽക്കുലേറ്റർ",
        water_crop_type: "വിളയുടെ തരം",
        water_area: "ഫാം ഏരിയ (ഏക്കർ)",
        water_season: "സീസൺ",
        water_calculate: "വെള്ളത്തിന്റെ ആവശ്യകതകൾ കണക്കാക്കുക",
        water_daily_req: "ദൈനംദിന വെള്ളത്തിന്റെ ആവശ്യകത",
        water_weekly_req: "പ്രതിവാര വെള്ളത്തിന്റെ ആവശ്യകത",
        water_monthly_req: "പ്രതിമാസ വെള്ളത്തിന്റെ ആവശ്യകത",
        water_irrigation_schedule: "ജലസേചന ഷെഡ്യൂൾ",
        water_tips: "ജല സംരക്ഷണ ടിപ്പുകൾ",
        
        // Calendar Page
        calendar_title: "കൃഷി കലണ്ടർ",
        calendar_subtitle: "വർഷം മുഴുവൻ നിങ്ങളുടെ കൃഷി പ്രവർത്തനങ്ങൾ ആസൂത്രണം ചെയ്യുക",
        calendar_current_month: "ഈ മാസത്തെ പ്രവർത്തനങ്ങൾ",
        calendar_upcoming: "വരാനിരിക്കുന്ന ജോലികൾ",
        calendar_planting: "നടീൽ സീസൺ",
        calendar_harvesting: "വിളവെടുപ്പ് സീസൺ",
        calendar_maintenance: "പരിപാലന ജോലികൾ",
        calendar_weather: "കാലാവസ്ഥാ പ്രവചനം",
        
        // Diseases Page
        diseases_title: "AI രോഗ കണ്ടെത്തൽ",
        diseases_subtitle: "രോഗങ്ങൾ നേരത്തെ കണ്ടെത്താനും ഓർഗാനിക് ചികിത്സാ ശുപാർശകൾ നേടാനും ബാധിത സസ്യങ്ങളുടെ ചിത്രങ്ങൾ അപ്‌ലോഡ് ചെയ്യുക.",
        diseases_scanner_title: "പ്ലാന്റ് ഡിസീസ് സ്കാനർ",
        diseases_upload: "പ്ലാന്റ് ഇമേജ് അപ്‌ലോഡ് ചെയ്യുക",
        diseases_camera: "ഫോട്ടോ എടുക്കുക",
        diseases_analyze: "ചിത്രം വിശകലനം ചെയ്യുക",
        diseases_result: "വിശകലന ഫലം",
        diseases_confidence: "വിശ്വാസ നില",
        diseases_treatment: "ചികിത്സാ ശുപാർശകൾ",
        diseases_prevention: "പ്രതിരോധ ടിപ്പുകൾ",
        diseases_common: "സാധാരണ രോഗങ്ങൾ"
    },
    
    pa: {
        // Navigation
        nav_home: "🏠 ਘਰ",
        nav_location: "📍 ਸਥਾਨ",
        nav_crops: "🌱 ਫਸਲਾਂ",
        nav_water: "💧 ਪਾਣੀ",
        nav_diseases: "⚠️ ਬਿਮਾਰੀਆਂ",
        nav_calendar: "📅 ਕੈਲੰਡਰ",
        nav_profit_prediction: "📊 ਲਾਭ ਪੂਰਵਾਨੁਮਾਨ",
        
        // Hero Section
        hero_title: "ਆਧੁਨਿਕ ਖੇਤੀ ਲਈ AI-ਸੰਚਾਲਿਤ ਸਮਾਰਟ ਫਾਰਮਿੰਗ",
        hero_description: "FarmAI ਵਿਅਕਤੀਗਤ ਖੇਤੀ ਸਿਫਾਰਸ਼ਾਂ ਪ੍ਰਦਾਨ ਕਰਨ, ਫਸਲ ਦੀ ਪੈਦਾਵਾਰ ਨੂੰ ਅਨੁਕੂਲ ਬਣਾਉਣ ਅਤੇ ਸਰੋਤਾਂ ਦੀ ਬਰਬਾਦੀ ਨੂੰ ਘਟਾਉਣ ਲਈ ਨਕਲੀ ਬੁੱਧੀ ਦੀ ਵਰਤੋਂ ਕਰਦਾ ਹੈ। ਤਕਨਾਲੋਜੀ ਨਾਲ ਖੇਤੀ ਵਿੱਚ ਕ੍ਰਾਂਤੀ ਲਿਆਉਣ ਵਾਲੇ ਹਜ਼ਾਰਾਂ ਕਿਸਾਨਾਂ ਨਾਲ ਜੁੜੋ।",
        btn_get_started: "ਮੁਫਤ ਸ਼ੁਰੂਆਤ ਕਰੋ",
        
        // Stats
        stat_farmers: "ਸਰਗਰਮ ਕਿਸਾਨ",
        stat_yield: "ਔਸਤ ਪੈਦਾਵਾਰ ਵਾਧਾ",
        stat_water: "ਪਾਣੀ ਦੀ ਬਚਤ",
        stat_income: "ਕਿਸਾਨ ਆਮਦਨ ਪੈਦਾਵਾਰ",
        
        // Features Section
        features_title: "ਸਮਾਰਟ ਫਾਰਮਿੰਗ ਲਈ ਸਮਾਰਟ ਵਿਸ਼ੇਸ਼ਤਾਵਾਂ",
        features_subtitle: "ਸਾਡਾ ਪਲੇਟਫਾਰਮ ਕਾਰਜਸ਼ੀਲ ਸੂਝ ਪ੍ਰਦਾਨ ਕਰਨ ਲਈ AI ਨੂੰ ਖੇਤੀ ਮਹਾਰਤ ਨਾਲ ਜੋੜਦਾ ਹੈ",
        
        feature_location_title: "ਸਟੀਕ ਸਥਾਨ ਮੈਪਿੰਗ",
        feature_location_desc: "ਮਿੱਟੀ ਵਿਸ਼ਲੇਸ਼ਣ ਅਤੇ ਖੇਤਰੀ ਜਲਵਾਯੂ ਅਨੁਕੂਲਨ ਨਾਲ GPS-ਅਧਾਰਿਤ ਫਾਰਮ ਮੈਪਿੰਗ।",
        feature_location_btn: "ਇਸ ਨੂੰ ਅਜ਼ਮਾਓ",
        
        feature_ai_title: "AI ਫਸਲ ਸਲਾਹਕਾਰ",
        feature_ai_desc: "ਮਿੱਟੀ, ਮੌਸਮ ਅਤੇ ਮਾਰਕੀਟ ਰੁਝਾਨਾਂ ਸਮੇਤ 50+ ਕਾਰਕਾਂ ਦੇ ਆਧਾਰ 'ਤੇ ਬੁੱਧੀਮਾਨ ਫਸਲ ਸਿਫਾਰਸ਼ਾਂ।",
        feature_ai_btn: "ਸਿਫਾਰਸ਼ਾਂ ਪ੍ਰਾਪਤ ਕਰੋ",
        
        feature_irrigation_title: "ਸਮਾਰਟ ਸਿੰਚਾਈ",
        feature_irrigation_desc: "ਆਟੋਮੈਟਿਕ ਸਮਾਂ-ਸਾਰਣੀ ਅਤੇ ਬਰਸਾਤੀ ਪਾਣੀ ਅਨੁਕੂਲਨ ਨਾਲ ਸਟੀਕ ਪਾਣੀ ਪ੍ਰਬੰਧਨ।",
        feature_irrigation_btn: "ਪਾਣੀ ਦੀ ਗਣਨਾ ਕਰੋ",
        
        feature_disease_title: "ਬਿਮਾਰੀ ਸੁਰੱਖਿਆ",
        feature_disease_desc: "AI ਚਿੱਤਰ ਪਛਾਣ ਅਤੇ ਜੈਵਿਕ ਇਲਾਜਾਂ ਨਾਲ 100+ ਫਸਲ ਬਿਮਾਰੀਆਂ ਦੀ ਸ਼ੁਰੂਆਤੀ ਖੋਜ।",
        feature_disease_btn: "ਸਿਹਤ ਦੀ ਜਾਂਚ ਕਰੋ",
        
        // How It Works
        how_it_works_title: "FarmAI ਕਿਵੇਂ ਕੰਮ ਕਰਦਾ ਹੈ",
        how_it_works_subtitle: "ਤੁਹਾਡੀ ਖੇਤੀ ਨੂੰ ਬਦਲਣ ਲਈ ਸਧਾਰਨ 4-ਪੜਾਅ ਪ੍ਰਕਿਰਿਆ",
        step1_title: "ਸਥਾਨ ਸੈੱਟ ਕਰੋ",
        step1_desc: "GPS ਜਾਂ ਮੈਨੁਅਲ ਇਨਪੁੱਟ ਰਾਹੀਂ ਆਪਣੇ ਖੇਤ ਦਾ ਸਥਾਨ ਦਾਖਲ ਕਰੋ",
        step2_title: "ਵਿਸ਼ਲੇਸ਼ਣ ਪ੍ਰਾਪਤ ਕਰੋ",
        step2_desc: "AI ਮਿੱਟੀ, ਮੌਸਮ ਅਤੇ ਖੇਤਰੀ ਡੇਟਾ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰਦਾ ਹੈ",
        step3_title: "ਯੋਜਨਾ ਪ੍ਰਾਪਤ ਕਰੋ",
        step3_desc: "ਵਿਅਕਤੀਗਤ ਫਸਲ ਅਤੇ ਸਰੋਤ ਸਿਫਾਰਸ਼ਾਂ ਪ੍ਰਾਪਤ ਕਰੋ",
        step4_title: "ਲਾਗੂ ਕਰੋ ਅਤੇ ਟਰੈਕ ਕਰੋ",
        step4_desc: "ਯੋਜਨਾ ਦਾ ਪਾਲਣ ਕਰੋ ਅਤੇ ਸਾਡੀ ਐਪ ਨਾਲ ਪ੍ਰਗਤੀ ਨੂੰ ਟਰੈਕ ਕਰੋ",
        
        // CTA Section
        cta_title: "ਆਪਣੀ ਖੇਤੀ ਨੂੰ ਬਦਲਣ ਲਈ ਤਿਆਰ ਹੋ?",
        cta_description: "FarmAI ਨਾਲ ਆਪਣੀ ਪੈਦਾਵਾਰ ਵਧਾਉਣ ਅਤੇ ਖਰਚੇ ਘਟਾਉਣ ਵਾਲੇ ਹਜ਼ਾਰਾਂ ਕਿਸਾਨਾਂ ਨਾਲ ਜੁੜੋ। ਅੱਜ ਹੀ ਆਪਣਾ 30-ਦਿਨਾਂ ਦਾ ਮੁਫਤ ਟਰਾਇਲ ਸ਼ੁਰੂ ਕਰੋ।",
        cta_button: "ਮੁਫਤ ਟਰਾਇਲ ਸ਼ੁਰੂ ਕਰੋ",
        cta_note: "ਕੋਈ ਕ੍ਰੈਡਿਟ ਕਾਰਡ ਦੀ ਲੋੜ ਨਹੀਂ • ਕਿਸੇ ਵੀ ਸਮੇਂ ਰੱਦ ਕਰੋ",
        
        // Footer
        footer_about: "ਟਿਕਾਊ ਅਤੇ ਲਾਭਕਾਰੀ ਖੇਤੀ ਲਈ ਨਕਲੀ ਬੁੱਧੀ ਨਾਲ ਕਿਸਾਨਾਂ ਨੂੰ ਸ਼ਕਤੀਸ਼ਾਲੀ ਬਣਾਉਣਾ।",
        footer_product: "ਉਤਪਾਦ",
        footer_company: "ਕੰਪਨੀ",
        footer_connect: "ਜੁੜੋ",
        copyright: "© 2024 FarmAI ਸਹਾਇਕ। ਸਾਰੇ ਅਧਿਕਾਰ ਸੁਰੱਖਿਅਤ। | ਹੈਕਾਥਨ ਪ੍ਰੋਜੈਕਟ",
        
        // Location Page
        location_title: "ਆਪਣਾ ਸਥਾਨ ਸੈੱਟ ਕਰੋ",
        location_subtitle: "ਆਪਣੇ ਖੇਤਰ ਲਈ ਵਿਅਕਤੀਗਤ ਖੇਤੀ ਸਿਫਾਰਸ਼ਾਂ ਪ੍ਰਦਾਨ ਕਰਨ ਵਿੱਚ ਸਾਡੀ ਮਦਦ ਕਰੋ",
        location_gps_title: "📍 GPS ਸਥਾਨ",
        location_gps_desc: "ਆਪਣਾ ਸਟੀਕ ਸਥਾਨ ਆਪਣੇ ਆਪ ਪ੍ਰਾਪਤ ਕਰੋ",
        location_gps_btn: "GPS ਸਥਾਨ ਵਰਤੋ",
        location_manual_title: "📝 ਮੈਨੁਅਲ ਐਂਟਰੀ",
        location_manual_desc: "ਆਪਣੇ ਸਥਾਨ ਦੀ ਜਾਣਕਾਰੀ ਮੈਨੁਅਲ ਰੂਪ ਵਿੱਚ ਦਾਖਲ ਕਰੋ",
        location_manual_btn: "ਮੈਨੁਅਲ ਰੂਪ ਵਿੱਚ ਦਾਖਲ ਕਰੋ",
        location_accuracy: "ਸਥਾਨ ਸਟੀਕਤਾ",
        location_coordinates: "ਕੋਆਰਡੀਨੇਟਸ",
        location_soil_type: "ਮਿੱਟੀ ਦੀ ਕਿਸਮ",
        location_climate: "ਜਲਵਾਯੂ ਖੇਤਰ",
        location_continue: "ਸਿਫਾਰਸ਼ਾਂ ਲਈ ਜਾਰੀ ਰੱਖੋ",
        
        // Crops Page
        crops_title: "AI ਫਸਲ ਸਿਫਾਰਸ਼ਾਂ",
        crops_subtitle: "ਆਪਣੇ ਫਾਰਮ ਦੇ ਸਥਾਨ, ਮਿੱਟੀ ਦੀ ਕਿਸਮ ਅਤੇ ਮੌਸਮ ਦੇ ਆਧਾਰ 'ਤੇ ਵਿਅਕਤੀਗਤ ਫਸਲ ਸੁਝਾਅ ਪ੍ਰਾਪਤ ਕਰੋ।",
        crops_seasonal_tab: "📅 ਮੌਸਮੀ ਵਿਸ਼ਲੇਸ਼ਣ",
        crops_recommendations_tab: "🚀 ਫਸਲ ਸਿਫਾਰਸ਼ਾਂ",
        crops_season: "ਮੌਸਮ",
        crops_soil: "ਮਿੱਟੀ ਦੀ ਕਿਸਮ",
        crops_climate: "ਜਲਵਾਯੂ",
        crops_recommended: "ਸਿਫਾਰਸ਼ ਕੀਤੀਆਂ ਫਸਲਾਂ",
        crops_profit: "ਉਮੀਦ ਕੀਤਾ ਮੁਨਾਫਾ",
        crops_duration: "ਵਿਕਾਸ ਦੀ ਮਿਆਦ",
        crops_difficulty: "ਮੁਸ਼ਕਿਲ ਦਾ ਪੱਧਰ",
        crops_water_req: "ਪਾਣੀ ਦੀ ਲੋੜ",
        crops_select: "ਇਸ ਫਸਲ ਨੂੰ ਚੁਣੋ",
        
        // Water Page
        water_title: "ਪਾਣੀ ਪ੍ਰਬੰਧਨ",
        water_subtitle: "ਸਰਵੋਤਮ ਫਸਲ ਵਿਕਾਸ ਲਈ ਸਮਾਰਟ ਸਿੰਚਾਈ ਯੋਜਨਾ",
        water_calculator: "ਪਾਣੀ ਕੈਲਕੁਲੇਟਰ",
        water_crop_type: "ਫਸਲ ਦੀ ਕਿਸਮ",
        water_area: "ਫਾਰਮ ਖੇਤਰ (ਏਕੜ)",
        water_season: "ਮੌਸਮ",
        water_calculate: "ਪਾਣੀ ਦੀਆਂ ਲੋੜਾਂ ਦੀ ਗਣਨਾ ਕਰੋ",
        water_daily_req: "ਰੋਜ਼ਾਨਾ ਪਾਣੀ ਦੀ ਲੋੜ",
        water_weekly_req: "ਹਫਤਾਵਾਰੀ ਪਾਣੀ ਦੀ ਲੋੜ",
        water_monthly_req: "ਮਹੀਨਾਵਾਰ ਪਾਣੀ ਦੀ ਲੋੜ",
        water_irrigation_schedule: "ਸਿੰਚਾਈ ਸਮਾਂ-ਸਾਰਣੀ",
        water_tips: "ਪਾਣੀ ਬਚਾਉਣ ਦੇ ਟਿਪਸ",
        
        // Calendar Page
        calendar_title: "ਖੇਤੀ ਕੈਲੰਡਰ",
        calendar_subtitle: "ਸਾਲ ਭਰ ਆਪਣੀਆਂ ਖੇਤੀ ਗਤੀਵਿਧੀਆਂ ਦੀ ਯੋਜਨਾ ਬਣਾਓ",
        calendar_current_month: "ਮੌਜੂਦਾ ਮਹੀਨੇ ਦੀਆਂ ਗਤੀਵਿਧੀਆਂ",
        calendar_upcoming: "ਆਉਣ ਵਾਲੇ ਕੰਮ",
        calendar_planting: "ਬੀਜਣ ਦਾ ਮੌਸਮ",
        calendar_harvesting: "ਵਾਢੀ ਦਾ ਮੌਸਮ",
        calendar_maintenance: "ਰੱਖ-ਰਖਾਅ ਦੇ ਕੰਮ",
        calendar_weather: "ਮੌਸਮ ਦੀ ਭਵਿੱਖਬਾਣੀ",
        
        // Diseases Page
        diseases_title: "AI ਬਿਮਾਰੀ ਖੋਜ",
        diseases_subtitle: "ਬਿਮਾਰੀਆਂ ਦੀ ਜਲਦੀ ਖੋਜ ਅਤੇ ਜੈਵਿਕ ਇਲਾਜ ਦੀਆਂ ਸਿਫਾਰਸ਼ਾਂ ਪ੍ਰਾਪਤ ਕਰਨ ਲਈ ਪ੍ਰਭਾਵਿਤ ਪੌਧਿਆਂ ਦੀਆਂ ਤਸਵੀਰਾਂ ਅਪਲੋਡ ਕਰੋ।",
        diseases_scanner_title: "ਪੌਧੇ ਦੀ ਬਿਮਾਰੀ ਸਕੈਨਰ",
        diseases_upload: "ਪੌਧੇ ਦੀ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ",
        diseases_camera: "ਫੋਟੋ ਲਓ",
        diseases_analyze: "ਤਸਵੀਰ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰੋ",
        diseases_result: "ਵਿਸ਼ਲੇਸ਼ਣ ਨਤੀਜਾ",
        diseases_confidence: "ਭਰੋਸੇ ਦਾ ਪੱਧਰ",
        diseases_treatment: "ਇਲਾਜ ਦੀਆਂ ਸਿਫਾਰਸ਼ਾਂ",
        diseases_prevention: "ਰੋਕਥਾਮ ਦੇ ਟਿਪਸ",
        diseases_common: "ਆਮ ਬਿਮਾਰੀਆਂ"
    }
};

// Language management functions
let currentLanguage = localStorage.getItem('selectedLanguage') || 'en';

// Initialize language on page load
document.addEventListener('DOMContentLoaded', function() {
    setLanguage(currentLanguage);
    updateLanguageButton(currentLanguage);
});

// Toggle language menu visibility
function toggleLanguageMenu() {
    const menu = document.getElementById('languageMenu');
    menu.classList.toggle('active');
    
    // Close menu when clicking outside
    document.addEventListener('click', function closeMenu(e) {
        if (!e.target.closest('.language-selector')) {
            menu.classList.remove('active');
            document.removeEventListener('click', closeMenu);
        }
    });
}

// Change language function
function changeLanguage(langCode) {
    currentLanguage = langCode;
    localStorage.setItem('selectedLanguage', langCode);
    setLanguage(langCode);
    updateLanguageButton(langCode);
    
    // Close the language menu
    document.getElementById('languageMenu').classList.remove('active');
    
    // Show success message
    showLanguageChangeMessage(langCode);
}

// Apply translations to the page
function setLanguage(langCode) {
    const translation = translations[langCode] || translations['en'];
    
    // Update all elements with data-translate attribute
    document.querySelectorAll('[data-translate]').forEach(element => {
        const key = element.getAttribute('data-translate');
        if (translation[key]) {
            // Handle different element types
            if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
                element.placeholder = translation[key];
            } else if (element.tagName === 'IMG') {
                element.alt = translation[key];
            } else {
                element.innerHTML = translation[key];
            }
        }
    });
    
    // Update page title
    if (translation.page_title) {
        document.title = translation.page_title;
    }
    
    // Update HTML lang attribute
    document.documentElement.lang = langCode;
}

// Update language button display
function updateLanguageButton(langCode) {
    const languageNames = {
        'en': 'EN',
        'hi': 'हि',
        'te': 'తె',
        'ta': 'த',
        'bn': 'বা',
        'mr': 'म',
        'gu': 'ગુ',
        'kn': 'ಕ',
        'ml': 'മ',
        'pa': 'ਪੰ'
    };
    
    const currentLangElement = document.getElementById('current-language');
    if (currentLangElement) {
        currentLangElement.textContent = languageNames[langCode] || 'EN';
    }
}

// Show language change confirmation
function showLanguageChangeMessage(langCode) {
    const messages = {
        'en': 'Language changed to English',
        'hi': 'भाषा हिंदी में बदली गई',
        'te': 'భాష తెలుగులోకి మార్చబడింది',
        'ta': 'மொழி தமிழுக்கு மாற்றப்பட்டது',
        'bn': 'ভাষা বাংলায় পরিবর্তিত হয়েছে',
        'mr': 'भाषा मराठीत बदलली',
        'gu': 'ભાષા ગુજરાતીમાં બદલાઈ',
        'kn': 'ಭಾಷೆ ಕನ್ನಡಕ್ಕೆ ಬದಲಾಯಿಸಲಾಗಿದೆ',
        'ml': 'ഭാഷ മലയാളത്തിലേക്ക് മാറ്റി',
        'pa': 'ਭਾਸ਼ਾ ਪੰਜਾਬੀ ਵਿੱਚ ਬਦਲੀ ਗਈ'
    };
    
    // Create and show notification
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: var(--primary);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        z-index: 10000;
        font-weight: 500;
        transform: translateX(100%);
        transition: transform 0.3s ease;
    `;
    notification.textContent = messages[langCode] || messages['en'];
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Export functions for global use
window.toggleLanguageMenu = toggleLanguageMenu;
window.changeLanguage = changeLanguage;