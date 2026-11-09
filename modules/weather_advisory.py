"""
Weather-Based Agricultural Advisory Engine
Generates real-time or manual weather recommendations:
- Suitable crops
- Crops to avoid under current weather stress
- Irrigation & water advisory
- Agro-meteorological precautions (disease alerts, drainage, heat/chill protection)
- Sowing/planting windows
"""

import os
import pandas as pd
from utils.weather_api import fetch_live_weather

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFO_PATH = os.path.join(BASE_DIR, "datasets", "crop_information.csv")

def get_crop_data():
    if os.path.exists(INFO_PATH):
        return pd.read_csv(INFO_PATH)
    return pd.DataFrame()

def generate_weather_advisory(state, district, temp, humidity, rainfall, season, is_live=False, source_label="Manual Input"):
    """
    Evaluates ambient weather parameters and generates holistic agronomic advisory.
    """
    df = get_crop_data()
    temp = float(temp)
    humidity = float(humidity)
    rainfall = float(rainfall)
    
    suitable_crops = []
    crops_to_avoid = []
    precautions = []
    water_advisory = []
    planting_windows = []
    
    # Analyze Temperature
    if temp > 35.0:
        precautions.append("🔥 <b>Heat Stress Warning:</b> High temperature detected (>35°C). Implement light evening/early morning irrigation to prevent floral abortion and sunscald.")
        crops_to_avoid.append({"crop": "Wheat / Potato / Mustard", "reason": "Cool-season crops suffer rapid desiccation and poor tuber/grain formation under high heat."})
    elif temp < 15.0:
        precautions.append("❄️ <b>Cold/Chilling Warning:</b> Low temperature (<15°C). Ensure mulch cover to maintain root zone warmth. Risk of delayed germination.")
        crops_to_avoid.append({"crop": "Cotton / Sugarcane (Germination) / Rice", "reason": "Sub-optimal thermal units stunt vegetative development and impede root nutrient uptake."})
    else:
        precautions.append("✅ <b>Favorable Thermal Conditions:</b> Temperature is within normal physiological limits for regional crops.")

    # Analyze Humidity
    if humidity > 80.0:
        precautions.append("💧 <b>Fungal Pathogen & Blight Alert:</b> High relative humidity (>80%) creates ideal conditions for fungal leaf spots, downy mildew, and rusts. Avoid overhead sprinkler irrigation and schedule prophylactic bio-fungicide sprays (Trichoderma / Mancozeb).")
        crops_to_avoid.append({"crop": "Chickpea / Mustard", "reason": "Highly susceptible to Aschochyta blight, sclerotinia stem rot, and white rust under sustained humid conditions."})
    elif humidity < 40.0:
        precautions.append("☀️ <b>Dry Atmosphere / Transpiration Alert:</b> Low humidity accelerates evapotranspiration. Deploy straw or plastic mulch to conserve surface moisture.")
    
    # Analyze Rainfall / Precipitation
    if rainfall > 1200.0:
        water_advisory.append("🌊 <b>Excess Moisture & Drainage:</b> High precipitation zone. Ensure free-flowing drainage ditches in fields to prevent root hypoxia and collar rot.")
        crops_to_avoid.append({"crop": "Pulses (Moong/Urad) / Onion / Groundnut", "reason": "Extremely vulnerable to root rot, pod rotting, and waterlogging."})
    elif rainfall < 500.0:
        water_advisory.append("🏜️ <b>Deficit Moisture Zone:</b> Low rainfall. Adopt micro-irrigation (Drip or Sprinkler) to achieve 40-50% water savings.")
    else:
        water_advisory.append("🌧️ <b>Moderate Moisture Availability:</b> Favorable for rainfed and semi-irrigated cultivation. Maintain regular moisture monitoring.")
        
    # Crop Suitability Matching
    for _, row in df.iterrows():
        crop_name = row["Crop Name"]
        states = str(row["Suitable Indian States"])
        seasons = str(row["Sowing & Harvesting Seasons"])
        
        # Check if suitable for state & season
        if state in states and season in seasons:
            # Parse temperature range
            temp_range = str(row.get("Optimal Temperature Range (°C)", "15 - 35")).split("-")
            t_min = float(temp_range[0].strip()) if len(temp_range) == 2 else 15.0
            t_max = float(temp_range[1].strip()) if len(temp_range) == 2 else 35.0
            
            if (t_min - 4.0) <= temp <= (t_max + 4.0):
                suitable_crops.append({
                    "crop": crop_name,
                    "category": row.get("Category", "Crop"),
                    "water_need": row.get("Irrigation Requirements", "Medium"),
                    "duration": row.get("Crop Duration", "Standard"),
                    "reason": f"Matches regional thermal ({temp:.1f}°C) and seasonal ({season}) dynamics."
                })
                
    # Fallback if filtered list is small
    if len(suitable_crops) < 3:
        for fallback_crop in ["Rice", "Maize", "Jowar", "Pulses"]:
            if not any(c["crop"] == fallback_crop for c in suitable_crops):
                suitable_crops.append({
                    "crop": fallback_crop,
                    "category": "Cereal / Pulse",
                    "water_need": "Medium",
                    "duration": "90-110 days",
                    "reason": "Hardy general-purpose crop adaptable to varying seasonal weather."
                })
                
    # Seasonal Planting Windows
    if season == "Kharif":
        planting_windows = [
            "Early Sowing: 1st June – 25th June (with onset of South-West Monsoon)",
            "Main Sowing Window: 25th June – 15th July (optimal germination conditions)",
            "Late Sowing (Short duration cultivars): 15th July – 31st July"
        ]
    elif season == "Rabi":
        planting_windows = [
            "Early Sowing: 15th October – 31st October (Mustard, Gram, Early Wheat)",
            "Timely Sowing Window: 1st November – 25th November (timely sown irrigated wheat)",
            "Late Sowing Window: 25th November – 15th December (heat-tolerant late varieties)"
        ]
    else: # Zaid
        planting_windows = [
            "Primary Sowing Window: 15th February – 15th March (Vegetables, Watermelon, Moong)",
            "Late Window: 15th March – 5th April (Summer fodder and short-duration pulses)"
        ]
        
    return {
        "is_live": is_live,
        "source": source_label,
        "location": f"{district}, {state}" if district else state,
        "current_weather": {
            "temperature": temp,
            "humidity": humidity,
            "rainfall": rainfall,
            "season": season
        },
        "suitable_crops": suitable_crops[:6],
        "crops_to_avoid": crops_to_avoid,
        "water_advisory": water_advisory,
        "precautions": precautions,
        "planting_windows": planting_windows
    }
