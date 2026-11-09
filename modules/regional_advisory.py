"""
Regional and Seasonal Agricultural Advisory Engine
Provides state-wise and season-wise (Kharif, Rabi, Zaid) cropping strategies,
agro-climatic zone characteristics, sowing calendars, and management protocols.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "indian_crop_dataset.csv")
INFO_PATH = os.path.join(BASE_DIR, "datasets", "crop_information.csv")

SEASON_OVERVIEWS = {
    "Kharif": {
        "period": "June – October (South-West Monsoon Season)",
        "weather_profile": "High ambient temperatures, abundant monsoon cloud cover, elevated relative humidity (65-90%), and rainfall ranging from 600mm to over 2500mm.",
        "cropping_strategy": "Focus on moisture-loving, heavy-feeding, long-duration cereals, commercial fiber crops, and wet pulses. Field drainage is critical during heavy cloudbursts.",
        "key_practices": [
            "Land preparation with deep summer ploughing to expose hibernating pests and eradicate perennial weeds.",
            "Timely nursery raising for paddy with the first monsoon showers or using tubewell irrigation.",
            "Seed treatment with bio-fungicides (Trichoderma viride) and Rhizobium/Azotobacter inoculants.",
            "Ensure ridge and furrow or broad-bed sowing for cotton and soybean to mitigate heavy rain stagnation."
        ]
    },
    "Rabi": {
        "period": "October – April (Winter Season)",
        "weather_profile": "Mild to chilly winters, cool nights (6-18°C), low relative humidity (40-65%), clear sunny skies, and limited rainfall (200-450mm primarily from Western Disturbances).",
        "cropping_strategy": "Focus on thermal-sensitive cool-weather cereals, oilseeds, temperate pulses, and winter horticultural vegetables requiring assured canal or tubewell irrigation.",
        "key_practices": [
            "Conserve residual monsoon soil moisture by planking immediately after ploughing.",
            "Calibrate sowing depth (3-5 cm for wheat, 4-6 cm for chickpea) to avoid dehydration in dry topsoil.",
            "Critical crown root initiation (CRI) irrigation at 21 days for wheat is decisive for yield.",
            "Monitor weekly for aphid build-up on mustard during cloudy overcast weather spells."
        ]
    },
    "Zaid": {
        "period": "March – June (Summer / Pre-Monsoon Season)",
        "weather_profile": "Intense solar radiation, high temperatures (30-42°C), dry hot winds (Loo in northern plains), low humidity, and minimal rainfall.",
        "cropping_strategy": "Short-duration (60-85 days) catch crops, heat-tolerant nutri-cereals, cucurbits (watermelon, muskmelon, cucumber), summer moong, and fodder crops requiring reliable irrigation.",
        "key_practices": [
            "Ensure reliable water source (drip, furrow, or sprinkler) before initiating summer planting.",
            "Select short-duration varieties that mature before heavy monsoon rains damage mature pods or fruits.",
            "Heavy straw or plastic mulching to restrict high soil surface evapotranspiration.",
            "Frequent light irrigations during late evening or early morning hours to protect against heat stress."
        ]
    }
}

STATE_PROFILES = {
    "Maharashtra": {
        "agro_zones": "Western Ghats, Sub-Ghat, Western Maharashtra Plain, Scarcity Zone, Central & Eastern Vidarbha.",
        "soil_dominance": "Medium to Deep Black Cotton Soils (Regur), Mixed Red & Laterite in Konkan.",
        "key_crops": "Cotton, Soybean, Sugarcane, Jowar, Onion, Pigeon Pea, Chickpea, Mango (Alphonso), Banana.",
        "irrigation_pattern": "Predominantly rainfed (~82%), canal and well irrigation in Western Maharashtra and Godavari basin."
    },
    "Gujarat": {
        "agro_zones": "South Gujarat Heavy Rainfall, Middle Gujarat, North Gujarat Semi-Arid, Saurashtra, Kutch Arid.",
        "soil_dominance": "Black soils, Alluvial soils, Coastal Saline, Desert Sandy soils.",
        "key_crops": "Cotton, Groundnut, Castor, Bajra, Mustard, Potato, Sugarcane, Banana, Cumin.",
        "irrigation_pattern": "Sardar Sarovar canal network, tube-wells, check-dams, drip systems widely adopted."
    },
    "Punjab": {
        "agro_zones": "Sub-Mountainous Undulating, Central Alluvial Plain, South-Western Semi-Arid Plain.",
        "soil_dominance": "Deep, fertile alluvial loams and sandy loams with high agricultural productivity.",
        "key_crops": "Wheat, Rice (Basmati & Non-Basmati), Maize, Cotton, Potato, Mustard.",
        "irrigation_pattern": "Over 98% irrigated via canal network and electric tube-wells."
    },
    "Haryana": {
        "agro_zones": "Eastern Alluvial Plain, Western Semi-Arid Sandy Plain.",
        "soil_dominance": "Alluvial loams in east, light sandy to loamy sand in west and south.",
        "key_crops": "Wheat, Rice, Mustard, Bajra, Sugarcane, Cotton, Chickpea.",
        "irrigation_pattern": "Western Yamuna canal, Bhakra canal system, and tubewell networks."
    },
    "Rajasthan": {
        "agro_zones": "Arid Western Plain, Semi-Arid Eastern Plain, Flood-Prone Eastern Plain, Sub-Humid Southern Plain.",
        "soil_dominance": "Desert sandy soils, Brown loams, Alluvial plains, Black soils in Hadoti.",
        "key_crops": "Bajra, Mustard, Chickpea, Pulses (Moong/Moth), Groundnut, Wheat, Soybean, Guar.",
        "irrigation_pattern": "Indira Gandhi Canal (IGNP) in north-west, wells and farm ponds elsewhere."
    },
    "Uttar Pradesh": {
        "agro_zones": "Bhabhar & Tarai, Western Plain, Central Plain, Purvanchal (Eastern Plain), Bundelkhand.",
        "soil_dominance": "Rich Gangetic alluvium (Khadar and Bhangar), Red and mixed soils in Bundelkhand.",
        "key_crops": "Sugarcane, Wheat, Rice, Potato, Mustard, Maize, Mango, Pulses.",
        "irrigation_pattern": "Dense network of state tubewells, canals (Sharda, Upper Ganga), private borings."
    },
    "Madhya Pradesh": {
        "agro_zones": "Malwa Plateau, Nimar Valley, Narmada Valley, Bundelkhand, Mahakoshal/Vindhyan.",
        "soil_dominance": "Deep and medium black soils, mixed red and yellow soils.",
        "key_crops": "Soybean (Soy State), Wheat (Sharbati), Chickpea, Maize, Cotton, Onion, Tomato.",
        "irrigation_pattern": "Narmada river valley projects, tube-wells, check dams, and open dug wells."
    },
    "Karnataka": {
        "agro_zones": "Coastal Zone, Hilly Zone (Malnad), Northern Dry Zone, Southern Dry Zone, Transition Zone.",
        "soil_dominance": "Red sandy loams in south, deep black soils in north, laterites in coastal/hilly tracts.",
        "key_crops": "Ragi, Coffee, Maize, Cotton, Sugarcane, Jowar, Groundnut, Tomato, Coconut.",
        "irrigation_pattern": "Krishna and Cauvery river basins, open borewells, tank irrigation systems."
    },
    "Tamil Nadu": {
        "agro_zones": "Cauvery Delta, Western Zone, Southern Zone, High Rainfall Hilly Zone.",
        "soil_dominance": "Red soils, Black soils, Coastal Alluvial soils, Laterites in western hills.",
        "key_crops": "Rice, Banana, Sugarcane, Groundnut, Coconut, Cotton, Ragi, Tea (Nilgiris).",
        "irrigation_pattern": "Cauvery river irrigation, seasonal tanks (Eris), open and bore wells."
    },
    "Kerala": {
        "agro_zones": "Highland (Western Ghats), Midland (Hills and Valleys), Lowland (Coastal Plains).",
        "soil_dominance": "Acidic Laterite soils (~65%), Coastal alluvial and peat (Kari/Kuttanad) soils.",
        "key_crops": "Coconut, Rubber, Tea, Coffee, Rice (Pokkali/Kuttanad), Banana, Spices (Pepper/Cardamom).",
        "irrigation_pattern": "Perennial rainfall (SW & NE monsoons), river lift irrigation, community tanks."
    },
    "Andhra Pradesh": {
        "agro_zones": "North Coastal Zone, Godavari Delta, Krishna Delta, Southern Zone, Scarce Rainfall Zone (Rayalaseema).",
        "soil_dominance": "Red soils (66%), Black soils (25%), Alluvial deltas, Coastal sands.",
        "key_crops": "Rice, Cotton, Tomato, Groundnut, Maize, Chillies, Tobacco, Mango, Coconut.",
        "irrigation_pattern": "Godavari and Krishna barrages, Polavaram canals, borewells in Rayalaseema."
    },
    "Telangana": {
        "agro_zones": "Northern Telangana Zone, Southern Telangana Zone, Central Telangana Zone.",
        "soil_dominance": "Chalka soils (Red sandy loams ~54%), Black cotton soils (~28%).",
        "key_crops": "Cotton, Rice, Maize, Soybean, Red Gram, Turmeric, Mango.",
        "irrigation_pattern": "Kaleshwaram Lift Irrigation Project, Mission Kakatiya restored tanks, borewells."
    },
    "West Bengal": {
        "agro_zones": "Hill Zone (Darjeeling), Terai-Dooars, Old Alluvial Zone, New Alluvial Zone, Coastal Saline.",
        "soil_dominance": "New and old Gangetic alluvium, coastal saline soils, red laterite in Rarh.",
        "key_crops": "Rice (Aman, Aus, Boro), Jute (Golden Fiber), Potato, Tea, Mustard, Vegetables.",
        "irrigation_pattern": "Canal networks (DVC, Mayurakshi), shallow tubewells, river lift irrigation."
    },
    "Bihar": {
        "agro_zones": "North-West Alluvial Plain, North-East Alluvial Plain, South Alluvial Plain.",
        "soil_dominance": "Deep Gangetic alluvial soils (calcareous in west, light acidic in east).",
        "key_crops": "Rice, Wheat, Maize (Kharif/Rabi/Zaid), Sugarcane, Potato, Jute, Mango, Litchi.",
        "irrigation_pattern": "Gandak, Kosi, and Sone canal networks, private diesel/electric tubewells."
    },
    "Assam": {
        "agro_zones": "Upper Brahmaputra Valley, Central Brahmaputra, Lower Brahmaputra, Barak Valley, Hills Zone.",
        "soil_dominance": "Riverine alluvium, acidic red and lateritic soils on valley borders.",
        "key_crops": "Tea (Assam Orthodox & CTC), Rice (Sali, Ahu, Boro), Jute, Mustard, Banana, Arecanut.",
        "irrigation_pattern": "High monsoonal precipitation; shallow tubewells and stream diversion channels (Dongs)."
    },
    "Odisha": {
        "agro_zones": "North-Western Plateau, North-Eastern Coastal Plain, East & South-Eastern Coastal Plain, Western Undulating.",
        "soil_dominance": "Red soils, Laterite soils, Coastal Alluvial soils, Black soils in pockets.",
        "key_crops": "Rice, Groundnut, Pulses, Ragi, Cotton, Sugarcane, Coconut, Tomato.",
        "irrigation_pattern": "Mahanadi delta canal system, Hirakud command, borewells, check dams."
    },
    "Himachal Pradesh": {
        "agro_zones": "Sub-Montane & Low Hills Sub-Tropical, Mid Hills Sub-Humid, High Hills Wet Temperate, Dry Cold Zone.",
        "soil_dominance": "Brown hill loams, mountain soils, podzols in conifer zones.",
        "key_crops": "Apple, Maize, Wheat, Potato, Tomato (Off-Season), Tea (Kangra), Pulses, Ginger.",
        "irrigation_pattern": "Kuhl systems (traditional mountain gravity water channels), drip in orchards, lift irrigation."
    }
}

def get_seasonal_recommendations(state, season):
    """
    Returns tailored seasonal agricultural guidance for a specific state and season.
    """
    season_info = SEASON_OVERVIEWS.get(season, SEASON_OVERVIEWS["Kharif"])
    state_info = STATE_PROFILES.get(state, {
        "agro_zones": "Regional Agro-Climatic Zone",
        "soil_dominance": "Regional Alluvial and Loamy Soils",
        "key_crops": "Rice, Wheat, Maize, Pulses",
        "irrigation_pattern": "Canals, wells, and rainfed systems"
    })
    
    # Filter dataset for crops matching state and season
    df = pd.read_csv(DATASET_PATH) if os.path.exists(DATASET_PATH) else pd.DataFrame()
    crop_info = pd.read_csv(INFO_PATH) if os.path.exists(INFO_PATH) else pd.DataFrame()
    
    suitable_crops = []
    if not df.empty:
        subset = df[(df["State"] == state) & (df["Season"] == season)]
        top_crops = subset["Crop Name"].value_counts().index.tolist()
        
        for crop in top_crops:
            matching = crop_info[crop_info["Crop Name"].str.lower() == crop.lower()] if not crop_info.empty else pd.DataFrame()
            if not matching.empty:
                row = matching.iloc[0]
                suitable_crops.append({
                    "crop": crop,
                    "category": row.get("Category", "Crop"),
                    "duration": row.get("Crop Duration", "Standard"),
                    "irrigation": row.get("Irrigation Requirements", "Medium"),
                    "yield": row.get("Expected Yield", "N/A"),
                    "guidance": row.get("Cultivation Guide", "Follow standard agronomic practices.")
                })
            else:
                suitable_crops.append({
                    "crop": crop,
                    "category": "Agricultural Crop",
                    "duration": "100-130 days",
                    "irrigation": "Medium",
                    "yield": "Standard regional yield",
                    "guidance": "Recommended for regional seasonal cultivation."
                })
                
    if not suitable_crops:
        # Fallback based on state profile key crops
        for c in [x.strip() for x in state_info["key_crops"].split(",")][:5]:
            suitable_crops.append({
                "crop": c,
                "category": "Regional Commercial / Food Crop",
                "duration": "90-130 days",
                "irrigation": "Medium",
                "yield": "Standard regional yield",
                "guidance": f"Major crop of {state} suitable for {season} cultivation."
            })
            
    return {
        "state": state,
        "season": season,
        "season_period": season_info["period"],
        "weather_profile": season_info["weather_profile"],
        "cropping_strategy": season_info["cropping_strategy"],
        "key_practices": season_info["key_practices"],
        "state_agro_zones": state_info["agro_zones"],
        "state_soils": state_info["soil_dominance"],
        "state_irrigation": state_info["irrigation_pattern"],
        "recommended_crops": suitable_crops
    }
