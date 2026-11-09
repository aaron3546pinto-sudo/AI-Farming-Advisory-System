"""
Soil-Based Agricultural Advisory Engine
Provides detailed agronomic recommendations based on Indian soil classifications:
- Black soil (Regur)
- Red soil
- Alluvial soil
- Laterite soil
- Sandy soil
- Clay soil
- Loamy soil

Includes suitable crops, nutrient dynamics, pH profiles, irrigation needs,
and soil health restoration & enhancement recommendations.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFO_PATH = os.path.join(BASE_DIR, "datasets", "crop_information.csv")

SOIL_PROFILES = {
    "Black soil": {
        "description": "Rich in montmorillonite clay minerals, deep black to dark brown in color. High moisture retention, swells when wet, cracks deeply when dry (self-ploughing nature). Highly fertile for deep-rooted and cash crops.",
        "dominant_regions": "Maharashtra (Deccan Plateau), Gujarat (Saurashtra), Madhya Pradesh (Malwa), Northern Karnataka, Telangana, Andhra Pradesh.",
        "typical_ph": "7.2 – 8.5 (Neutral to mildly alkaline)",
        "nutrient_status": {
            "Nitrogen": "Low to Medium (Requires basal and split application)",
            "Phosphorus": "Medium to Low (Fixation by calcium occurs; use SSP/DAP with bio-fertilizers)",
            "Potassium": "High to Very High (Naturally abundant in clay minerals)",
            "Micronutrients": "Prone to Zinc and Iron deficiency; responsive to foliar zinc sulfate"
        },
        "irrigation_guidance": "High water-holding capacity. Space irrigations further apart to prevent waterlogging and root rot. Laser land leveling is advised.",
        "improvement_suggestions": [
            "Apply 10-15 tonnes/ha of well-decomposed Farmyard Manure (FYM) or Vermicompost annually to improve soil structure.",
            "Practice green manuring with Sunnhemp (Crotalaria juncea) or Dhaincha (Sesbania aculeata) before Kharif sowing.",
            "In saline-sodic black soils, apply Gypsum based on soil testing and ensure sub-surface drainage.",
            "Adopt broad-bed and furrow (BBF) systems to facilitate safe excess drainage during heavy monsoon rains."
        ],
        "primary_crops": ["Cotton", "Soybean", "Sugarcane", "Jowar", "Chickpea", "Onion", "Wheat", "Pigeon Pea"]
    },
    "Red soil": {
        "description": "Formed by weathering of ancient crystalline and metamorphic rocks. Reddish tint due to high ferric oxide diffusion. Highly porous, friable, with low moisture retention capacity.",
        "dominant_regions": "Tamil Nadu, Southern Karnataka (Old Mysore/Plateau), Andhra Pradesh (Rayalaseema), Odisha, Telangana, Chota Nagpur plateau.",
        "typical_ph": "5.5 – 6.8 (Mildly acidic to neutral)",
        "nutrient_status": {
            "Nitrogen": "Low (Fast leaching losses)",
            "Phosphorus": "Low (High phosphorus fixation by iron/aluminum oxides)",
            "Potassium": "Medium",
            "Micronutrients": "Deficient in Boron and Molybdenum"
        },
        "irrigation_guidance": "Low water retention requires light, frequent irrigations or micro-irrigation systems (Drip/Sprinkler). Highly receptive to rainwater harvesting.",
        "improvement_suggestions": [
            "Incorporate substantial organic matter (FYM, compost, biochar) to boost water-holding capacity.",
            "Apply agricultural lime (CaCO3) or dolomite if pH drops below 5.8 to neutralize acidity and unlock fixed phosphorus.",
            "Use Phospho-bacteria (PSB) along with rock phosphate or SSP to enhance phosphorus availability.",
            "Deploy surface mulch (paddy straw, sugarcane trash) to reduce high evaporation losses."
        ],
        "primary_crops": ["Groundnut", "Ragi", "Maize", "Pulses", "Tomato", "Pigeon Pea", "Cotton", "Mango"]
    },
    "Alluvial soil": {
        "description": "Transported and deposited by major Indo-Gangetic and deltaic river systems (Khadar and Bhangar). Extremely fertile, loamy to silty clay texture, deep profile, and favorable water transmission.",
        "dominant_regions": "Punjab, Haryana, Uttar Pradesh, Bihar, West Bengal, Coastal Deltas of Andhra Pradesh, Tamil Nadu, and Odisha.",
        "typical_ph": "6.5 – 7.8 (Near neutral, optimal for nutrient uptake)",
        "nutrient_status": {
            "Nitrogen": "Medium to Low (High crop removal requires split urea/neem-coated urea)",
            "Phosphorus": "Medium to High",
            "Potassium": "Adequate to High",
            "Micronutrients": "Zinc and Sulfur deficiencies are widespread due to intensive cropping"
        },
        "irrigation_guidance": "Excellent balance of retention and percolation. Suitable for canal, tubewell, and drip irrigation systems.",
        "improvement_suggestions": [
            "Practice balanced NPK fertilization matching the recommended 4:2:1 ratio; avoid over-application of nitrogen.",
            "Apply 25 kg/ha Zinc Sulfate heptahydrate every 2-3 seasons to overcome widespread zinc depletion.",
            "Incorporate crop residues using happy seeders/super seeders instead of burning stubble to regenerate organic carbon.",
            "Include a short-duration pulse crop (Moong/Cowpea) in rice-wheat rotations to replenish nitrogen."
        ],
        "primary_crops": ["Rice", "Wheat", "Sugarcane", "Potato", "Mustard", "Maize", "Jute", "Tomato"]
    },
    "Laterite soil": {
        "description": "Formed under monsoon climate with alternate wet and dry seasons. Heavy tropical leaching washes away silica, leaving behind aluminum and iron sesquioxides. Coarse, acidic, and well-drained.",
        "dominant_regions": "Western Ghats (Kerala, Malnad Karnataka, Konkan Maharashtra), Nilgiris (Tamil Nadu), Assam hills, Coastal Odisha.",
        "typical_ph": "4.5 – 5.8 (Strongly to moderately acidic)",
        "nutrient_status": {
            "Nitrogen": "Low to Medium",
            "Phosphorus": "Very Low (High fixation by iron oxides)",
            "Potassium": "Low (Rapid leaching in hill slopes)",
            "Micronutrients": "High in Iron and Manganese; deficient in Calcium, Magnesium, and Boron"
        },
        "irrigation_guidance": "High infiltration rate. Hill slopes require contour bunding, bench terracing, and drip systems to prevent soil erosion.",
        "improvement_suggestions": [
            "Apply agricultural lime or dolomite at 1-2 tonnes/ha periodically based on soil test lime requirement.",
            "Apply rock phosphate or SSP enriched with compost to minimize phosphate fixation.",
            "Plant cover crops like Mucuna, Stylosanthes, or Centrosema in plantation orchards to prevent soil wash-off.",
            "Add neem cake and bio-fertilizers (Azospirillum, Mycorrhiza) to build biological soil activity."
        ],
        "primary_crops": ["Tea", "Coffee", "Coconut", "Rubber", "Cashew", "Ragi", "Rice", "Banana"]
    },
    "Sandy soil": {
        "description": "High sand fraction (>85%), coarse texture, very large pore spaces, negligible cohesion, and very low moisture retention. Extremely prone to wind erosion and nutrient leaching.",
        "dominant_regions": "Thar Desert & Semi-Arid Rajasthan (Marwar/Shekhawati), North-Western Gujarat (Kutch/Banaskantha), South-Western Haryana, Coastal sand dunes.",
        "typical_ph": "7.0 – 8.5 (Neutral to moderately alkaline)",
        "nutrient_status": {
            "Nitrogen": "Critically Deficient",
            "Phosphorus": "Low",
            "Potassium": "Low to Moderate",
            "Organic Carbon": "Extremely Low (< 0.2%)"
        },
        "irrigation_guidance": "Drip irrigation and low-discharge micro-sprinklers are mandatory. Avoid flood irrigation which causes rapid deep drainage below root zones.",
        "improvement_suggestions": [
            "Heavy incorporation of sheep/goat manure, biogas slurry, and compost to improve cation exchange capacity (CEC).",
            "Apply Hydrogel (Pusa Hydrogel) at 2.5-5 kg/ha at root depth to retain moisture during dry spells.",
            "Establish windbreaks and shelterbelts using indigenous drought-hardy trees (Khejri, Neem, Acacia).",
            "Apply fertilizers in multiple small split doses or through fertigation to prevent leaching below root zones."
        ],
        "primary_crops": ["Bajra", "Mustard", "Groundnut", "Chickpea", "Pulses", "Cluster Bean (Guar)", "Watermelon"]
    },
    "Clay soil": {
        "description": "Fine-textured soil composed of dense clay particles. Excellent nutrient holding capacity and water retention, but low permeability and poor internal aeration when saturated.",
        "dominant_regions": "Low-lying river basins, delta floodplains, coastal backwaters, and inland depressional zones across West Bengal, Kerala, Bihar, and Tamil Nadu.",
        "typical_ph": "6.0 – 7.5 (Slightly acidic to neutral)",
        "nutrient_status": {
            "Nitrogen": "Medium to High",
            "Phosphorus": "Medium",
            "Potassium": "High",
            "CEC": "Very High"
        },
        "irrigation_guidance": "High retention. Slow infiltration necessitates careful scheduling to avoid water stagnation. Puddling is ideal for wetland crops.",
        "improvement_suggestions": [
            "Add coarse sand, decomposed paddy husk, or sugarcane press mud to loosen heavy soil matrix and increase aeration.",
            "Avoid tilling when soil is excessively wet or dry to prevent hardpan formation and clod development.",
            "Install ridge and furrow layouts or raised beds for non-paddy crops to ensure aerated root zones.",
            "Use bio-inoculants like Trichoderma and mycorrhiza to prevent soil-borne damping-off fungi."
        ],
        "primary_crops": ["Rice", "Jute", "Sugarcane", "Banana", "Soybean", "Cotton", "Wheat"]
    },
    "Loamy soil": {
        "description": "The agricultural gold standard. Balanced blend of sand, silt, and clay particles (~40-40-20). Optimal water retention, excellent root aeration, and high biological activity.",
        "dominant_regions": "Alluvial plains, river valleys, and well-managed farming belts across Punjab, Haryana, Uttar Pradesh, Madhya Pradesh, Maharashtra, Karnataka.",
        "typical_ph": "6.2 – 7.5 (Optimal neutral range)",
        "nutrient_status": {
            "Nitrogen": "Moderate (Responds quickly to organic and mineral fertilizers)",
            "Phosphorus": "Optimal availability",
            "Potassium": "Balanced",
            "Organic Matter": "Medium to Good"
        },
        "irrigation_guidance": "Highly versatile; compatible with furrow, sprinkler, drip, and canal irrigation. Moisture drains easily while retaining plant-available water.",
        "improvement_suggestions": [
            "Maintain soil organic carbon above 0.75% through regular crop residue recycling and compost applications.",
            "Implement multi-tier crop rotation (cereals-pulses-oilseeds) to break pest cycles and preserve balanced nutrients.",
            "Test soil health card every 2 years to tailor micro-nutrient replenishment (Zinc, Boron, Sulfur).",
            "Adopt minimal or conservation tillage to protect beneficial earthworm populations and soil microbial biodiversity."
        ],
        "primary_crops": ["Wheat", "Maize", "Tomato", "Potato", "Onion", "Pulses", "Mustard", "Mango", "Rice"]
    }
}

def get_soil_advisory(soil_type):
    """
    Returns comprehensive agronomic advisory for the selected soil type.
    """
    profile = SOIL_PROFILES.get(soil_type, SOIL_PROFILES["Loamy soil"])
    
    # Get additional crop details from dataset
    crop_info = pd.DataFrame()
    if os.path.exists(INFO_PATH):
        crop_info = pd.read_csv(INFO_PATH)
        
    suitable_crop_details = []
    for crop in profile["primary_crops"]:
        matching = crop_info[crop_info["Crop Name"].str.lower() == crop.lower()] if not crop_info.empty else pd.DataFrame()
        if not matching.empty:
            row = matching.iloc[0]
            suitable_crop_details.append({
                "name": crop,
                "category": row.get("Category", "Crop"),
                "duration": row.get("Crop Duration", "Standard"),
                "irrigation": row.get("Irrigation Requirements", "Medium"),
                "yield": row.get("Expected Yield", "N/A"),
                "ph_range": row.get("Soil pH Range", "6.0-7.5")
            })
        else:
            suitable_crop_details.append({
                "name": crop,
                "category": "Agricultural Crop",
                "duration": "100-130 days",
                "irrigation": "Medium",
                "yield": "Standard yield",
                "ph_range": profile["typical_ph"]
            })
            
    return {
        "soil_type": soil_type,
        "description": profile["description"],
        "dominant_regions": profile["dominant_regions"],
        "typical_ph": profile["typical_ph"],
        "nutrient_status": profile["nutrient_status"],
        "irrigation_guidance": profile["irrigation_guidance"],
        "improvement_suggestions": profile["improvement_suggestions"],
        "crops": suitable_crop_details
    }
