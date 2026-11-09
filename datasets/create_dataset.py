"""
Dataset Generator for Indian Agricultural Advisory System
Generates:
1. datasets/indian_crop_dataset.csv (Observation records for ML training)
2. datasets/crop_information.csv (Agronomic encyclopedia & reference guidelines)

Based on benchmarks from ICAR, IMD, and Ministry of Agriculture & Farmers Welfare.
"""

import os
import random
import numpy as np
import pandas as pd

# Define random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "indian_crop_dataset.csv")
INFO_PATH = os.path.join(BASE_DIR, "crop_information.csv")

# State & District mapping with representative soils & rainfall baselines
STATE_DISTRICT_MAP = {
    "Maharashtra": {
        "districts": ["Vidarbha (Nagpur/Amravati)", "Marathwada (Aurangabad/Nanded)", "Western Maharashtra (Pune/Nashik)", "Konkan (Ratnagiri/Thane)"],
        "soils": ["Black soil", "Red soil", "Laterite soil", "Loamy soil"],
        "base_temp": (22, 34),
        "base_rain": (750, 1400)
    },
    "Gujarat": {
        "districts": ["Saurashtra (Rajkot/Junagadh)", "North Gujarat (Mehsana/Patan)", "South Gujarat (Surat/Navsari)", "Kutch (Bhuj)"],
        "soils": ["Black soil", "Alluvial soil", "Sandy soil", "Clay soil"],
        "base_temp": (23, 36),
        "base_rain": (550, 1100)
    },
    "Punjab": {
        "districts": ["Majha (Amritsar/Gurdaspur)", "Doaba (Jalandhar/Hoshiarpur)", "Malwa (Ludhiana/Bathinda)"],
        "soils": ["Alluvial soil", "Loamy soil", "Clay soil"],
        "base_temp": (14, 33),
        "base_rain": (450, 750)
    },
    "Haryana": {
        "districts": ["Eastern Haryana (Karnal/Kurukshetra)", "Western Haryana (Hisar/Sirsa)", "Southern Haryana (Gurugram/Rewari)"],
        "soils": ["Alluvial soil", "Sandy soil", "Loamy soil"],
        "base_temp": (15, 34),
        "base_rain": (400, 700)
    },
    "Rajasthan": {
        "districts": ["Marwar (Jodhpur/Nagaur)", "Mewar (Udaipur/Chittorgarh)", "Shekhawati (Sikar/Jhunjhunu)", "Hadoti (Kota/Baran)"],
        "soils": ["Sandy soil", "Loamy soil", "Black soil", "Red soil"],
        "base_temp": (18, 38),
        "base_rain": (300, 650)
    },
    "Uttar Pradesh": {
        "districts": ["Western UP (Meerut/Aligarh)", "Purvanchal (Varanasi/Gorakhpur)", "Awadh (Lucknow/Kanpur)", "Bundelkhand (Jhansi/Banda)"],
        "soils": ["Alluvial soil", "Loamy soil", "Clay soil", "Red soil"],
        "base_temp": (16, 34),
        "base_rain": (650, 1150)
    },
    "Madhya Pradesh": {
        "districts": ["Malwa (Indore/Ujjain)", "Nimar (Khargone/Khandwa)", "Bundelkhand (Sagar/Chhatarpur)", "Mahakoshal (Jabalpur/Seoni)"],
        "soils": ["Black soil", "Red soil", "Alluvial soil", "Loamy soil"],
        "base_temp": (18, 35),
        "base_rain": (750, 1250)
    },
    "Karnataka": {
        "districts": ["Old Mysore (Mysuru/Mandya)", "North Karnataka (Belagavi/Dharwad)", "Coastal Karnataka (Mangaluru/Udupi)", "Malnad (Chikkamagaluru/Shivamogga)"],
        "soils": ["Red soil", "Black soil", "Laterite soil", "Loamy soil"],
        "base_temp": (20, 32),
        "base_rain": (700, 2200)
    },
    "Tamil Nadu": {
        "districts": ["Cauvery Delta (Thanjavur/Tiruvarur)", "Western Zone (Coimbatore/Erode)", "Southern Zone (Madurai/Tirunelveli)", "Northern Zone (Vellore/Kanchipuram)"],
        "soils": ["Red soil", "Black soil", "Alluvial soil", "Laterite soil"],
        "base_temp": (24, 35),
        "base_rain": (800, 1300)
    },
    "Kerala": {
        "districts": ["Central Kerala (Ernakulam/Thrissur)", "Malabar (Kozhikode/Wayanad)", "Travancore (Kottayam/Alappuzha)", "High Ranges (Idukki)"],
        "soils": ["Laterite soil", "Alluvial soil", "Red soil", "Clay soil"],
        "base_temp": (23, 32),
        "base_rain": (1800, 3100)
    },
    "Andhra Pradesh": {
        "districts": ["Coastal Andhra (Guntur/Krishna/Godavari)", "Rayalaseema (Anantapur/Kurnool/Kadapa)", "North Coastal (Visakhapatnam)"],
        "soils": ["Red soil", "Black soil", "Alluvial soil", "Sandy soil"],
        "base_temp": (24, 36),
        "base_rain": (650, 1150)
    },
    "Telangana": {
        "districts": ["Northern Telangana (Warangal/Karimnagar)", "Southern Telangana (Mahabubnagar/Nalgonda)", "Central Telangana (Ranga Reddy/Medak)"],
        "soils": ["Red soil", "Black soil", "Loamy soil"],
        "base_temp": (22, 36),
        "base_rain": (750, 1100)
    },
    "West Bengal": {
        "districts": ["Gangetic Plain (Nadia/Murshidabad/Burdwan)", "Terai & Dooars (Jalpaiguri/Cooch Behar)", "Rarh (Birbhum/Bankura)", "Himalayan (Darjeeling)"],
        "soils": ["Alluvial soil", "Clay soil", "Loamy soil", "Laterite soil"],
        "base_temp": (18, 33),
        "base_rain": (1200, 2400)
    },
    "Bihar": {
        "districts": ["North Bihar (Muzaffarpur/Darbhanga/Purnia)", "South Bihar (Patna/Gaya/Nalanda)", "Kosi Region (Saharsa/Katihar)"],
        "soils": ["Alluvial soil", "Clay soil", "Loamy soil", "Sandy soil"],
        "base_temp": (16, 33),
        "base_rain": (950, 1400)
    },
    "Assam": {
        "districts": ["Brahmaputra Valley (Kamrup/Sonitpur/Jorhat)", "Upper Assam (Dibrugarh/Tinsukia)", "Barak Valley (Cachar/Karimganj)"],
        "soils": ["Alluvial soil", "Red soil", "Laterite soil", "Clay soil"],
        "base_temp": (18, 31),
        "base_rain": (1600, 2800)
    },
    "Odisha": {
        "districts": ["Coastal Odisha (Cuttack/Puri/Balasore)", "Western Odisha (Sambalpur/Bargarh)", "Southern Odisha (Ganjam/Koraput)"],
        "soils": ["Red soil", "Laterite soil", "Alluvial soil", "Clay soil"],
        "base_temp": (21, 35),
        "base_rain": (1150, 1650)
    },
    "Himachal Pradesh": {
        "districts": ["Kangra Valley (Dharamshala/Kangra)", "Mid Hills (Mandi/Solan/Bilaspur)", "High Hills (Shimla/Kullu)"],
        "soils": ["Loamy soil", "Red soil", "Sandy soil", "Alluvial soil"],
        "base_temp": (8, 26),
        "base_rain": (900, 1600)
    }
}

# Crop Agronomic Profiles (Benchmarks from ICAR / Directorate of Economics & Statistics)
CROP_PROFILES = {
    "Rice": {
        "suitable_states": ["West Bengal", "Uttar Pradesh", "Punjab", "Andhra Pradesh", "Tamil Nadu", "Bihar", "Odisha", "Assam", "Kerala", "Telangana", "Karnataka", "Maharashtra", "Haryana"],
        "seasons": ["Kharif", "Rabi"],
        "soils": ["Clay soil", "Alluvial soil", "Loamy soil", "Red soil"],
        "temp_opt": (22.0, 34.0),
        "temp_min": 18.0,
        "temp_max": 38.0,
        "humidity_opt": (68.0, 88.0),
        "rain_opt": (1000.0, 2200.0),
        "ph_opt": (5.5, 7.2),
        "npk": (100, 50, 50),
        "irrigation": "High",
        "growing_period": "June - Nov (Kharif) / Dec - April (Rabi)",
        "duration": "110-145 days",
        "yield": "35-55 quintals/ha"
    },
    "Wheat": {
        "suitable_states": ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", "Rajasthan", "Bihar", "Gujarat", "Maharashtra", "Himachal Pradesh"],
        "seasons": ["Rabi"],
        "soils": ["Alluvial soil", "Loamy soil", "Clay soil", "Black soil"],
        "temp_opt": (12.0, 25.0),
        "temp_min": 7.0,
        "temp_max": 28.0,
        "humidity_opt": (42.0, 68.0),
        "rain_opt": (300.0, 550.0),
        "ph_opt": (6.0, 7.8),
        "npk": (120, 60, 40),
        "irrigation": "Medium",
        "growing_period": "Nov - April (Rabi)",
        "duration": "115-140 days",
        "yield": "40-62 quintals/ha"
    },
    "Cotton": {
        "suitable_states": ["Maharashtra", "Gujarat", "Telangana", "Andhra Pradesh", "Karnataka", "Punjab", "Haryana", "Rajasthan", "Madhya Pradesh", "Tamil Nadu"],
        "seasons": ["Kharif"],
        "soils": ["Black soil", "Alluvial soil", "Red soil", "Loamy soil"],
        "temp_opt": (23.0, 35.0),
        "temp_min": 18.0,
        "temp_max": 40.0,
        "humidity_opt": (50.0, 75.0),
        "rain_opt": (600.0, 1100.0),
        "ph_opt": (6.0, 8.2),
        "npk": (110, 50, 50),
        "irrigation": "Medium",
        "growing_period": "May/June - Nov/Dec",
        "duration": "150-180 days",
        "yield": "18-32 quintals/ha"
    },
    "Sugarcane": {
        "suitable_states": ["Uttar Pradesh", "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat", "Bihar", "Andhra Pradesh", "Telangana", "Punjab", "Haryana"],
        "seasons": ["Kharif", "Zaid"],
        "soils": ["Alluvial soil", "Black soil", "Loamy soil", "Clay soil"],
        "temp_opt": (22.0, 36.0),
        "temp_min": 16.0,
        "temp_max": 42.0,
        "humidity_opt": (60.0, 85.0),
        "rain_opt": (1200.0, 2000.0),
        "ph_opt": (6.5, 8.0),
        "npk": (180, 80, 80),
        "irrigation": "Very High",
        "growing_period": "Jan - Dec (Year-round cycle)",
        "duration": "300-365 days",
        "yield": "750-1100 quintals/ha"
    },
    "Soybean": {
        "suitable_states": ["Madhya Pradesh", "Maharashtra", "Rajasthan", "Karnataka", "Telangana", "Gujarat", "Andhra Pradesh"],
        "seasons": ["Kharif"],
        "soils": ["Black soil", "Loamy soil", "Clay soil"],
        "temp_opt": (21.0, 32.0),
        "temp_min": 18.0,
        "temp_max": 36.0,
        "humidity_opt": (60.0, 82.0),
        "rain_opt": (650.0, 1000.0),
        "ph_opt": (6.0, 7.5),
        "npk": (30, 60, 40),
        "irrigation": "Medium",
        "growing_period": "June/July - Oct",
        "duration": "90-110 days",
        "yield": "16-26 quintals/ha"
    },
    "Groundnut": {
        "suitable_states": ["Gujarat", "Andhra Pradesh", "Tamil Nadu", "Karnataka", "Rajasthan", "Maharashtra", "Odisha", "Telangana"],
        "seasons": ["Kharif", "Zaid"],
        "soils": ["Sandy soil", "Red soil", "Loamy soil"],
        "temp_opt": (22.0, 33.0),
        "temp_min": 18.0,
        "temp_max": 36.0,
        "humidity_opt": (50.0, 75.0),
        "rain_opt": (500.0, 850.0),
        "ph_opt": (6.0, 7.5),
        "npk": (25, 50, 50),
        "irrigation": "Medium",
        "growing_period": "June - Oct (Kharif) / Jan - May (Zaid)",
        "duration": "105-125 days",
        "yield": "18-30 quintals/ha"
    },
    "Mustard": {
        "suitable_states": ["Rajasthan", "Haryana", "Madhya Pradesh", "Uttar Pradesh", "West Bengal", "Punjab", "Gujarat", "Bihar", "Assam"],
        "seasons": ["Rabi"],
        "soils": ["Loamy soil", "Alluvial soil", "Sandy soil"],
        "temp_opt": (12.0, 24.0),
        "temp_min": 6.0,
        "temp_max": 28.0,
        "humidity_opt": (45.0, 70.0),
        "rain_opt": (250.0, 450.0),
        "ph_opt": (6.0, 7.6),
        "npk": (70, 35, 30),
        "irrigation": "Low",
        "growing_period": "Oct - March (Rabi)",
        "duration": "105-130 days",
        "yield": "12-20 quintals/ha"
    },
    "Chickpea": {
        "suitable_states": ["Madhya Pradesh", "Maharashtra", "Rajasthan", "Karnataka", "Uttar Pradesh", "Gujarat", "Andhra Pradesh", "Haryana"],
        "seasons": ["Rabi"],
        "soils": ["Black soil", "Loamy soil", "Alluvial soil"],
        "temp_opt": (14.0, 26.0),
        "temp_min": 8.0,
        "temp_max": 30.0,
        "humidity_opt": (40.0, 62.0),
        "rain_opt": (300.0, 520.0),
        "ph_opt": (6.2, 7.8),
        "npk": (25, 50, 25),
        "irrigation": "Low",
        "growing_period": "Oct/Nov - March/April",
        "duration": "95-120 days",
        "yield": "14-25 quintals/ha"
    },
    "Pigeon Pea": {
        "suitable_states": ["Maharashtra", "Madhya Pradesh", "Karnataka", "Uttar Pradesh", "Gujarat", "Telangana", "Andhra Pradesh", "Odisha", "Bihar"],
        "seasons": ["Kharif"],
        "soils": ["Black soil", "Red soil", "Loamy soil", "Alluvial soil"],
        "temp_opt": (20.0, 34.0),
        "temp_min": 16.0,
        "temp_max": 38.0,
        "humidity_opt": (52.0, 75.0),
        "rain_opt": (600.0, 950.0),
        "ph_opt": (6.0, 7.8),
        "npk": (25, 50, 25),
        "irrigation": "Low",
        "growing_period": "June - Dec/Jan",
        "duration": "140-180 days",
        "yield": "12-22 quintals/ha"
    },
    "Jowar": {
        "suitable_states": ["Maharashtra", "Karnataka", "Madhya Pradesh", "Andhra Pradesh", "Rajasthan", "Telangana", "Tamil Nadu", "Gujarat"],
        "seasons": ["Kharif", "Rabi"],
        "soils": ["Black soil", "Red soil", "Loamy soil", "Clay soil"],
        "temp_opt": (22.0, 34.0),
        "temp_min": 16.0,
        "temp_max": 40.0,
        "humidity_opt": (40.0, 70.0),
        "rain_opt": (450.0, 800.0),
        "ph_opt": (6.0, 8.5),
        "npk": (75, 35, 35),
        "irrigation": "Low",
        "growing_period": "June - Oct (Kharif) / Sept - Jan (Rabi)",
        "duration": "100-120 days",
        "yield": "20-35 quintals/ha"
    },
    "Bajra": {
        "suitable_states": ["Rajasthan", "Gujarat", "Haryana", "Uttar Pradesh", "Maharashtra", "Madhya Pradesh", "Karnataka", "Tamil Nadu"],
        "seasons": ["Kharif", "Zaid"],
        "soils": ["Sandy soil", "Loamy soil", "Red soil"],
        "temp_opt": (25.0, 38.0),
        "temp_min": 20.0,
        "temp_max": 43.0,
        "humidity_opt": (35.0, 65.0),
        "rain_opt": (350.0, 600.0),
        "ph_opt": (6.5, 8.5),
        "npk": (70, 35, 25),
        "irrigation": "Low",
        "growing_period": "June - Sept (Kharif) / March - June (Zaid)",
        "duration": "75-90 days",
        "yield": "16-28 quintals/ha"
    },
    "Ragi": {
        "suitable_states": ["Karnataka", "Tamil Nadu", "Odisha", "Andhra Pradesh", "Maharashtra", "Himachal Pradesh", "Kerala"],
        "seasons": ["Kharif"],
        "soils": ["Red soil", "Laterite soil", "Loamy soil"],
        "temp_opt": (20.0, 32.0),
        "temp_min": 16.0,
        "temp_max": 35.0,
        "humidity_opt": (60.0, 82.0),
        "rain_opt": (600.0, 1000.0),
        "ph_opt": (5.5, 7.5),
        "npk": (60, 30, 25),
        "irrigation": "Low",
        "growing_period": "July - Nov (Kharif)",
        "duration": "100-125 days",
        "yield": "18-32 quintals/ha"
    },
    "Pulses": {
        "suitable_states": ["Madhya Pradesh", "Rajasthan", "Maharashtra", "Uttar Pradesh", "Andhra Pradesh", "Odisha", "Karnataka", "Bihar", "Gujarat", "Tamil Nadu"],
        "seasons": ["Kharif", "Rabi", "Zaid"],
        "soils": ["Loamy soil", "Alluvial soil", "Red soil", "Black soil"],
        "temp_opt": (22.0, 34.0),
        "temp_min": 16.0,
        "temp_max": 38.0,
        "humidity_opt": (45.0, 75.0),
        "rain_opt": (400.0, 750.0),
        "ph_opt": (6.0, 7.5),
        "npk": (20, 45, 25),
        "irrigation": "Low",
        "growing_period": "June - Sept / Oct - Jan / March - May",
        "duration": "65-85 days",
        "yield": "8-16 quintals/ha"
    },
    "Maize": {
        "suitable_states": ["Karnataka", "Madhya Pradesh", "Bihar", "Maharashtra", "Telangana", "Punjab", "Andhra Pradesh", "Rajasthan", "Uttar Pradesh", "Himachal Pradesh"],
        "seasons": ["Kharif", "Rabi"],
        "soils": ["Alluvial soil", "Red soil", "Loamy soil", "Black soil"],
        "temp_opt": (18.0, 32.0),
        "temp_min": 14.0,
        "temp_max": 36.0,
        "humidity_opt": (55.0, 80.0),
        "rain_opt": (600.0, 1050.0),
        "ph_opt": (5.8, 7.5),
        "npk": (120, 60, 50),
        "irrigation": "Medium",
        "growing_period": "June - Oct (Kharif) / Oct - Feb (Rabi)",
        "duration": "90-115 days",
        "yield": "36-58 quintals/ha"
    },
    "Potato": {
        "suitable_states": ["Uttar Pradesh", "West Bengal", "Bihar", "Gujarat", "Madhya Pradesh", "Punjab", "Haryana", "Assam", "Himachal Pradesh"],
        "seasons": ["Rabi"],
        "soils": ["Sandy soil", "Loamy soil", "Alluvial soil"],
        "temp_opt": (14.0, 24.0),
        "temp_min": 8.0,
        "temp_max": 28.0,
        "humidity_opt": (55.0, 78.0),
        "rain_opt": (400.0, 650.0),
        "ph_opt": (5.2, 6.8),
        "npk": (150, 80, 120),
        "irrigation": "High",
        "growing_period": "Oct/Nov - Feb/March",
        "duration": "80-110 days",
        "yield": "220-380 quintals/ha"
    },
    "Onion": {
        "suitable_states": ["Maharashtra", "Karnataka", "Madhya Pradesh", "Gujarat", "Bihar", "Rajasthan", "Andhra Pradesh", "Telangana", "Tamil Nadu"],
        "seasons": ["Kharif", "Rabi", "Zaid"],
        "soils": ["Loamy soil", "Black soil", "Alluvial soil", "Red soil"],
        "temp_opt": (15.0, 30.0),
        "temp_min": 10.0,
        "temp_max": 35.0,
        "humidity_opt": (48.0, 72.0),
        "rain_opt": (450.0, 750.0),
        "ph_opt": (6.2, 7.5),
        "npk": (110, 55, 60),
        "irrigation": "Medium",
        "growing_period": "June - Oct / Nov - March / Feb - May",
        "duration": "110-140 days",
        "yield": "190-320 quintals/ha"
    },
    "Tomato": {
        "suitable_states": ["Andhra Pradesh", "Madhya Pradesh", "Karnataka", "Odisha", "Gujarat", "Maharashtra", "West Bengal", "Bihar", "Tamil Nadu", "Himachal Pradesh"],
        "seasons": ["Kharif", "Rabi", "Zaid"],
        "soils": ["Loamy soil", "Red soil", "Black soil", "Alluvial soil"],
        "temp_opt": (18.0, 30.0),
        "temp_min": 12.0,
        "temp_max": 35.0,
        "humidity_opt": (52.0, 76.0),
        "rain_opt": (500.0, 850.0),
        "ph_opt": (6.0, 7.2),
        "npk": (120, 60, 70),
        "irrigation": "Medium",
        "growing_period": "June - Oct / Oct - Feb / Feb - June",
        "duration": "100-130 days",
        "yield": "260-460 quintals/ha"
    },
    "Banana": {
        "suitable_states": ["Tamil Nadu", "Maharashtra", "Gujarat", "Andhra Pradesh", "Kerala", "Karnataka", "Bihar", "Assam", "West Bengal", "Odisha"],
        "seasons": ["Kharif", "Zaid"],
        "soils": ["Loamy soil", "Clay soil", "Alluvial soil", "Red soil"],
        "temp_opt": (22.0, 36.0),
        "temp_min": 16.0,
        "temp_max": 40.0,
        "humidity_opt": (65.0, 88.0),
        "rain_opt": (1200.0, 2200.0),
        "ph_opt": (6.0, 7.5),
        "npk": (200, 60, 250),
        "irrigation": "Very High",
        "growing_period": "June/July - May (Annual crop)",
        "duration": "300-365 days",
        "yield": "450-750 quintals/ha"
    },
    "Mango": {
        "suitable_states": ["Uttar Pradesh", "Andhra Pradesh", "Karnataka", "Bihar", "Gujarat", "Maharashtra", "West Bengal", "Odisha", "Kerala", "Tamil Nadu", "Telangana"],
        "seasons": ["Zaid", "Kharif"],
        "soils": ["Loamy soil", "Alluvial soil", "Red soil", "Laterite soil"],
        "temp_opt": (23.0, 38.0),
        "temp_min": 15.0,
        "temp_max": 42.0,
        "humidity_opt": (45.0, 75.0),
        "rain_opt": (750.0, 1500.0),
        "ph_opt": (5.5, 7.5),
        "npk": (90, 45, 80),
        "irrigation": "Medium",
        "growing_period": "Perennial (Bloom: Jan-March, Harvest: April-July)",
        "duration": "Perennial Orchard",
        "yield": "85-160 quintals/ha"
    },
    "Coconut": {
        "suitable_states": ["Kerala", "Tamil Nadu", "Karnataka", "Andhra Pradesh", "Odisha", "West Bengal", "Maharashtra", "Assam"],
        "seasons": ["Kharif", "Rabi", "Zaid"],
        "soils": ["Laterite soil", "Sandy soil", "Alluvial soil", "Red soil"],
        "temp_opt": (24.0, 34.0),
        "temp_min": 20.0,
        "temp_max": 36.0,
        "humidity_opt": (70.0, 90.0),
        "rain_opt": (1300.0, 2600.0),
        "ph_opt": (5.2, 7.8),
        "npk": (70, 40, 130),
        "irrigation": "High",
        "growing_period": "Perennial Plantation (Continuous bearing)",
        "duration": "Perennial Plantation",
        "yield": "80-120 nuts/palm/yr"
    },
    "Tea": {
        "suitable_states": ["Assam", "West Bengal", "Himachal Pradesh", "Kerala", "Tamil Nadu", "Karnataka"],
        "seasons": ["Kharif", "Rabi"],
        "soils": ["Laterite soil", "Red soil", "Loamy soil"],
        "temp_opt": (16.0, 28.0),
        "temp_min": 10.0,
        "temp_max": 32.0,
        "humidity_opt": (75.0, 94.0),
        "rain_opt": (1600.0, 3000.0),
        "ph_opt": (4.5, 5.8),
        "npk": (120, 35, 60),
        "irrigation": "High",
        "growing_period": "March - Nov (Flush picking cycles)",
        "duration": "Perennial Plantation",
        "yield": "16-26 quintals/ha (Made Tea)"
    },
    "Coffee": {
        "suitable_states": ["Karnataka", "Kerala", "Tamil Nadu", "Andhra Pradesh", "Odisha", "Assam"],
        "seasons": ["Kharif", "Rabi"],
        "soils": ["Laterite soil", "Red soil", "Loamy soil"],
        "temp_opt": (16.0, 28.0),
        "temp_min": 12.0,
        "temp_max": 32.0,
        "humidity_opt": (70.0, 90.0),
        "rain_opt": (1500.0, 2500.0),
        "ph_opt": (5.0, 6.5),
        "npk": (100, 50, 75),
        "irrigation": "Medium",
        "growing_period": "Bloom: March-April, Harvest: Nov-Feb",
        "duration": "Perennial Plantation",
        "yield": "10-18 quintals/ha"
    },
    "Jute": {
        "suitable_states": ["West Bengal", "Bihar", "Assam", "Odisha", "Uttar Pradesh"],
        "seasons": ["Kharif"],
        "soils": ["Alluvial soil", "Clay soil", "Loamy soil"],
        "temp_opt": (24.0, 36.0),
        "temp_min": 20.0,
        "temp_max": 38.0,
        "humidity_opt": (72.0, 92.0),
        "rain_opt": (1200.0, 2000.0),
        "ph_opt": (6.0, 7.4),
        "npk": (75, 40, 50),
        "irrigation": "High",
        "growing_period": "March/April - July/Aug",
        "duration": "120-140 days",
        "yield": "25-36 quintals/ha"
    }
}

def generate_agricultural_dataset(num_samples_target=3200):
    rows = []
    
    # We will generate a balanced, realistic distribution across all crops and regions
    crops = list(CROP_PROFILES.keys())
    
    samples_per_crop = num_samples_target // len(crops)
    
    for crop_name in crops:
        profile = CROP_PROFILES[crop_name]
        suitable_states = profile["suitable_states"]
        seasons = profile["seasons"]
        soils = profile["soils"]
        
        for _ in range(samples_per_crop + random.randint(5, 20)):
            # Pick a state suitable for this crop
            state = random.choice(suitable_states)
            state_info = STATE_DISTRICT_MAP[state]
            district = random.choice(state_info["districts"])
            
            # Select soil: prefer intersection of crop soils and state soils, fallback to crop soil
            common_soils = [s for s in soils if s in state_info["soils"]]
            soil = random.choice(common_soils if common_soils else soils)
            
            # Season
            season = random.choice(seasons)
            
            # Regional seasonal temperature adjustment
            base_t_min, base_t_max = state_info["base_temp"]
            crop_t_min, crop_t_max = profile["temp_opt"]
            
            # Blended temperature based on season and crop requirement
            if season == "Rabi":
                temp = random.gauss((crop_t_min + base_t_min) / 2.0, 2.0)
            elif season == "Zaid":
                temp = random.gauss((crop_t_max + base_t_max) / 2.0, 2.5)
            else: # Kharif
                temp = random.gauss((crop_t_min + crop_t_max) / 2.0, 2.0)
                
            temp = round(max(profile["temp_min"] - 2.0, min(profile["temp_max"] + 3.0, temp)), 1)
            min_temp = round(temp - random.uniform(4.0, 9.0), 1)
            max_temp = round(temp + random.uniform(4.0, 9.0), 1)
            
            # Humidity
            h_min, h_max = profile["humidity_opt"]
            humidity = round(random.gauss((h_min + h_max) / 2.0, 4.0), 1)
            humidity = max(30.0, min(95.0, humidity))
            
            # Rainfall
            r_min, r_max = profile["rain_opt"]
            state_r_min, state_r_max = state_info["base_rain"]
            mean_rain = (r_min * 0.6 + ((state_r_min + state_r_max) / 2.0) * 0.4)
            rainfall = round(random.gauss(mean_rain, (r_max - r_min) * 0.12), 1)
            rainfall = max(200.0, min(3500.0, rainfall))
            
            # Soil pH
            ph_min, ph_max = profile["ph_opt"]
            soil_ph = round(random.gauss((ph_min + ph_max) / 2.0, 0.35), 2)
            soil_ph = max(4.2, min(8.8, soil_ph))
            
            # NPK with realistic field variations (+- 15%)
            opt_n, opt_p, opt_k = profile["npk"]
            n_val = round(random.gauss(opt_n, opt_n * 0.12), 1)
            p_val = round(random.gauss(opt_p, opt_p * 0.12), 1)
            k_val = round(random.gauss(opt_k, opt_k * 0.12), 1)
            n_val = max(10.0, n_val)
            p_val = max(10.0, p_val)
            k_val = max(10.0, k_val)
            
            row = {
                "State": state,
                "District or Region": district,
                "Soil Type": soil,
                "Crop Name": crop_name,
                "Season": season,
                "Temperature": temp,
                "Minimum Temperature": min_temp,
                "Maximum Temperature": max_temp,
                "Humidity": humidity,
                "Rainfall": rainfall,
                "Soil pH": soil_ph,
                "Nitrogen (N)": n_val,
                "Phosphorus (P)": p_val,
                "Potassium (K)": k_val,
                "Irrigation Requirement": profile["irrigation"],
                "Suitable Growing Period": profile["growing_period"],
                "Crop Duration": profile["duration"],
                "Expected Yield": profile["yield"]
            }
            rows.append(row)
            
    df = pd.DataFrame(rows)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Generated {len(df)} crop recommendation records at: {DATASET_PATH}")
    return df

def generate_crop_information_catalog():
    records = []
    
    categories = {
        "Rice": "Cereal Crop",
        "Wheat": "Cereal Crop",
        "Maize": "Cereal & Fodder Crop",
        "Cotton": "Commercial Fiber Crop",
        "Sugarcane": "Commercial Cash Crop",
        "Soybean": "Oilseed & Legume",
        "Groundnut": "Oilseed Crop",
        "Mustard": "Rabi Oilseed Crop",
        "Chickpea": "Rabi Pulse (Bengal Gram)",
        "Pigeon Pea": "Kharif Pulse (Red Gram / Arhar)",
        "Jowar": "Millet / Coarse Cereal",
        "Bajra": "Nutri-Cereal / Pearl Millet",
        "Ragi": "Nutri-Cereal / Finger Millet",
        "Pulses": "Short-duration Pulses (Green/Black Gram)",
        "Potato": "Horticultural Tuber Crop",
        "Onion": "High-value Vegetable Crop",
        "Tomato": "Commercial Solanaceous Vegetable",
        "Banana": "Tropical Horticultural Fruit",
        "Mango": "Perennial Fruit Crop (King of Fruits)",
        "Coconut": "Coastal Plantation Crop",
        "Tea": "Hill Slope Plantation Beverage",
        "Coffee": "Shade-grown Hill Plantation Crop",
        "Jute": "Golden Fiber Commercial Crop"
    }
    
    cultivation_guides = {
        "Rice": "Direct seeding or transplanting 20-25 days old seedlings in puddled soil. Maintain 2-5 cm standing water during vegetative stages. Drain water 10 days before harvest.",
        "Wheat": "Requires well-pulverized seedbed. Sow in lines (20-22 cm apart) with seed-cum-fertilizer drill. First irrigation (CRI stage) at 21 days after sowing is critical.",
        "Cotton": "Thrives on deep black cotton soil (regur) with good internal drainage. Dibble seeds at 90x60 cm. Monitor for bollworm and sucking pests using yellow sticky traps.",
        "Sugarcane": "Plant three-budded setts in furrows 90 cm apart. Requires frequent irrigations and trash mulching to conserve moisture. Earthing up prevents lodging.",
        "Soybean": "Inoculate seeds with Bradyrhizobium japonicum culture before sowing. Requires well-drained fertile loam; extremely sensitive to waterlogging.",
        "Groundnut": "Requires light sandy loam with friable texture for easy peg penetration. Apply gypsum at 200-400 kg/ha at pegging stage to enhance pod filling.",
        "Mustard": "Conserves winter moisture well. Apply balanced NPK with elemental sulfur (20-30 kg/ha) to boost seed oil content. Guard against aphid attacks during cloudy weather.",
        "Chickpea": "Deep taproot system suitable for drylands. Avoid excessive moisture which causes wilt and root rot. Treat seeds with Trichoderma and Rhizobium.",
        "Pigeon Pea": "Deep rooted drought-tolerant pulse. Excellent for intercropping with soybean, cotton, or millets. Nipping apical bud at 45 days induces profuse branching.",
        "Jowar": "Highly drought-resistant C4 plant. Tolerates moderate salinity. Ideal for arid and semi-arid tracts of Deccan plateau.",
        "Bajra": "Remarkable tolerance to heat and drought. Can be grown in marginal sandy soils of Rajasthan and Gujarat with very low water input.",
        "Ragi": "Rich in calcium and dietary fiber. Hardy crop resisting dry spells, grown across red soils of Southern Karnataka and Odisha hills.",
        "Pulses": "Short duration catch crop fitting neatly between Kharif and Rabi rotations. Fixes atmospheric nitrogen, improving organic soil fertility.",
        "Potato": "Needs well-aerated loose sandy loam. Plant disease-free certified seed tubers with sprouts upward. Regular earthing up protects tubers from direct sunlight.",
        "Onion": "Transplant 6-8 week old nursery seedlings. Sensitive to weed competition in early stages. Stop irrigation 15 days prior to harvest to prevent rotting.",
        "Tomato": "Stake plants for indeterminate varieties. Provide drip fertigation for high fruit quality. Prune suckers to channel energy into fruit sizing.",
        "Banana": "Propagated using tissue culture plants or disease-free sword suckers. High potassium feeder. Requires windbreaks and drip irrigation for optimal bunch weight.",
        "Mango": "Prefers well-drained deep alluvial or loamy soils. Prune dead wood after harvest. Regulate flowering through paclobutrazol where recommended by SAUs.",
        "Coconut": "Plant seed palms in pits filled with red earth and compost. Mulch basin with coconut husks. Requires crown cleaning and periodic micronutrient spraying.",
        "Tea": "Cultivated on acidic hill slopes (pH 4.5-5.5) under partial shade. Regular plucking of 'two leaves and a bud' ensures premium commercial quality.",
        "Coffee": "Requires shade canopy (two-tier shade system), humid mist, and well-distributed rainfall. Arabica prefers higher elevations (1000m+) than Robusta.",
        "Jute": "Grown in fertile Indo-Gangetic alluvial plains. Retting in clean slow-moving water for 15-20 days is crucial for extracting golden lustrous fiber."
    }

    pest_management = {
        "Rice": "Yellow stem borer, brown planthopper (BPH), blast disease. Use pheromone traps and resistant cultivars.",
        "Wheat": "Yellow rust, brown rust, loose smut. Treat seed with Carboxin/Thiram; spray Tebuconazole if rust appears.",
        "Cotton": "Pink bollworm, whitefly, aphids. Implement IPM, avoid synthetic pyrethroids early, use neem seed kernel extract.",
        "Sugarcane": "Early shoot borer, top borer, red rot. Use trichogramma cards and plant disease-resistant varieties (Co 0238, Co 86032).",
        "Soybean": "Girdle beetle, semilooper, yellow mosaic virus. Timely weed control and seed treatment with Imidacloprid.",
        "Groundnut": "Tikka leaf spot, collar rot, white grub. Seed treatment with Trichoderma and timely Chlorpyrifos soil drenching.",
        "Mustard": "Mustard aphid, white rust, Alternaria blight. Spray Dimethoate or neem oil when aphid colony reaches ETL.",
        "Chickpea": "Gram pod borer (Helicoverpa armigera), Fusarium wilt. Install pheromone traps and spray HaNPV or bio-pesticides.",
        "Pigeon Pea": "Pod fly, pod borer, sterility mosaic. Intercrop with sorghum or maize and apply targeted sprays at 50% flowering.",
        "Jowar": "Shoot fly, stem borer. Early sowing with higher seed rate and seed treatment with Chlorpyrifos.",
        "Bajra": "Downy mildew (green ear), ergot. Use certified hybrid seeds with metalaxyl treatment.",
        "Ragi": "Blast (leaf, neck, and finger blast). Spray Tricyclazole or Mancozeb on first appearance of spindle lesions.",
        "Pulses": "Yellow mosaic virus transmitted by whitefly, pod borer. Control whiteflies with yellow sticky traps and systemic sprays.",
        "Potato": "Late blight, early blight, bacterial wilt. Spray Mancozeb / Cymoxanil before rainy spells to prevent late blight.",
        "Onion": "Thrips, purple blotch, basal rot. Spray Fipronil or Profenofos and ensure adequate crop rotation.",
        "Tomato": "Tomato leaf curl virus (whitefly vector), fruit borer, early blight. Use nylon net nurseries and fruit borer pheromone lures.",
        "Banana": "Panama wilt (Fusarium), Sigatoka leaf spot, rhizome weevil. Dip suckers in Carbendazim and maintain good field drainage.",
        "Mango": "Mango hopper, fruit fly, anthracnose. Hang methyl eugenol pheromone traps and spray sulfur/carbendazim during flowering.",
        "Coconut": "Rhinoceros beetle, red palm weevil, root wilt. Place naphthalene balls in top leaf axils; inject bio-agents.",
        "Tea": "Tea mosquito bug, red spider mite, blister blight. Maintain shade balance and spray wettable sulfur or hexaconazole.",
        "Coffee": "Coffee berry borer, white stem borer, leaf rust (Hemileia vastatrix). Trace and burn infested plants; spray Bordeaux mixture.",
        "Jute": "Stem rot, yellow mite, semilooper. Ensure timely thinning, clean water retting, and avoid water stagnation in young crop."
    }

    for crop, prof in CROP_PROFILES.items():
        records.append({
            "Crop Name": crop,
            "Category": categories.get(crop, "Agricultural Crop"),
            "Suitable Indian States": ", ".join(prof["suitable_states"]),
            "Suitable Soil Types": ", ".join(prof["soils"]),
            "Optimal Temperature Range (°C)": f"{prof['temp_opt'][0]} - {prof['temp_opt'][1]}",
            "Optimal Rainfall Range (mm)": f"{prof['rain_opt'][0]} - {prof['rain_opt'][1]}",
            "Sowing & Harvesting Seasons": ", ".join(prof["seasons"]),
            "Crop Duration": prof["duration"],
            "Irrigation Requirements": prof["irrigation"],
            "Soil pH Range": f"{prof['ph_opt'][0]} - {prof['ph_opt'][1]}",
            "Standard NPK (kg/ha)": f"{prof['npk'][0]}:{prof['npk'][1]}:{prof['npk'][2]}",
            "Expected Yield": prof["yield"],
            "Cultivation Guide": cultivation_guides.get(crop, "Ensure proper soil preparation, balanced nutrition, and timely irrigation."),
            "Pest & Disease Advisory": pest_management.get(crop, "Adopt Integrated Pest Management (IPM) practices.")
        })
        
    df_info = pd.DataFrame(records)
    df_info.to_csv(INFO_PATH, index=False)
    print(f"Generated crop information catalog for {len(df_info)} crops at: {INFO_PATH}")
    return df_info

if __name__ == "__main__":
    generate_agricultural_dataset()
    generate_crop_information_catalog()
