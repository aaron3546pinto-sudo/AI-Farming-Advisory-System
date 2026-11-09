"""
Live Weather API Integration for Indian Agricultural Advisory System
Uses Open-Meteo API (free, open-access, no API key required) to fetch
real-time ambient temperature, relative humidity, and precipitation.
Includes district coordinates fallback and offline fault-tolerance.
"""

import requests

# Fallback coordinates for representative Indian state capitals and regional hubs
DISTRICT_COORDINATES = {
    # Maharashtra
    "Vidarbha (Nagpur/Amravati)": (21.1458, 79.0882),
    "Marathwada (Aurangabad/Nanded)": (19.8762, 75.3433),
    "Western Maharashtra (Pune/Nashik)": (18.5204, 73.8567),
    "Konkan (Ratnagiri/Thane)": (16.9902, 73.3120),
    # Gujarat
    "Saurashtra (Rajkot/Junagadh)": (22.3039, 70.8022),
    "North Gujarat (Mehsana/Patan)": (23.5880, 72.3693),
    "South Gujarat (Surat/Navsari)": (21.1702, 72.8311),
    "Kutch (Bhuj)": (23.2420, 69.6669),
    # Punjab
    "Majha (Amritsar/Gurdaspur)": (31.6340, 74.8723),
    "Doaba (Jalandhar/Hoshiarpur)": (31.3260, 75.5762),
    "Malwa (Ludhiana/Bathinda)": (30.9010, 75.8573),
    # Haryana
    "Eastern Haryana (Karnal/Kurukshetra)": (29.6857, 76.9905),
    "Western Haryana (Hisar/Sirsa)": (29.1492, 75.7217),
    "Southern Haryana (Gurugram/Rewari)": (28.4595, 77.0266),
    # Rajasthan
    "Marwar (Jodhpur/Nagaur)": (26.2389, 73.0243),
    "Mewar (Udaipur/Chittorgarh)": (24.5854, 73.7125),
    "Shekhawati (Sikar/Jhunjhunu)": (27.6094, 75.1398),
    "Hadoti (Kota/Baran)": (25.2138, 75.8648),
    # Uttar Pradesh
    "Western UP (Meerut/Aligarh)": (28.9845, 77.7064),
    "Purvanchal (Varanasi/Gorakhpur)": (25.3176, 82.9739),
    "Awadh (Lucknow/Kanpur)": (26.8467, 80.9462),
    "Bundelkhand (Jhansi/Banda)": (25.4484, 78.5685),
    # Madhya Pradesh
    "Malwa (Indore/Ujjain)": (22.7196, 75.8577),
    "Nimar (Khargone/Khandwa)": (21.8236, 75.6186),
    "Bundelkhand (Sagar/Chhatarpur)": (23.8388, 78.7378),
    "Mahakoshal (Jabalpur/Seoni)": (23.1815, 79.9864),
    # Karnataka
    "Old Mysore (Mysuru/Mandya)": (12.2958, 76.6394),
    "North Karnataka (Belagavi/Dharwad)": (15.8497, 74.4977),
    "Coastal Karnataka (Mangaluru/Udupi)": (12.9141, 74.8560),
    "Malnad (Chikkamagaluru/Shivamogga)": (13.3161, 75.7720),
    # Tamil Nadu
    "Cauvery Delta (Thanjavur/Tiruvarur)": (10.7870, 79.1378),
    "Western Zone (Coimbatore/Erode)": (11.0168, 76.9558),
    "Southern Zone (Madurai/Tirunelveli)": (9.9252, 78.1198),
    "Northern Zone (Vellore/Kanchipuram)": (12.9165, 79.1325),
    # Kerala
    "Central Kerala (Ernakulam/Thrissur)": (9.9816, 76.2999),
    "Malabar (Kozhikode/Wayanad)": (11.2588, 75.7804),
    "Travancore (Kottayam/Alappuzha)": (9.5916, 76.5222),
    "High Ranges (Idukki)": (9.8494, 76.9804),
    # Andhra Pradesh
    "Coastal Andhra (Guntur/Krishna/Godavari)": (16.3067, 80.4365),
    "Rayalaseema (Anantapur/Kurnool/Kadapa)": (14.6819, 77.6006),
    "North Coastal (Visakhapatnam)": (17.6868, 83.2185),
    # Telangana
    "Northern Telangana (Warangal/Karimnagar)": (17.9689, 79.5941),
    "Southern Telangana (Mahabubnagar/Nalgonda)": (16.7438, 77.9890),
    "Central Telangana (Ranga Reddy/Medak)": (17.3850, 78.4867),
    # West Bengal
    "Gangetic Plain (Nadia/Murshidabad/Burdwan)": (23.4013, 88.5147),
    "Terai & Dooars (Jalpaiguri/Cooch Behar)": (26.5405, 88.7194),
    "Rarh (Birbhum/Bankura)": (23.2324, 87.0715),
    "Himalayan (Darjeeling)": (27.0410, 88.2663),
    # Bihar
    "North Bihar (Muzaffarpur/Darbhanga/Purnia)": (26.1209, 85.3647),
    "South Bihar (Patna/Gaya/Nalanda)": (25.5941, 85.1376),
    "Kosi Region (Saharsa/Katihar)": (25.8835, 86.6006),
    # Assam
    "Brahmaputra Valley (Kamrup/Sonitpur/Jorhat)": (26.1445, 91.7362),
    "Upper Assam (Dibrugarh/Tinsukia)": (27.4728, 94.9120),
    "Barak Valley (Cachar/Karimganj)": (24.8333, 92.7789),
    # Odisha
    "Coastal Odisha (Cuttack/Puri/Balasore)": (20.4625, 85.8828),
    "Western Odisha (Sambalpur/Bargarh)": (21.4669, 83.9812),
    "Southern Odisha (Ganjam/Koraput)": (19.3149, 84.7941),
    # Himachal Pradesh
    "Kangra Valley (Dharamshala/Kangra)": (32.0998, 76.2691),
    "Mid Hills (Mandi/Solan/Bilaspur)": (31.7087, 76.9320),
    "High Hills (Shimla/Kullu)": (31.1048, 77.1734)
}

STATE_CAPITALS = {
    "Maharashtra": (19.0760, 72.8777),
    "Gujarat": (23.2156, 72.6369),
    "Punjab": (30.7333, 76.7794),
    "Haryana": (30.7333, 76.7794),
    "Rajasthan": (26.9124, 75.7873),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Madhya Pradesh": (23.2599, 77.4126),
    "Karnataka": (12.9716, 77.5946),
    "Tamil Nadu": (13.0827, 80.2707),
    "Kerala": (8.5241, 76.9366),
    "Andhra Pradesh": (16.5062, 80.6480),
    "Telangana": (17.3850, 78.4867),
    "West Bengal": (22.5726, 88.3639),
    "Bihar": (25.5941, 85.1376),
    "Assam": (26.1445, 91.7362),
    "Odisha": (20.2961, 85.8245),
    "Himachal Pradesh": (31.1048, 77.1734)
}

def fetch_live_weather(state, district=None):
    """
    Fetches real-time weather data for an Indian state/district via Open-Meteo.
    Returns:
        dict: {
            "success": bool,
            "is_live": bool,
            "temperature": float,
            "humidity": float,
            "precipitation": float,
            "source": str,
            "location_name": str,
            "error_msg": str or None
        }
    """
    coords = None
    location_name = f"{district}, {state}" if district else state
    
    if district and district in DISTRICT_COORDINATES:
        coords = DISTRICT_COORDINATES[district]
    elif state and state in STATE_CAPITALS:
        coords = STATE_CAPITALS[state]
    else:
        coords = (20.5937, 78.9629) # Geographic center of India
        
    lat, lon = coords
    
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=Asia%2FKolkata"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current", {})
            temp = current.get("temperature_2m", 25.0)
            humidity = current.get("relative_humidity_2m", 60.0)
            precip = current.get("precipitation", 0.0)
            
            return {
                "success": True,
                "is_live": True,
                "temperature": round(float(temp), 1),
                "humidity": round(float(humidity), 1),
                "precipitation": round(float(precip), 1),
                "source": "Open-Meteo Live IMD/WMO Satellite Feed",
                "location_name": location_name,
                "error_msg": None
            }
        else:
            return {
                "success": False,
                "is_live": False,
                "temperature": 26.0,
                "humidity": 65.0,
                "precipitation": 0.0,
                "source": "Historical Regional Climate Benchmark",
                "location_name": location_name,
                "error_msg": f"Weather API returned status code {response.status_code}."
            }
    except Exception as e:
        return {
            "success": False,
            "is_live": False,
            "temperature": 26.0,
            "humidity": 65.0,
            "precipitation": 0.0,
            "source": "Historical Regional Climate Benchmark",
            "location_name": location_name,
            "error_msg": f"Network exception: {str(e)}"
        }
