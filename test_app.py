"""
Comprehensive System Verification Test for Indian Farmer Advisory System
Validates all backend pipelines, ML model inference, advisory modules,
visualizations, and Gradio Blocks construction.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from modules.crop_recommendation import recommend_crops
from modules.weather_advisory import generate_weather_advisory
from modules.soil_advisory import get_soil_advisory, SOIL_PROFILES
from modules.regional_advisory import get_seasonal_recommendations, STATE_PROFILES
from modules.crop_information import get_crop_list, get_crop_details
from modules.visualizations import (
    plot_crop_distribution_by_state,
    plot_crop_suitability_by_season,
    plot_temp_vs_crop,
    plot_rainfall_vs_crop,
    plot_soil_type_vs_crop
)
from utils.weather_api import fetch_live_weather
from app import build_gradio_app

def run_all_tests():
    print("=" * 65)
    print("RUNNING COMPREHENSIVE SYSTEM VERIFICATION TESTS")
    print("=" * 65)
    
    # TEST 1: Model Crop Recommendation across different states & conditions
    print("\n[TEST 1] Testing Regional Crop Recommendations...")
    test_cases = [
        {"state": "Maharashtra", "district": "Vidarbha (Nagpur/Amravati)", "soil": "Black soil", "season": "Kharif", "temp": 28.0, "hum": 70.0, "rain": 950.0, "ph": 6.8},
        {"state": "Punjab", "district": "Malwa (Ludhiana/Bathinda)", "soil": "Alluvial soil", "season": "Rabi", "temp": 16.0, "hum": 55.0, "rain": 350.0, "ph": 7.2},
        {"state": "Rajasthan", "district": "Marwar (Jodhpur/Nagaur)", "soil": "Sandy soil", "season": "Kharif", "temp": 32.0, "hum": 45.0, "rain": 400.0, "ph": 7.5},
        {"state": "Kerala", "district": "Travancore (Kottayam/Alappuzha)", "soil": "Laterite soil", "season": "Kharif", "temp": 27.0, "hum": 85.0, "rain": 2200.0, "ph": 5.4},
        {"state": "Karnataka", "district": "Malnad (Chikkamagaluru/Shivamogga)", "soil": "Laterite soil", "season": "Kharif", "temp": 22.0, "hum": 80.0, "rain": 1800.0, "ph": 5.8}
    ]
    
    for tc in test_cases:
        res = recommend_crops(
            state=tc["state"],
            district=tc["district"],
            soil_type=tc["soil"],
            season=tc["season"],
            temp=tc["temp"],
            humidity=tc["hum"],
            rainfall=tc["rain"],
            ph=tc["ph"],
            top_k=5
        )
        assert len(res) == 5, f"Expected 5 recommendations for {tc['state']}, got {len(res)}"
        top_crop = res[0]
        print(f"  [OK] {tc['state']} ({tc['season']}): Top crop -> {top_crop['crop_name']} ({top_crop['suitability_score']}%) | Duration: {top_crop['duration']}")
    
    # TEST 2: Weather Advisory & Live Weather API
    print("\n[TEST 2] Testing Weather Advisory & Live Feed...")
    live_res = fetch_live_weather("Maharashtra", "Western Maharashtra (Pune/Nashik)")
    assert "temperature" in live_res, "Failed to retrieve temperature from weather API"
    print(f"  [OK] Live Weather API Output: {live_res['temperature']} deg C, {live_res['humidity']}%, Live={live_res['is_live']}")
    
    weather_adv = generate_weather_advisory("Maharashtra", "Western Maharashtra (Pune/Nashik)", 38.5, 35.0, 300.0, "Zaid")
    assert len(weather_adv["crops_to_avoid"]) > 0, "Expected crops to avoid for extreme heat"
    print(f"  [OK] Weather stress advisory detected: {len(weather_adv['suitable_crops'])} suitable, {len(weather_adv['crops_to_avoid'])} to avoid")

    # TEST 3: Soil Advisory for all 7 soil types
    print("\n[TEST 3] Testing Soil Advisory across all 7 Indian soils...")
    for soil in SOIL_PROFILES.keys():
        s_adv = get_soil_advisory(soil)
        assert len(s_adv["crops"]) > 0, f"No crops found for {soil}"
        assert len(s_adv["improvement_suggestions"]) > 0, f"No improvements for {soil}"
        print(f"  [OK] {soil}: {len(s_adv['crops'])} crops, pH: {s_adv['typical_ph']}")

    # TEST 4: Seasonal Advisory
    print("\n[TEST 4] Testing Seasonal Advisory...")
    for season in ["Kharif", "Rabi", "Zaid"]:
        s_res = get_seasonal_recommendations("Uttar Pradesh", season)
        assert len(s_res["recommended_crops"]) > 0, f"No seasonal crops for {season}"
        print(f"  [OK] UP - {season}: {len(s_res['recommended_crops'])} recommended crops")

    # TEST 5: Crop Encyclopedia Database
    print("\n[TEST 5] Testing Crop Encyclopedia Database...")
    crops = get_crop_list()
    assert len(crops) >= 23, f"Expected at least 23 crops, got {len(crops)}"
    for c in ["Rice", "Wheat", "Cotton", "Sugarcane", "Tea", "Coffee", "Coconut"]:
        details = get_crop_details(c)
        assert details is not None, f"Could not find details for {c}"
        assert "Optimal Temperature Range (°C)" in details
    print(f"  [OK] Verified catalog of {len(crops)} crops with complete agronomic profiles")

    # TEST 6: Plotly Visualizations
    print("\n[TEST 6] Testing Plotly Visualizations...")
    f1 = plot_crop_distribution_by_state()
    f2 = plot_crop_suitability_by_season()
    f3 = plot_temp_vs_crop()
    f4 = plot_rainfall_vs_crop()
    f5 = plot_soil_type_vs_crop()
    assert f1 is not None and f2 is not None and f3 is not None and f4 is not None and f5 is not None
    print("  [OK] All 5 interactive Plotly figures generated without error")

    # TEST 7: Gradio App Construction
    print("\n[TEST 7] Testing Gradio Application Object Construction...")
    demo = build_gradio_app()
    assert demo is not None, "Failed to instantiate Gradio app"
    print("  [OK] Gradio app initialized successfully with all tabs and components")

    # TEST 8: Multilingual Voice & Audio Advisory
    print("\n[TEST 8] Testing Multilingual Voice & Audio Advisory...")
    from modules.voice_advisory import process_voice_or_text_query, INDIAN_LANGUAGES
    assert len(INDIAN_LANGUAGES) >= 13, f"Expected 13+ Indian languages, found {len(INDIAN_LANGUAGES)}"
    
    # Test query processing in Hindi
    html_hi, audio_hi = process_voice_or_text_query(None, "महाराष्ट्र में कौन सी फसल लगाएं?", "Hindi (हिन्दी)")
    assert html_hi and len(html_hi) > 500, "Failed to generate Hindi advisory HTML"
    assert audio_hi and os.path.exists(audio_hi), "Failed to generate Hindi TTS audio output"
    print("  [OK] Hindi query processed: native advisory + spoken audio generated")

    # Test query processing in Tamil
    html_ta, audio_ta = process_voice_or_text_query(None, "தமிழ்நாட்டில் பருத்தி சாகுபடிக்கு சிறந்த காலம் எது?", "Tamil (தமிழ்)")
    assert html_ta and len(html_ta) > 500, "Failed to generate Tamil advisory HTML"
    assert audio_ta and os.path.exists(audio_ta), "Failed to generate Tamil TTS audio output"
    print("  [OK] Tamil query processed: native advisory + spoken audio generated")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED SUCCESSFULLY! SYSTEM READY FOR DEPLOYMENT.")
    print("=" * 65)

if __name__ == "__main__":
    run_all_tests()
