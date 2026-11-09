"""
AI FARMING ADVISORY – SMART AGRICULTURE FOR INDIA
A comprehensive agricultural decision support system powered by
Gradio, Scikit-learn, and Plotly.
Styled in a crisp Light Ice-Blue and Royal Blue / Dark Navy theme.
"""

import os
import json
import gradio as gr
import pandas as pd

# Import project modules
from modules.crop_recommendation import recommend_crops
from modules.weather_advisory import generate_weather_advisory
from modules.soil_advisory import get_soil_advisory, SOIL_PROFILES
from modules.regional_advisory import get_seasonal_recommendations, STATE_PROFILES
from modules.crop_information import get_crop_list, get_crop_details, search_crops_by_keyword
from modules.visualizations import (
    plot_crop_distribution_by_state,
    plot_crop_suitability_by_season,
    plot_temp_vs_crop,
    plot_rainfall_vs_crop,
    plot_soil_type_vs_crop
)
from utils.weather_api import fetch_live_weather, DISTRICT_COORDINATES
from modules.voice_advisory import process_voice_or_text_query, INDIAN_LANGUAGES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")

# State & District mapping
INDIAN_STATES = sorted(list(STATE_PROFILES.keys()))

STATE_DISTRICT_MAPPING = {
    "Maharashtra": ["Vidarbha (Nagpur/Amravati)", "Marathwada (Aurangabad/Nanded)", "Western Maharashtra (Pune/Nashik)", "Konkan (Ratnagiri/Thane)"],
    "Gujarat": ["Saurashtra (Rajkot/Junagadh)", "North Gujarat (Mehsana/Patan)", "South Gujarat (Surat/Navsari)", "Kutch (Bhuj)"],
    "Punjab": ["Majha (Amritsar/Gurdaspur)", "Doaba (Jalandhar/Hoshiarpur)", "Malwa (Ludhiana/Bathinda)"],
    "Haryana": ["Eastern Haryana (Karnal/Kurukshetra)", "Western Haryana (Hisar/Sirsa)", "Southern Haryana (Gurugram/Rewari)"],
    "Rajasthan": ["Marwar (Jodhpur/Nagaur)", "Mewar (Udaipur/Chittorgarh)", "Shekhawati (Sikar/Jhunjhunu)", "Hadoti (Kota/Baran)"],
    "Uttar Pradesh": ["Western UP (Meerut/Aligarh)", "Purvanchal (Varanasi/Gorakhpur)", "Awadh (Lucknow/Kanpur)", "Bundelkhand (Jhansi/Banda)"],
    "Madhya Pradesh": ["Malwa (Indore/Ujjain)", "Nimar (Khargone/Khandwa)", "Bundelkhand (Sagar/Chhatarpur)", "Mahakoshal (Jabalpur/Seoni)"],
    "Karnataka": ["Old Mysore (Mysuru/Mandya)", "North Karnataka (Belagavi/Dharwad)", "Coastal Karnataka (Mangaluru/Udupi)", "Malnad (Chikkamagaluru/Shivamogga)"],
    "Tamil Nadu": ["Cauvery Delta (Thanjavur/Tiruvarur)", "Western Zone (Coimbatore/Erode)", "Southern Zone (Madurai/Tirunelveli)", "Northern Zone (Vellore/Kanchipuram)"],
    "Kerala": ["Central Kerala (Ernakulam/Thrissur)", "Malabar (Kozhikode/Wayanad)", "Travancore (Kottayam/Alappuzha)", "High Ranges (Idukki)"],
    "Andhra Pradesh": ["Coastal Andhra (Guntur/Krishna/Godavari)", "Rayalaseema (Anantapur/Kurnool/Kadapa)", "North Coastal (Visakhapatnam)"],
    "Telangana": ["Northern Telangana (Warangal/Karimnagar)", "Southern Telangana (Mahabubnagar/Nalgonda)", "Central Telangana (Ranga Reddy/Medak)"],
    "West Bengal": ["Gangetic Plain (Nadia/Murshidabad/Burdwan)", "Terai & Dooars (Jalpaiguri/Cooch Behar)", "Rarh (Birbhum/Bankura)", "Himalayan (Darjeeling)"],
    "Bihar": ["North Bihar (Muzaffarpur/Darbhanga/Purnia)", "South Bihar (Patna/Gaya/Nalanda)", "Kosi Region (Saharsa/Katihar)"],
    "Assam": ["Brahmaputra Valley (Kamrup/Sonitpur/Jorhat)", "Upper Assam (Dibrugarh/Tinsukia)", "Barak Valley (Cachar/Karimganj)"],
    "Odisha": ["Coastal Odisha (Cuttack/Puri/Balasore)", "Western Odisha (Sambalpur/Bargarh)", "Southern Odisha (Ganjam/Koraput)"],
    "Himachal Pradesh": ["Kangra Valley (Dharamshala/Kangra)", "Mid Hills (Mandi/Solan/Bilaspur)", "High Hills (Shimla/Kullu)"]
}

SOIL_TYPES = list(SOIL_PROFILES.keys())
SEASONS = ["Kharif", "Rabi", "Zaid"]

# UI Callbacks
def on_state_change(state):
    districts = STATE_DISTRICT_MAPPING.get(state, ["General Region"])
    return gr.Dropdown(choices=districts, value=districts[0] if districts else None)

def fetch_weather_for_ui(state, district):
    res = fetch_live_weather(state, district)
    is_live = res["is_live"]
    status_html = f"""
    <div style='background: {'#eff6ff' if is_live else '#fffbeb'}; border: 2px solid {'#2563eb' if is_live else '#d97706'}; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px;'>
        <b style='color: {'#1d4ed8' if is_live else '#b45309'}; font-size: 1.1em;'>{'🛰️ Live Satellite Weather Connected' if is_live else '⚠️ Regional Climate Baseline'}</b><br>
        <span style='font-size: 0.98em; color: #000000; font-weight: 700;'>Source: <b>{res['source']}</b> | Location: <b>{res['location_name']}</b></span>
    </div>
    """
    return (
        res["temperature"],
        res["humidity"],
        res["precipitation"],
        status_html,
        res["is_live"],
        res["source"]
    )

def handle_crop_recommendation(state, district, soil_type, season, temp, humidity, rainfall, ph, n, p, k):
    try:
        results = recommend_crops(
            state=state,
            district=district,
            soil_type=soil_type,
            season=season,
            temp=temp,
            humidity=humidity,
            rainfall=rainfall,
            ph=ph,
            n=n if n > 0 else None,
            p=p if p > 0 else None,
            k=k if k > 0 else None,
            top_k=5
        )
        
        cards_html = "<div style='display: flex; flex-direction: column; gap: 16px; margin-top: 15px;'>"
        
        rank_badges = [
            ("🏆 Top Match (#1)", "#1e40af", "#dbeafe"),
            ("🥈 High Suitability (#2)", "#1d4ed8", "#eff6ff"),
            ("🥉 Viable Choice (#3)", "#0369a1", "#e0f2fe"),
            ("🌿 Alternative (#4)", "#4338ca", "#ede9fe"),
            ("🌾 Alternative (#5)", "#334155", "#f1f5f9")
        ]
        
        for idx, item in enumerate(results):
            badge_title, badge_color, badge_bg = rank_badges[idx] if idx < len(rank_badges) else ("Recommended", "#1d4ed8", "#eff6ff")
            score = item["suitability_score"]
            score_bar_color = "#1d4ed8" if score >= 80 else "#2563eb" if score >= 65 else "#64748b"
            
            card = f"""
            <div style='background: #ffffff; border: 2px solid #2563eb; border-radius: 12px; padding: 22px 26px; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);'>
                <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 14px;'>
                    <div>
                        <span style='background: {badge_bg}; color: {badge_color}; padding: 5px 14px; border-radius: 16px; font-weight: 800; font-size: 0.88em; border: 1.5px solid {badge_color}66;'>
                            {badge_title}
                        </span>
                        <h2 style='margin: 8px 0 2px 0; color: #1e3a8a; font-size: 1.7em; font-weight: 900; display: inline-block; vertical-align: middle; margin-left: 10px; letter-spacing: -0.5px;'>
                            {item['crop_name']}
                        </h2>
                        <span style='color: #1e293b; font-size: 0.95em; font-weight: 700; margin-left: 6px;'>({item['category']})</span>
                    </div>
                    <div style='text-align: right; min-width: 190px;'>
                        <div style='font-size: 0.88em; color: #1e293b; font-weight: 800;'>Estimated Model Suitability</div>
                        <div style='font-size: 1.6em; font-weight: 900; color: {score_bar_color};'>{score}%</div>
                        <div style='background: #cbd5e1; border-radius: 6px; height: 10px; width: 100%; overflow: hidden; margin-top: 3px;'>
                            <div style='background: {score_bar_color}; width: {score}%; height: 100%; border-radius: 6px;'></div>
                        </div>
                    </div>
                </div>
                
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; background: #f8fafc; padding: 14px; border-radius: 8px; border: 1.5px solid #cbd5e1; margin-bottom: 14px;'>
                    <div><b style='color: #1e3a8a; font-size: 0.9em; font-weight: 800;'>Best Season:</b><br><span style='font-size: 1em; color: #000000; font-weight: 800;'>{item['season']}</span></div>
                    <div><b style='color: #1e3a8a; font-size: 0.9em; font-weight: 800;'>Duration:</b><br><span style='font-size: 1em; color: #000000; font-weight: 800;'>{item['duration']}</span></div>
                    <div><b style='color: #1e3a8a; font-size: 0.9em; font-weight: 800;'>Ideal Temp:</b><br><span style='font-size: 1em; color: #000000; font-weight: 800;'>{item['temp_range']} °C</span></div>
                    <div><b style='color: #1e3a8a; font-size: 0.9em; font-weight: 800;'>Rainfall Req:</b><br><span style='font-size: 1em; color: #000000; font-weight: 800;'>{item['rainfall_range']} mm</span></div>
                    <div><b style='color: #1e3a8a; font-size: 0.9em; font-weight: 800;'>Irrigation Need:</b><br><span style='font-size: 1em; color: #000000; font-weight: 800;'>{item['irrigation']}</span></div>
                    <div><b style='color: #1e3a8a; font-size: 0.9em; font-weight: 800;'>Expected Yield:</b><br><span style='font-size: 1em; color: #000000; font-weight: 800;'>{item['yield']}</span></div>
                </div>
                
                <div style='background: #eff6ff; border-left: 6px solid #2563eb; border: 1.5px solid #bfdbfe; border-left-width: 6px; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 12px;'>
                    <b style='color: #1e3a8a; font-size: 1.05em; font-weight: 800;'>Why this crop is suitable for your region:</b>
                    <div style='font-size: 0.98em; color: #000000; line-height: 1.6; margin-top: 6px; font-weight: 600;'>
                        {item['explanation']}
                    </div>
                </div>
                
                <div style='font-size: 0.95em; color: #000000; background: #f0f9ff; border: 2px solid #0284c7; padding: 12px 16px; border-radius: 8px; font-weight: 600;'>
                    💡 <b style='color: #0369a1; font-weight: 800;'>Cultivation Tip:</b> {item['cultivation_tip']}
                </div>
            </div>
            """
            cards_html += card
            
        cards_html += """
        <div style='margin-top: 10px; padding: 10px; font-size: 0.85em; color: #475569; text-align: center; font-weight: 500;'>
            * Suitability score represents a statistical machine learning estimate combined with agro-climatic boundaries. Soil testing is recommended.
        </div>
        </div>
        """
        return cards_html
    except Exception as e:
        return f"<div style='color: #991b1b; background: #fef2f2; border: 1.5px solid #f87171; padding: 15px; border-radius: 8px; font-weight: bold;'>Error generating recommendation: {str(e)}</div>"

def handle_weather_advisory(state, district, temp, humidity, rainfall, season, is_live, source_label):
    try:
        adv = generate_weather_advisory(
            state=state,
            district=district,
            temp=temp,
            humidity=humidity,
            rainfall=rainfall,
            season=season,
            is_live=is_live,
            source_label=source_label
        )
        
        # Build HTML summary
        suitable_cards = "".join([
            f"<div style='background: #f0fdf4; border: 2px solid #22c55e; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;'>"
            f"<b style='color: #15803d; font-size: 1.15em; font-weight: 800;'>{c['crop']}</b> <span style='color: #0284c7; font-size: 0.88em; font-weight: 700;'>({c['category']})</span><br>"
            f"<span style='font-size: 0.94em; color: #000000; font-weight: 600;'>Water Need: <b>{c['water_need']}</b> | Duration: <b>{c['duration']}</b></span><br>"
            f"<span style='font-size: 0.92em; color: #166534; font-weight: 700;'>{c['reason']}</span>"
            f"</div>"
            for c in adv["suitable_crops"]
        ])
        
        avoid_cards = "".join([
            f"<div style='background: #fef2f2; border: 2px solid #ef4444; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;'>"
            f"<b style='color: #b91c1c; font-size: 1.15em; font-weight: 800;'>⚠️ {c['crop']}</b><br>"
            f"<span style='font-size: 0.92em; color: #000000; font-weight: 600;'>{c['reason']}</span>"
            f"</div>"
            for c in adv["crops_to_avoid"]
        ]) if adv["crops_to_avoid"] else "<div style='color: #15803d; font-size: 1em; font-weight: 700;'>✅ No extreme weather deterrents detected for common regional crops.</div>"
        
        precautions_html = "".join([f"<li style='margin-bottom: 8px; color: #c2410c; font-size: 0.96em; font-weight: 700;'>{p}</li>" for p in adv["precautions"]])
        water_html = "".join([f"<li style='margin-bottom: 8px; color: #0369a1; font-size: 0.96em; font-weight: 700;'>{w}</li>" for w in adv["water_advisory"]])
        windows_html = "".join([f"<li style='margin-bottom: 8px; color: #1e40af; font-size: 0.96em; font-weight: 700;'>{w}</li>" for w in adv["planting_windows"]])
        
        source_badge = f"<span style='background: {'#dbeafe' if is_live else '#fef3c7'}; color: {'#1e40af' if is_live else '#b45309'}; border: 1.5px solid {'#2563eb' if is_live else '#d97706'}; padding: 5px 14px; border-radius: 14px; font-size: 0.85em; font-weight: 800;'>{source_label}</span>"
        
        result_html = f"""
        <div style='background: #ffffff; border: 2px solid #2563eb; border-radius: 12px; padding: 22px; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);'>
            <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #cbd5e1; padding-bottom: 12px; margin-bottom: 16px;'>
                <div>
                    <h3 style='margin: 0; color: #1e3a8a; font-size: 1.5em; font-weight: 900;'>Weather-Guided Agro-Advisory</h3>
                    <div style='font-size: 0.98em; color: #000000; margin-top: 4px; font-weight: 600;'>Region: <b>{adv['location']}</b> | Cropping Season: <b>{season}</b></div>
                </div>
                <div>{source_badge}</div>
            </div>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
                <div>
                    <h4 style='color: #1e3a8a; margin-top: 0; margin-bottom: 12px; font-size: 1.2em; font-weight: 800;'>✅ Weather-Favorable Crops</h4>
                    {suitable_cards}
                </div>
                <div>
                    <h4 style='color: #b91c1c; margin-top: 0; margin-bottom: 12px; font-size: 1.2em; font-weight: 800;'>⛔ Crops to Avoid Under Weather Stress</h4>
                    {avoid_cards}
                </div>
            </div>
            
            <div style='margin-top: 20px; padding-top: 16px; border-top: 1.5px solid #cbd5e1;'>
                <h4 style='color: #0369a1; margin-top: 0; font-size: 1.2em; font-weight: 800;'>💧 Water & Irrigation Advisory</h4>
                <ul style='margin-top: 8px; padding-left: 20px;'>{water_html}</ul>
            </div>
            
            <div style='margin-top: 16px;'>
                <h4 style='color: #c2410c; margin-top: 0; font-size: 1.2em; font-weight: 800;'>⚠️ Agro-Meteorological Precautions & Alerts</h4>
                <ul style='margin-top: 8px; padding-left: 20px;'>{precautions_html}</ul>
            </div>
            
            <div style='margin-top: 16px;'>
                <h4 style='color: #1e40af; margin-top: 0; font-size: 1.2em; font-weight: 800;'>🗓️ Recommended Sowing Windows</h4>
                <ul style='margin-top: 8px; padding-left: 20px;'>{windows_html}</ul>
            </div>
        </div>
        """
        return result_html
    except Exception as e:
        return f"<div style='color: #991b1b; background: #fef2f2; border: 1.5px solid #f87171; padding: 15px; border-radius: 8px; font-weight: bold;'>Error analyzing weather: {str(e)}</div>"

def handle_soil_advisory(soil_type):
    try:
        adv = get_soil_advisory(soil_type)
        
        crops_html = "".join([
            f"<div style='background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 12px; margin-bottom: 10px;'>"
            f"<b style='color: #1e3a8a; font-size: 1.15em; font-weight: 800;'>{c['name']}</b> <span style='font-size: 0.9em; color: #0284c7; font-weight: 700;'>({c['category']})</span><br>"
            f"<span style='font-size: 0.94em; color: #000000; font-weight: 600;'>Duration: <b>{c['duration']}</b> | Water: <b>{c['irrigation']}</b> | pH: <b>{c['ph_range']}</b></span><br>"
            f"<span style='font-size: 0.94em; color: #1d4ed8; font-weight: 800;'>Expected Yield: {c['yield']}</span>"
            f"</div>"
            for c in adv["crops"]
        ])
        
        nutrient_rows = "".join([
            f"<tr><td style='padding: 10px 14px; border-bottom: 1.5px solid #cbd5e1; font-weight: 800; color: #1e3a8a; font-size: 0.95em;'>{k}</td>"
            f"<td style='padding: 10px 14px; border-bottom: 1.5px solid #cbd5e1; color: #000000; font-weight: 600; font-size: 0.95em;'>{v}</td></tr>"
            for k, v in adv["nutrient_status"].items()
        ])
        
        improvements_html = "".join([
            f"<li style='margin-bottom: 8px; color: #000000; font-size: 0.96em; font-weight: 600;'>{imp}</li>"
            for imp in adv["improvement_suggestions"]
        ])
        
        result_html = f"""
        <div style='background: #ffffff; border: 2px solid #2563eb; border-radius: 12px; padding: 24px; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);'>
            <div style='border-bottom: 2px solid #cbd5e1; padding-bottom: 14px; margin-bottom: 18px;'>
                <h2 style='color: #1e3a8a; margin: 0; font-size: 1.7em; font-weight: 900;'>{adv['soil_type']} Profile & Agronomic Guidance</h2>
                <p style='color: #000000; margin: 8px 0 0 0; font-size: 1.02em; line-height: 1.6; font-weight: 600;'>{adv['description']}</p>
                <div style='font-size: 0.96em; color: #1e3a8a; margin-top: 8px; font-weight: 800;'><b>Major Indian States:</b> {adv['dominant_regions']}</div>
            </div>
            
            <div style='display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 22px;'>
                <div>
                    <h4 style='color: #1e3a8a; margin-top: 0; font-size: 1.2em; font-weight: 800;'>🧪 Typical Nutrient Status & Soil Chemistry</h4>
                    <div style='background: #dbeafe; border: 2px solid #2563eb; color: #1e40af; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; font-weight: 800; font-size: 1.02em;'>
                        Optimal Soil pH Range: {adv['typical_ph']}
                    </div>
                    <table style='width: 100%; border-collapse: collapse; font-size: 0.95em; background: #ffffff; border: 1.5px solid #cbd5e1; border-radius: 6px;'>
                        <thead>
                            <tr style='background: #1e3a8a; text-align: left;'>
                                <th style='padding: 10px 14px; color: #ffffff !important; font-weight: 900;'>Nutrient</th>
                                <th style='padding: 10px 14px; color: #ffffff !important; font-weight: 900;'>Status & Field Dynamics</th>
                            </tr>
                        </thead>
                        <tbody>
                            {nutrient_rows}
                        </tbody>
                    </table>
                    
                    <h4 style='color: #0284c7; margin-top: 18px; margin-bottom: 8px; font-size: 1.2em; font-weight: 800;'>💧 Irrigation Guidance</h4>
                    <p style='font-size: 0.96em; color: #000000; background: #eff6ff; border: 1.5px solid #93c5fd; padding: 14px; border-radius: 8px; margin: 0; font-weight: 600; line-height: 1.5;'>
                        {adv['irrigation_guidance']}
                    </p>
                </div>
                
                <div>
                    <h4 style='color: #1e3a8a; margin-top: 0; font-size: 1.2em; font-weight: 800;'>🌾 Well-Adapted Crops for {adv['soil_type']}</h4>
                    <div style='max-height: 400px; overflow-y: auto;'>
                        {crops_html}
                    </div>
                </div>
            </div>
            
            <div style='margin-top: 22px; padding-top: 16px; border-top: 2px solid #cbd5e1;'>
                <h4 style='color: #1e3a8a; margin-top: 0; font-size: 1.2em; font-weight: 800;'>🌿 Soil Health Restoration & Improvement Strategies</h4>
                <ul style='margin-top: 8px; padding-left: 20px;'>
                    {improvements_html}
                </ul>
            </div>
        </div>
        """
        return result_html
    except Exception as e:
        return f"<div style='color: #991b1b; background: #fef2f2; border: 1.5px solid #f87171; padding: 15px; border-radius: 8px; font-weight: bold;'>Error retrieving soil advisory: {str(e)}</div>"

def handle_seasonal_advisory(state, season):
    try:
        adv = get_seasonal_recommendations(state, season)
        
        crops_html = "".join([
            f"<div style='background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 14px; margin-bottom: 10px;'>"
            f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
            f"<b style='color: #1e3a8a; font-size: 1.2em; font-weight: 800;'>{c['crop']}</b>"
            f"<span style='background: #dbeafe; color: #1e40af; font-size: 0.85em; font-weight: 800; padding: 4px 12px; border-radius: 12px; border: 1.5px solid #2563eb;'>{c['category']}</span>"
            f"</div>"
            f"<div style='font-size: 0.94em; color: #000000; margin-top: 6px; font-weight: 600;'>Duration: <b>{c['duration']}</b> | Water: <b>{c['irrigation']}</b> | Expected Yield: <b>{c['yield']}</b></div>"
            f"<div style='font-size: 0.92em; color: #1d4ed8; margin-top: 6px; font-weight: 700;'>💡 {c['guidance']}</div>"
            f"</div>"
            for c in adv["recommended_crops"]
        ])
        
        practices_html = "".join([f"<li style='margin-bottom: 8px; color: #000000; font-size: 0.96em; font-weight: 600;'>{p}</li>" for p in adv["key_practices"]])
        
        result_html = f"""
        <div style='background: #ffffff; border: 2px solid #2563eb; border-radius: 12px; padding: 24px; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);'>
            <div style='border-bottom: 2px solid #cbd5e1; padding-bottom: 14px; margin-bottom: 18px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <h2 style='color: #1e3a8a; margin: 0; font-size: 1.7em; font-weight: 900;'>{state} – {season} Cropping Strategy</h2>
                    <span style='background: #dbeafe; color: #1e40af; padding: 6px 16px; border-radius: 16px; font-weight: 800; font-size: 0.9em; border: 1.5px solid #2563eb;'>
                        {adv['season_period']}
                    </span>
                </div>
                <p style='color: #000000; margin: 10px 0 0 0; font-size: 1.02em; line-height: 1.6; font-weight: 600;'>{adv['weather_profile']}</p>
            </div>
            
            <div style='background: #eff6ff; border-left: 6px solid #2563eb; border: 1.5px solid #bfdbfe; border-left-width: 6px; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 20px;'>
                <b style='color: #1e3a8a; font-size: 1.05em; font-weight: 800;'>Agronomic Strategy for {season}:</b>
                <div style='font-size: 0.98em; color: #000000; margin-top: 6px; font-weight: 600; line-height: 1.5;'>{adv['cropping_strategy']}</div>
            </div>
            
            <div style='display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 22px;'>
                <div>
                    <h4 style='color: #1e3a8a; margin-top: 0; font-size: 1.2em; font-weight: 800;'>🌾 Recommended {season} Crops for {state}</h4>
                    {crops_html}
                </div>
                
                <div>
                    <h4 style='color: #1e3a8a; margin-top: 0; font-size: 1.2em; font-weight: 800;'>📋 Key Seasonal Management Protocols</h4>
                    <ul style='padding-left: 20px; margin-top: 8px;'>
                        {practices_html}
                    </ul>
                    
                    <div style='background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 14px; margin-top: 20px;'>
                        <h4 style='color: #1e3a8a; margin: 0 0 8px 0; font-size: 1.1em; font-weight: 900;'>🏛️ Regional Agro-Climatic Profile: {state}</h4>
                        <div style='font-size: 0.94em; color: #000000; line-height: 1.6; font-weight: 600;'>
                            <b>Agro-Ecological Zones:</b> {adv['state_agro_zones']}<br>
                            <b>Dominant Soil Types:</b> {adv['state_soils']}<br>
                            <b>Irrigation Infrastructure:</b> {adv['state_irrigation']}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        return result_html
    except Exception as e:
        return f"<div style='color: #991b1b; background: #fef2f2; border: 1.5px solid #f87171; padding: 15px; border-radius: 8px; font-weight: bold;'>Error retrieving seasonal advisory: {str(e)}</div>"

def handle_crop_details(crop_name):
    try:
        details = get_crop_details(crop_name)
        if not details:
            return "<div style='padding: 15px; color: #475569; font-weight: 600;'>Crop profile not found.</div>"
            
        return f"""
        <div style='background: #ffffff; border: 2px solid #2563eb; border-radius: 12px; padding: 24px; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);'>
            <div style='border-bottom: 2px solid #cbd5e1; padding-bottom: 14px; margin-bottom: 18px;'>
                <span style='background: #dbeafe; color: #1e40af; padding: 5px 16px; border-radius: 14px; font-weight: 800; font-size: 0.88em; border: 1.5px solid #2563eb;'>
                    {details.get('Category', 'Agricultural Crop')}
                </span>
                <h1 style='color: #1e3a8a; margin: 10px 0 4px 0; font-size: 2em; font-weight: 900;'>{details['Crop Name']}</h1>
                <div style='color: #000000; font-size: 0.98em; font-weight: 600;'>Growing Seasons: <b>{details.get('Sowing & Harvesting Seasons', 'N/A')}</b> | Duration: <b>{details.get('Crop Duration', 'N/A')}</b></div>
            </div>
            
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; background: #f8fafc; padding: 16px; border-radius: 8px; border: 1.5px solid #cbd5e1; margin-bottom: 20px;'>
                <div><b style='color: #1e3a8a; font-size: 0.9em; font-weight: 800;'>Optimal Temp:</b><br><span style='font-size: 1em; color: #000000; font-weight: 800;'>{details.get('Optimal Temperature Range (°C)', 'N/A')} °C</span></div>
                <div><b style='color: #1e3a8a; font-size: 0.9em; font-weight: 800;'>Rainfall Threshold:</b><br><span style='font-size: 1em; color: #000000; font-weight: 800;'>{details.get('Optimal Rainfall Range (mm)', 'N/A')} mm</span></div>
                <div><b style='color: #1e3a8a; font-size: 0.9em; font-weight: 800;'>Soil pH Range:</b><br><span style='font-size: 1em; color: #000000; font-weight: 800;'>{details.get('Soil pH Range', 'N/A')}</span></div>
                <div><b style='color: #1e3a8a; font-size: 0.9em; font-weight: 800;'>Standard NPK:</b><br><span style='font-size: 1em; color: #000000; font-weight: 800;'>{details.get('Standard NPK (kg/ha)', 'N/A')}</span></div>
                <div><b style='color: #1e3a8a; font-size: 0.9em; font-weight: 800;'>Irrigation:</b><br><span style='font-size: 1em; color: #000000; font-weight: 800;'>{details.get('Irrigation Requirements', 'N/A')}</span></div>
                <div><b style='color: #1e3a8a; font-size: 0.9em; font-weight: 800;'>Expected Yield:</b><br><span style='font-size: 1em; color: #000000; font-weight: 800;'>{details.get('Expected Yield', 'N/A')}</span></div>
            </div>
            
            <div style='margin-bottom: 18px;'>
                <h4 style='color: #1e3a8a; margin: 0 0 8px 0; font-size: 1.15em; font-weight: 800;'>📍 Major Producing States</h4>
                <p style='font-size: 0.98em; color: #000000; margin: 0; background: #ffffff; padding: 12px; border: 1.5px solid #cbd5e1; border-radius: 8px; font-weight: 600;'>
                    {details.get('Suitable Indian States', 'N/A')}
                </p>
            </div>
            
            <div style='margin-bottom: 18px;'>
                <h4 style='color: #1e3a8a; margin: 0 0 8px 0; font-size: 1.15em; font-weight: 800;'>🪴 Suitable Soil Types</h4>
                <p style='font-size: 0.98em; color: #000000; margin: 0; background: #ffffff; padding: 12px; border: 1.5px solid #cbd5e1; border-radius: 8px; font-weight: 600;'>
                    {details.get('Suitable Soil Types', 'N/A')}
                </p>
            </div>
            
            <div style='margin-bottom: 18px;'>
                <h4 style='color: #1e3a8a; margin: 0 0 8px 0; font-size: 1.15em; font-weight: 800;'>📖 Best Cultivation Practices</h4>
                <p style='font-size: 0.98em; color: #000000; margin: 0; background: #eff6ff; border: 1.5px solid #93c5fd; padding: 14px; border-radius: 8px; line-height: 1.6; font-weight: 600;'>
                    {details.get('Cultivation Guide', 'N/A')}
                </p>
            </div>
            
            <div>
                <h4 style='color: #b45309; margin: 0 0 8px 0; font-size: 1.15em; font-weight: 800;'>🛡️ Integrated Pest & Disease Management (IPM)</h4>
                <p style='font-size: 0.98em; color: #78350f; margin: 0; background: #fffbeb; padding: 14px; border-radius: 8px; border: 2px solid #f59e0b; line-height: 1.6; font-weight: 600;'>
                    {details.get('Pest & Disease Advisory', 'N/A')}
                </p>
            </div>
        </div>
        """
    except Exception as e:
        return f"<div style='color: #991b1b; background: #fef2f2; border: 1.5px solid #f87171; padding: 15px; border-radius: 8px; font-weight: bold;'>Error retrieving crop details: {str(e)}</div>"

# High-Visibility Light Ice-Blue & Royal Navy Theme
CUSTOM_CSS = """
/* ============================================================
   HIGH-VISIBILITY THEME: LIGHT ICE-BLUE & ROYAL NAVY
   Engineered for maximum contrast, clarity, and readability
   ============================================================ */

/* Global Page Background & Base Text */
:root, html, body, .gradio-container, .dark, .dark html, .dark body, .dark .gradio-container {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    color-scheme: light !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Force all Gradio Cards / Panels to Crisp White with Strong Borders */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.dark .block,
.dark .form,
.dark .panel {
    background-color: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05) !important;
}

/* Main Top Banner (Deep Royal Navy with Crisp Bright White Text) */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    padding: 24px 30px !important;
    border-radius: 14px !important;
    margin-bottom: 22px !important;
    border: 2px solid #3b82f6 !important;
    box-shadow: 0 6px 20px rgba(30, 58, 138, 0.25) !important;
}
.main-header,
.main-header *,
.main-header h1,
.main-header h2,
.main-header div,
.main-header span {
    color: #ffffff !important;
}
.main-header p {
    color: #dbeafe !important;
    font-weight: 600 !important;
}

/* Headings in Deep Navy Blue */
.gradio-container h1, 
.gradio-container h2, 
.gradio-container h3, 
.gradio-container h4, 
.gradio-container h5, 
.gradio-container h6,
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4 {
    color: #1e3a8a !important;
    font-weight: 800 !important;
}

/* Body & Markdown Text */
.gradio-container .prose p, 
.gradio-container .prose li,
.gradio-container p {
    color: #0f172a !important;
    font-weight: 500;
}
.gradio-container .prose strong,
.gradio-container .prose b {
    color: #1e3a8a !important;
    font-weight: 800 !important;
}

/* Form Labels */
.gradio-container label, 
.gradio-container label span, 
.gradio-container .block label span,
.gradio-container .form label span,
.dark label,
.dark label span {
    color: #1e3a8a !important;
    font-weight: 800 !important;
    font-size: 0.98em !important;
}

/* Component info / helper text */
.gradio-container .info,
.dark .info {
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 0.88em !important;
}

/* Input Fields, Dropdowns, Number fields, Textareas */
.gradio-container input, 
.gradio-container select, 
.gradio-container textarea, 
.gradio-container .gr-input,
.gradio-container .gr-select,
.gradio-container .wrap-default,
.dark input,
.dark select,
.dark textarea,
.dark .gr-input,
.dark .gr-select {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #2563eb !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #64748b !important;
    font-weight: 500 !important;
}

/* Dropdown menus & items */
.gradio-container .token,
.gradio-container .single-select,
.gradio-container ul.options,
.gradio-container li.item,
.dark ul.options,
.dark li.item {
    color: #000000 !important;
    background-color: #ffffff !important;
    font-weight: 600 !important;
}
.gradio-container li.item:hover,
.gradio-container li.item.selected {
    background-color: #2563eb !important;
    color: #ffffff !important;
}

/* Sliders */
.gradio-container input[type="range"] {
    accent-color: #2563eb !important;
}
.gradio-container .value,
.dark .value {
    color: #1e3a8a !important;
    font-weight: 800 !important;
}

/* Primary Action Buttons (Vibrant Royal Blue with Bold White Text) */
.primary-agri-btn,
.primary-agri-btn *,
.primary-agri-btn span,
button.primary-agri-btn,
button.primary-agri-btn *,
button.primary-agri-btn span {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.5px !important;
    border: 1.5px solid #1e40af !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(29, 78, 216, 0.35) !important;
}

.primary-agri-btn:hover,
.primary-agri-btn:hover *,
button.primary-agri-btn:hover,
button.primary-agri-btn:hover * {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
    color: #ffffff !important;
}

/* Secondary Buttons */
.secondary-agri-btn,
.secondary-agri-btn *,
.secondary-agri-btn span,
button.secondary-agri-btn,
button.secondary-agri-btn *,
button.secondary-agri-btn span {
    background-color: #dbeafe !important;
    color: #1e3a8a !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    border: 2px solid #2563eb !important;
    border-radius: 8px !important;
}

.secondary-agri-btn:hover,
.secondary-agri-btn:hover *,
button.secondary-agri-btn:hover,
button.secondary-agri-btn:hover * {
    background-color: #bfdbfe !important;
    color: #172554 !important;
}

/* Tab Navigation Bar */
.gradio-container div.tab-nav,
.dark div.tab-nav {
    background-color: #e2e8f0 !important;
    border-bottom: 3px solid #2563eb !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 6px 6px 0 6px !important;
}

.gradio-container div.tab-nav button,
.dark div.tab-nav button {
    color: #1e293b !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    border-radius: 8px 8px 0 0 !important;
    background-color: #cbd5e1 !important;
    margin-right: 6px !important;
    border: 1.5px solid #94a3b8 !important;
    border-bottom: none !important;
    padding: 10px 16px !important;
}

.gradio-container div.tab-nav button * {
    color: inherit !important;
    font-weight: inherit !important;
}

.gradio-container div.tab-nav button:hover {
    background-color: #e2e8f0 !important;
    color: #1d4ed8 !important;
}

.gradio-container div.tab-nav button.selected,
.dark div.tab-nav button.selected {
    color: #1e3a8a !important;
    border: 2px solid #2563eb !important;
    border-bottom: 3px solid #ffffff !important;
    background-color: #ffffff !important;
    font-weight: 900 !important;
    box-shadow: 0 -2px 8px rgba(37, 99, 235, 0.1) !important;
}

.gradio-container div.tab-nav button.selected * {
    color: #1e3a8a !important;
    font-weight: 900 !important;
}

/* Stat Cards on Home Dashboard */
.stat-card {
    background: #ffffff !important;
    border: 2px solid #2563eb !important;
    border-radius: 12px !important;
    padding: 20px !important;
    text-align: center !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1) !important;
}

.stat-number {
    font-size: 2.4em !important;
    font-weight: 900 !important;
    color: #1d4ed8 !important;
    margin: 0 !important;
}

.stat-label {
    font-size: 0.92em !important;
    color: #000000 !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Accordion */
.gradio-container .accordion,
.dark .accordion {
    border: 2px solid #93c5fd !important;
    border-radius: 8px !important;
    background-color: #ffffff !important;
}
.gradio-container .accordion > button,
.dark .accordion > button {
    color: #1e3a8a !important;
    font-weight: 800 !important;
}
"""

def build_gradio_app():
    with gr.Blocks(title="AI Farming Advisory – Smart Agriculture for India") as demo:
        
        # Banner Header
        gr.HTML("""
        <div class="main-header" style="background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #1d4ed8 100%) !important; color: #ffffff !important; padding: 24px 30px !important; border-radius: 14px !important; margin-bottom: 22px !important; border: 2px solid #3b82f6 !important; box-shadow: 0 6px 20px rgba(30, 58, 138, 0.25) !important;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
                <div>
                    <h1 style="color: #ffffff !important; font-size: 2.3em !important; font-weight: 900 !important; margin: 0 !important; letter-spacing: -0.5px !important;">🌾 AI FARMING ADVISORY</h1>
                    <p style="color: #dbeafe !important; font-size: 1.05em !important; font-weight: 700 !important; margin: 6px 0 0 0 !important;">Smart Agriculture Decision Support for India • Grounded in ICAR & IMD Research Guidelines</p>
                </div>
                <div style="background: rgba(255,255,255,0.18) !important; padding: 12px 22px !important; border-radius: 10px !important; border: 2px solid #ffffff !important; text-align: right;">
                    <div style="font-size: 0.85em !important; text-transform: uppercase !important; letter-spacing: 1px !important; color: #dbeafe !important; font-weight: 800 !important;">Advisory Standard</div>
                    <div style="font-size: 1.25em !important; font-weight: 900 !important; color: #ffffff !important;">National Farming Guidelines</div>
                </div>
            </div>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            
            # ==========================================
            # TAB 1: HOME DASHBOARD
            # ==========================================
            with gr.Tab("🏠 Home Dashboard", id="tab_home"):
                with gr.Row():
                    gr.HTML("""
                    <div class="stat-card">
                        <div class="stat-number">23</div>
                        <div class="stat-label">Crops Supported</div>
                    </div>
                    """)
                    gr.HTML("""
                    <div class="stat-card">
                        <div class="stat-number">17</div>
                        <div class="stat-label">Indian States</div>
                    </div>
                    """)
                    gr.HTML("""
                    <div class="stat-card">
                        <div class="stat-number">7</div>
                        <div class="stat-label">Soil Classifications</div>
                    </div>
                    """)
                    gr.HTML("""
                    <div class="stat-card">
                        <div class="stat-number">13+</div>
                        <div class="stat-label">Indian Languages (Voice & Audio)</div>
                    </div>
                    """)
                
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        gr.HTML("""
                        <div style='background: #ffffff; border: 2px solid #2563eb; border-radius: 12px; padding: 24px; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08); margin-top: 15px;'>
                            <h3 style='color: #1e3a8a; font-size: 1.45em; font-weight: 800; margin-top: 0;'>🌱 Welcome to AI Farming Advisory</h3>
                            <p style='color: #000000; font-size: 1.02em; line-height: 1.6; font-weight: 600;'>
                                India possesses 15 distinct agro-climatic zones, each demanding tailored crop choices, soil amendments, and seasonal water management. 
                                This advisory system assists farmers, agronomists, and extension workers in determining the most suitable crops and farming protocols based on regional geography, soil texture, ambient weather, and seasonal timelines.
                            </p>
                            <h4 style='color: #1e3a8a; font-size: 1.2em; font-weight: 900; margin-bottom: 12px;'>🧭 Available Advisory Modules:</h4>
                            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 14px;'>
                                <div style='background: #eff6ff; border: 1.5px solid #93c5fd; border-radius: 8px; padding: 14px; grid-column: span 2;'>
                                    <b style='color: #1d4ed8; font-size: 1.05em; font-weight: 800;'>🎙️ 1. Multilingual Voice & Audio Advisory (All Indian Languages)</b>
                                    <div style='font-size: 0.94em; color: #000000; margin-top: 6px; font-weight: 600; line-height: 1.5;'>Speak or type your question in Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu, Assamese, or English. Returns spoken voice audio and written advisory in your exact native language.</div>
                                </div>
                                <div style='background: #eff6ff; border: 1.5px solid #93c5fd; border-radius: 8px; padding: 14px;'>
                                    <b style='color: #1d4ed8; font-size: 1.02em; font-weight: 800;'>2. Region-Based Crop Recommendation</b>
                                    <div style='font-size: 0.92em; color: #000000; margin-top: 6px; font-weight: 600; line-height: 1.5;'>Evaluates state, district, soil type, and weather to recommend the top 5 suitable crops with localized agricultural rationales.</div>
                                </div>
                                <div style='background: #eff6ff; border: 1.5px solid #93c5fd; border-radius: 8px; padding: 14px;'>
                                    <b style='color: #1d4ed8; font-size: 1.02em; font-weight: 800;'>3. Weather-Based Advisory</b>
                                    <div style='font-size: 0.92em; color: #000000; margin-top: 6px; font-weight: 600; line-height: 1.5;'>Live satellite weather feed or manual inputs, warning against climate stresses and advising irrigation and drainage.</div>
                                </div>
                                <div style='background: #eff6ff; border: 1.5px solid #93c5fd; border-radius: 8px; padding: 14px;'>
                                    <b style='color: #1d4ed8; font-size: 1.02em; font-weight: 800;'>4. Soil-Based Recommendation</b>
                                    <div style='font-size: 0.92em; color: #000000; margin-top: 6px; font-weight: 600; line-height: 1.5;'>Nutrient status, pH adjustments, and organic soil health improvement techniques for 7 major soil classes.</div>
                                </div>
                                <div style='background: #eff6ff; border: 1.5px solid #93c5fd; border-radius: 8px; padding: 14px;'>
                                    <b style='color: #1d4ed8; font-size: 1.02em; font-weight: 800;'>5. Seasonal & Regional Strategy</b>
                                    <div style='font-size: 0.92em; color: #000000; margin-top: 6px; font-weight: 600; line-height: 1.5;'>State-specific sowing calendars and agronomic protocols for Kharif, Rabi, and Zaid seasons.</div>
                                </div>
                            </div>
                        </div>
                        """)
                    
                    with gr.Column(scale=2):
                        gr.HTML("""
                        <div style='background: #ffffff; border: 2px solid #2563eb; border-radius: 12px; padding: 24px; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08); margin-top: 15px;'>
                            <h3 style='color: #1e3a8a; font-size: 1.35em; font-weight: 800; margin-top: 0;'>🌾 Sustainable Farming Practices</h3>
                            <div style='display: flex; flex-direction: column; gap: 12px;'>
                                <div style='background: #eff6ff; border-left: 5px solid #2563eb; border: 1px solid #bfdbfe; border-left-width: 5px; border-radius: 6px; padding: 12px 14px;'>
                                    <b style='color: #1e3a8a; font-size: 1em; font-weight: 800;'>🧪 Soil Health Card Testing</b>
                                    <div style='font-size: 0.92em; color: #000000; margin-top: 4px; font-weight: 600;'>Test your soil every 2 years to apply exact N-P-K and micronutrient doses, preventing over-fertilization.</div>
                                </div>
                                <div style='background: #eff6ff; border-left: 5px solid #1d4ed8; border: 1px solid #bfdbfe; border-left-width: 5px; border-radius: 6px; padding: 12px 14px;'>
                                    <b style='color: #1e3a8a; font-size: 1em; font-weight: 800;'>💧 Precision Micro-Irrigation</b>
                                    <div style='font-size: 0.92em; color: #000000; margin-top: 4px; font-weight: 600;'>Drip and sprinkler irrigation save 40–50% water while maintaining optimal root zone moisture.</div>
                                </div>
                                <div style='background: #eff6ff; border-left: 5px solid #0284c7; border: 1px solid #bfdbfe; border-left-width: 5px; border-radius: 6px; padding: 12px 14px;'>
                                    <b style='color: #1e3a8a; font-size: 1em; font-weight: 800;'>🌱 Crop Rotation with Legumes</b>
                                    <div style='font-size: 0.92em; color: #000000; margin-top: 4px; font-weight: 600;'>Rotate cereals with pulses (Moong, Chickpea) to naturally fix atmospheric nitrogen into the soil.</div>
                                </div>
                                <div style='background: #eff6ff; border-left: 5px solid #4338ca; border: 1px solid #bfdbfe; border-left-width: 5px; border-radius: 6px; padding: 12px 14px;'>
                                    <b style='color: #1e3a8a; font-size: 1em; font-weight: 800;'>🛡️ Integrated Pest Management (IPM)</b>
                                    <div style='font-size: 0.92em; color: #000000; margin-top: 4px; font-weight: 600;'>Deploy pheromone traps, yellow sticky cards, and bio-pesticides before resorting to chemical sprays.</div>
                                </div>
                            </div>
                        </div>
                        """)
            
            # ==========================================
            # TAB 2: MULTILINGUAL VOICE ADVISORY (ALL INDIAN LANGUAGES)
            # ==========================================
            with gr.Tab("🎙️ Voice Advisory (All Indian Languages)", id="tab_voice"):
                gr.Markdown("### 🎙️ Indian Languages Voice & Audio Advisory (किसान वाणी / உழவர் குரல் / రైతు వాణి)")
                gr.Markdown(
                    "Speak into your microphone or type your question in **any Indian language** "
                    "(Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu, Assamese, or English). "
                    "The system automatically understands your language, consults agronomic scientific guidelines, "
                    "and returns both **spoken voice audio** and **native written advisory** in your exact language."
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        voice_lang_select = gr.Dropdown(
                            choices=list(INDIAN_LANGUAGES.keys()),
                            value="Auto-Detect",
                            label="🌐 Select Input Language (or keep Auto-Detect)",
                            info="Choose your native tongue or let the AI detect it automatically"
                        )
                        voice_audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="🎙️ 1. Speak or Upload Voice Note (Microphone)"
                        )
                        voice_text_input = gr.Textbox(
                            lines=3,
                            placeholder="✍️ Or type your question in any Indian language (e.g. महाराष्ट्र में कौन सी फसल लगाएं?, தமிழ்நாட்டில் பருத்தி எப்போது பயிரிட வேண்டும்?, Which crop is best for black soil?)...",
                            label="✍️ 2. Or Type Your Farming Question"
                        )
                        
                        with gr.Row():
                            voice_submit_btn = gr.Button("🗣️ GET VOICE & WRITTEN ADVISORY", variant="primary", elem_classes=["primary-agri-btn"])
                            voice_clear_btn = gr.Button("🔄 Clear Input", variant="secondary", elem_classes=["secondary-agri-btn"])
                            
                        gr.Markdown("**Quick Sample Questions (Click to Test):**")
                        with gr.Row():
                            ex1 = gr.Button("🌾 Maharashtra Kharif (Hindi)", size="sm", elem_classes=["secondary-agri-btn"])
                            ex2 = gr.Button("🌱 Tamil Nadu Cotton (Tamil)", size="sm", elem_classes=["secondary-agri-btn"])
                        with gr.Row():
                            ex3 = gr.Button("🪴 Black Soil Crops (Telugu)", size="sm", elem_classes=["secondary-agri-btn"])
                            ex4 = gr.Button("💧 Wheat Irrigation (Punjabi)", size="sm", elem_classes=["secondary-agri-btn"])
                    
                    with gr.Column(scale=1):
                        voice_audio_output = gr.Audio(
                            label="🔊 Spoken Audio Response in Your Native Language",
                            type="filepath",
                            autoplay=True
                        )
                        voice_html_output = gr.HTML(
                            label="Native Advisory Details"
                        )
                
                # Wire voice advisory interaction
                voice_submit_btn.click(
                    fn=process_voice_or_text_query,
                    inputs=[voice_audio_input, voice_text_input, voice_lang_select],
                    outputs=[voice_html_output, voice_audio_output]
                )
                
                def clear_voice_inputs():
                    return None, "", None, "<div style='color: #475569; padding: 15px; font-weight: 600;'>Inputs cleared. Speak or type a new question.</div>"
                    
                voice_clear_btn.click(
                    fn=clear_voice_inputs,
                    outputs=[voice_audio_input, voice_text_input, voice_audio_output, voice_html_output]
                )
                
                ex1.click(lambda: ("महाराष्ट्र में खरीफ में कौन सी फसल उगानी चाहिए?", "Hindi (हिन्दी)"), outputs=[voice_text_input, voice_lang_select])
                ex2.click(lambda: ("தமிழ்நாட்டில் பருத்தி சாகுபடிக்கு சிறந்த காலம் எது?", "Tamil (தமிழ்)"), outputs=[voice_text_input, voice_lang_select])
                ex3.click(lambda: ("నల్ల నేలలో ఏ పంటలు బాగా పండుతాయి?", "Telugu (తెలుగు)"), outputs=[voice_text_input, voice_lang_select])
                ex4.click(lambda: ("ਕਣਕ ਦੀ ਫਸਲ ਲਈ ਸਿੰਚਾਈ ਦਾ ਤਰੀਕਾ ਕੀ ਹੈ?", "Punjabi (ਪੰਜਾਬੀ)"), outputs=[voice_text_input, voice_lang_select])
            
            # ==========================================
            # TAB 3: REGION-BASED CROP RECOMMENDATION
            # ==========================================
            with gr.Tab("🌾 Region-Based Crop Recommendation", id="tab_rec"):
                gr.Markdown("### 📍 Enter Your Regional, Soil & Weather Parameters")
                gr.Markdown("This module recommends the **Top 5 crops** suitable for your specific location and explains the agricultural rationale behind each choice.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        state_input = gr.Dropdown(
                            choices=INDIAN_STATES,
                            value="Maharashtra",
                            label="1. Indian State",
                            info="Select your state"
                        )
                        district_input = gr.Dropdown(
                            choices=STATE_DISTRICT_MAPPING["Maharashtra"],
                            value=STATE_DISTRICT_MAPPING["Maharashtra"][0],
                            label="2. District or Agro-Climatic Region",
                            info="Select your regional district"
                        )
                        soil_input = gr.Dropdown(
                            choices=SOIL_TYPES,
                            value="Black soil",
                            label="3. Soil Classification",
                            info="Soil type in your field"
                        )
                        season_input = gr.Dropdown(
                            choices=SEASONS,
                            value="Kharif",
                            label="4. Target Season",
                            info="Kharif (Monsoon), Rabi (Winter), Zaid (Summer)"
                        )
                    
                    with gr.Column(scale=1):
                        temp_slider = gr.Slider(
                            minimum=10.0,
                            maximum=45.0,
                            value=28.0,
                            step=0.5,
                            label="5. Ambient Temperature (°C)",
                            info="Average growing temperature"
                        )
                        humidity_slider = gr.Slider(
                            minimum=20.0,
                            maximum=100.0,
                            value=68.0,
                            step=1.0,
                            label="6. Relative Humidity (%)",
                            info="Atmospheric moisture level"
                        )
                        rainfall_slider = gr.Slider(
                            minimum=200.0,
                            maximum=3000.0,
                            value=900.0,
                            step=50.0,
                            label="7. Expected Rainfall (mm)",
                            info="Seasonal precipitation"
                        )
                        ph_slider = gr.Slider(
                            minimum=4.5,
                            maximum=9.0,
                            value=6.8,
                            step=0.1,
                            label="8. Soil pH Level",
                            info="Acidity/Alkalinity (7.0 is neutral)"
                        )
                        
                        with gr.Accordion("Optional: Soil Health Card N-P-K (kg/ha)", open=False):
                            gr.Markdown("*Leave as 0 to automatically use regional soil type baselines.*")
                            with gr.Row():
                                n_input = gr.Number(value=0, label="Nitrogen (N)", precision=1)
                                p_input = gr.Number(value=0, label="Phosphorus (P)", precision=1)
                                k_input = gr.Number(value=0, label="Potassium (K)", precision=1)
                
                rec_btn = gr.Button("RECOMMEND SUITABLE CROPS", variant="primary", elem_classes=["primary-agri-btn"])
                
                # Dynamic update of districts when state changes
                state_input.change(fn=on_state_change, inputs=[state_input], outputs=[district_input])
                
                recommendation_output = gr.HTML(label="Recommendation Results")
                
                rec_btn.click(
                    fn=handle_crop_recommendation,
                    inputs=[
                        state_input,
                        district_input,
                        soil_input,
                        season_input,
                        temp_slider,
                        humidity_slider,
                        rainfall_slider,
                        ph_slider,
                        n_input,
                        p_input,
                        k_input
                    ],
                    outputs=[recommendation_output]
                )

            # ==========================================
            # TAB 3: WEATHER-BASED CROP ADVISORY
            # ==========================================
            with gr.Tab("🌦️ Weather-Based Advisory", id="tab_weather"):
                gr.Markdown("### 🛰️ Live & Manual Agro-Meteorological Advisory")
                gr.Markdown("Analyze current weather risks (heat stress, chilling, waterlogging, or fungal humidity) and retrieve crop suitability and precaution guidelines.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        w_state = gr.Dropdown(choices=INDIAN_STATES, value="Maharashtra", label="State")
                        w_district = gr.Dropdown(choices=STATE_DISTRICT_MAPPING["Maharashtra"], value=STATE_DISTRICT_MAPPING["Maharashtra"][0], label="District / Region")
                        w_fetch_btn = gr.Button("🌐 Fetch Live Weather (Free IMD/Open-Meteo Satellite Feed)", elem_classes=["secondary-agri-btn"])
                        w_status_html = gr.HTML("""
                        <div style='background: #eff6ff; border: 2px solid #2563eb; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px;'>
                            <b style='color: #1d4ed8; font-size: 1.1em;'>Manual Weather Mode</b><br>
                            <span style='font-size: 0.95em; color: #000000; font-weight: 700;'>Click 'Fetch Live Weather' or adjust the values below.</span>
                        </div>
                        """)
                    
                    with gr.Column(scale=1):
                        w_season = gr.Dropdown(choices=SEASONS, value="Kharif", label="Current Cropping Season")
                        w_temp = gr.Slider(minimum=5.0, maximum=48.0, value=28.0, step=0.5, label="Temperature (°C)")
                        w_humidity = gr.Slider(minimum=15.0, maximum=100.0, value=65.0, step=1.0, label="Humidity (%)")
                        w_rainfall = gr.Slider(minimum=0.0, maximum=3000.0, value=750.0, step=25.0, label="Precipitation / Rainfall (mm)")
                
                # Hidden state variables to track live vs manual source
                w_is_live = gr.State(False)
                w_source_label = gr.State("Manual Input")
                
                w_analyze_btn = gr.Button("ANALYZE WEATHER & GENERATE ADVISORY", variant="primary", elem_classes=["primary-agri-btn"])
                weather_output = gr.HTML(label="Weather Advisory Results")
                
                w_state.change(fn=on_state_change, inputs=[w_state], outputs=[w_district])
                
                w_fetch_btn.click(
                    fn=fetch_weather_for_ui,
                    inputs=[w_state, w_district],
                    outputs=[w_temp, w_humidity, w_rainfall, w_status_html, w_is_live, w_source_label]
                )
                
                w_analyze_btn.click(
                    fn=handle_weather_advisory,
                    inputs=[w_state, w_district, w_temp, w_humidity, w_rainfall, w_season, w_is_live, w_source_label],
                    outputs=[weather_output]
                )

            # ==========================================
            # TAB 4: SOIL-BASED CROP ADVISORY
            # ==========================================
            with gr.Tab("🪴 Soil-Based Crop Advisory", id="tab_soil"):
                gr.Markdown("### 🧪 Indian Soil Classification & Nutrient Advisory")
                gr.Markdown("Select a soil type to inspect its physical characteristics, typical N-P-K nutrient profile, pH requirements, and organic restoration practices.")
                
                with gr.Row():
                    soil_select = gr.Dropdown(choices=SOIL_TYPES, value="Black soil", label="Select Soil Classification")
                    soil_btn = gr.Button("ANALYZE SOIL PROFILE", variant="primary", elem_classes=["primary-agri-btn"])
                
                soil_output = gr.HTML()
                soil_btn.click(fn=handle_soil_advisory, inputs=[soil_select], outputs=[soil_output])

            # ==========================================
            # TAB 5: SEASONAL CROP ADVISORY
            # ==========================================
            with gr.Tab("🗓️ Seasonal Crop Advisory", id="tab_season"):
                gr.Markdown("### 🌾 State-Specific Seasonal Cropping Calendars")
                gr.Markdown("Receive customized agronomic guidance for Kharif (Monsoon), Rabi (Winter), or Zaid (Summer) in your state.")
                
                with gr.Row():
                    s_state = gr.Dropdown(choices=INDIAN_STATES, value="Punjab", label="State")
                    s_season = gr.Dropdown(choices=SEASONS, value="Rabi", label="Season")
                    s_btn = gr.Button("GET SEASONAL ADVISORY", variant="primary", elem_classes=["primary-agri-btn"])
                    
                seasonal_output = gr.HTML()
                s_btn.click(fn=handle_seasonal_advisory, inputs=[s_state, s_season], outputs=[seasonal_output])

            # ==========================================
            # TAB 6: CROP INFORMATION ENCYCLOPEDIA
            # ==========================================
            with gr.Tab("📖 Crop Encyclopedia", id="tab_info"):
                gr.Markdown("### 📚 Searchable Indian Crop Knowledge Base")
                gr.Markdown("Explore complete cultivation guidelines, standard NPK requirements, irrigation cycles, and Integrated Pest Management (IPM) for all supported crops.")
                
                all_crops = get_crop_list()
                with gr.Row():
                    crop_dropdown = gr.Dropdown(choices=all_crops, value="Rice" if "Rice" in all_crops else all_crops[0], label="Select Crop")
                    crop_btn = gr.Button("VIEW CROP DETAILS", variant="primary", elem_classes=["primary-agri-btn"])
                    
                crop_details_output = gr.HTML()
                crop_btn.click(fn=handle_crop_details, inputs=[crop_dropdown], outputs=[crop_details_output])
                crop_dropdown.change(fn=handle_crop_details, inputs=[crop_dropdown], outputs=[crop_details_output])

            # ==========================================
            # TAB 7: DATA VISUALIZATIONS
            # ==========================================
            with gr.Tab("📊 Agricultural Data Insights", id="tab_viz"):
                gr.Markdown("### 📈 Interactive Agricultural Data Visualizations")
                gr.Markdown("Explore the relationships between crops, soil types, seasonal windows, temperatures, and rainfall across India.")
                
                with gr.Tabs():
                    with gr.Tab("🗺️ Crop Distribution by State"):
                        plot_1 = gr.Plot(value=plot_crop_distribution_by_state)
                    with gr.Tab("🗓️ Crop Suitability by Season"):
                        plot_2 = gr.Plot(value=plot_crop_suitability_by_season)
                    with gr.Tab("🌡️ Temperature vs. Crop Suitability"):
                        plot_3 = gr.Plot(value=plot_temp_vs_crop)
                    with gr.Tab("🌧️ Rainfall vs. Humidity Dynamics"):
                        plot_4 = gr.Plot(value=plot_rainfall_vs_crop)
                    with gr.Tab("🪴 Soil Type Compatibility Matrix"):
                        plot_5 = gr.Plot(value=plot_soil_type_vs_crop)

        # Footer
        gr.HTML("""
        <div style='margin-top: 30px; padding: 20px 10px; border-top: 2px solid #cbd5e1; text-align: center; color: #1e3a8a; font-size: 0.95em; font-weight: 700;'>
            <b style='color: #1d4ed8; font-size: 1.05em;'>AI Farming Advisory – Smart Agriculture for India</b><br>
            <span style='color: #0f172a;'>Developed with Scikit-learn, Gradio, Pandas, and Plotly • Sourced from ICAR, IMD, and Ministry of Agriculture Benchmarks.</span>
        </div>
        """)

    return demo

FORCE_LIGHT_MODE_HEAD = """
<script>
    function ensureLightMode() {
        if (document.documentElement.classList.contains('dark')) {
            document.documentElement.classList.remove('dark');
        }
        if (document.body && document.body.classList.contains('dark')) {
            document.body.classList.remove('dark');
        }
    }
    ensureLightMode();
    const observer = new MutationObserver(ensureLightMode);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    window.addEventListener('DOMContentLoaded', () => {
        ensureLightMode();
        if (document.body) {
            observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
        }
    });
</script>
"""

if __name__ == "__main__":
    app = build_gradio_app()
    # Launch on localhost with high-visibility light theme, CSS, and head script
    # Gradio will automatically use 7860 or the next available open port
    app.launch(
        server_name="127.0.0.1",
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky", neutral_hue="slate"),
        css=CUSTOM_CSS,
        head=FORCE_LIGHT_MODE_HEAD
    )
