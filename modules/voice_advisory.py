"""
Multilingual Voice Advisory Module for Indian Farmers
Supports speech-to-text, query understanding, agronomic knowledge retrieval,
and multilingual text-to-speech response across all major Indian languages:
Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam,
Punjabi, Odia, Urdu, Assamese, and English.
"""

import os
import re
import tempfile
import requests
import pandas as pd
import speech_recognition as sr
from gtts import gTTS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFO_PATH = os.path.join(BASE_DIR, "datasets", "crop_information.csv")
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "indian_crop_dataset.csv")

# Language Mapping Catalog for Major Indian Languages
INDIAN_LANGUAGES = {
    "Auto-Detect": {"code": "auto", "speech": "hi-IN", "tts": "hi", "name": "Auto-Detect Language"},
    "Hindi (हिन्दी)": {"code": "hi", "speech": "hi-IN", "tts": "hi", "name": "Hindi"},
    "Bengali (বাংলা)": {"code": "bn", "speech": "bn-IN", "tts": "bn", "name": "Bengali"},
    "Telugu (తెలుగు)": {"code": "te", "speech": "te-IN", "tts": "te", "name": "Telugu"},
    "Marathi (मराठी)": {"code": "mr", "speech": "mr-IN", "tts": "mr", "name": "Marathi"},
    "Tamil (தமிழ்)": {"code": "ta", "speech": "ta-IN", "tts": "ta", "name": "Tamil"},
    "Gujarati (ગુજરાતી)": {"code": "gu", "speech": "gu-IN", "tts": "gu", "name": "Gujarati"},
    "Kannada (ಕನ್ನಡ)": {"code": "kn", "speech": "kn-IN", "tts": "kn", "name": "Kannada"},
    "Malayalam (മലയാളം)": {"code": "ml", "speech": "ml-IN", "tts": "ml", "name": "Malayalam"},
    "Punjabi (ਪੰਜਾਬੀ)": {"code": "pa", "speech": "pa-IN", "tts": "hi", "name": "Punjabi"},
    "Odia (ଓଡ଼ିଆ)": {"code": "or", "speech": "or-IN", "tts": "hi", "name": "Odia"},
    "Urdu (اردو)": {"code": "ur", "speech": "ur-IN", "tts": "ur", "name": "Urdu"},
    "Assamese (অসমীয়া)": {"code": "as", "speech": "as-IN", "tts": "bn", "name": "Assamese"},
    "English": {"code": "en", "speech": "en-IN", "tts": "en", "name": "English"},
}

# Reverse lookup from code to display name
CODE_TO_LANG_NAME = {
    "hi": "Hindi (हिन्दी)",
    "bn": "Bengali (বাংলা)",
    "te": "Telugu (తెలుగు)",
    "mr": "Marathi (मराठी)",
    "ta": "Tamil (தமிழ்)",
    "gu": "Gujarati (ગુજરાતી)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "or": "Odia (ଓଡ଼ିଆ)",
    "ur": "Urdu (اردو)",
    "as": "Assamese (অসমীয়া)",
    "en": "English"
}

def ensure_wav(audio_path):
    """
    Ensures input audio file is converted to a clean WAV format suitable for SpeechRecognition.
    """
    if not audio_path or not os.path.exists(audio_path):
        return None
        
    if audio_path.lower().endswith(".wav"):
        return audio_path
        
    try:
        from pydub import AudioSegment
        sound = AudioSegment.from_file(audio_path)
        wav_path = tempfile.mktemp(suffix=".wav")
        sound.set_channels(1).set_frame_rate(16000).export(wav_path, format="wav")
        return wav_path
    except Exception:
        try:
            import soundfile as sf
            data, samplerate = sf.read(audio_path)
            wav_path = tempfile.mktemp(suffix=".wav")
            sf.write(wav_path, data, samplerate, format="WAV")
            return wav_path
        except Exception:
            return audio_path

def transcribe_audio(audio_path, selected_lang_key="Auto-Detect"):
    """
    Transcribes spoken voice audio in any Indian language.
    """
    if not audio_path or not os.path.exists(audio_path):
        return "", "No audio provided. Please speak into the microphone or upload an audio file."
        
    lang_info = INDIAN_LANGUAGES.get(selected_lang_key, INDIAN_LANGUAGES["Auto-Detect"])
    speech_lang = lang_info["speech"]
    
    wav_path = ensure_wav(audio_path)
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=speech_lang)
            return text, None
    except sr.UnknownValueError:
        return "", "Could not detect clear speech. Please ensure your microphone is enabled and speak clearly."
    except sr.RequestError as e:
        return "", f"Speech recognition connection error: {str(e)}"
    except Exception as e:
        return "", f"Audio transcription error: {str(e)}"

def translate_text(text, source_lang='auto', target_lang='en'):
    """
    Robust translation function supporting all Indian languages.
    Returns: (translated_text, detected_source_lang_code)
    """
    if not text or not str(text).strip():
        return "", source_lang
        
    text_clean = str(text).strip()
    
    if source_lang == target_lang and source_lang != 'auto':
        return text_clean, source_lang
        
    # Primary: Google Translate single endpoint
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": source_lang,
            "tl": target_lang,
            "dt": "t",
            "q": text_clean
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, params=params, headers=headers, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            translated_chunks = [chunk[0] for chunk in data[0] if chunk and chunk[0]]
            translated_text = "".join(translated_chunks)
            detected_lang = data[2] if len(data) > 2 and data[2] else source_lang
            if detected_lang == "auto":
                detected_lang = "en"
            return translated_text, detected_lang
    except Exception:
        pass
        
    # Secondary fallback: deep_translator MyMemory
    try:
        from deep_translator import MyMemoryTranslator
        src = "en" if source_lang == "auto" else source_lang
        tgt = target_lang if target_lang != "auto" else "en"
        tr = MyMemoryTranslator(source=src, target=tgt).translate(text_clean)
        return tr, source_lang
    except Exception:
        pass
        
    return text_clean, source_lang

def answer_agricultural_query(query_en):
    """
    Intelligent agricultural expert query engine.
    Analyzes English translation of user query, matches crops, states, soils,
    seasons, and cultivation problems, and formulates a comprehensive agronomic response.
    """
    q = query_en.lower()
    
    # Load knowledge databases
    crop_info_df = pd.read_csv(INFO_PATH) if os.path.exists(INFO_PATH) else pd.DataFrame()
    crop_dataset_df = pd.read_csv(DATASET_PATH) if os.path.exists(DATASET_PATH) else pd.DataFrame()
    
    all_crops = crop_info_df["Crop Name"].dropna().unique().tolist() if not crop_info_df.empty else []
    all_states = [
        "Maharashtra", "Gujarat", "Punjab", "Haryana", "Rajasthan", "Uttar Pradesh",
        "Madhya Pradesh", "Karnataka", "Tamil Nadu", "Kerala", "Andhra Pradesh",
        "Telangana", "West Bengal", "Bihar", "Assam", "Odisha", "Himachal Pradesh"
    ]
    all_soils = ["Black", "Red", "Alluvial", "Laterite", "Sandy", "Clay", "Loamy"]
    all_seasons = ["Kharif", "Rabi", "Zaid"]
    
    # Entity Extraction
    matched_crops = [c for c in all_crops if c.lower() in q]
    matched_states = [s for s in all_states if s.lower() in q]
    matched_soils = [s for s in all_soils if s.lower() in q]
    
    matched_seasons = []
    if "kharif" in q or "monsoon" in q or "rainy" in q:
        matched_seasons.append("Kharif")
    if "rabi" in q or "winter" in q or "cold" in q:
        matched_seasons.append("Rabi")
    if "zaid" in q or "summer" in q or "hot" in q:
        matched_seasons.append("Zaid")
        
    # Scenario 1: Specific Crop Inquiry
    if matched_crops:
        crop_name = matched_crops[0]
        row = crop_info_df[crop_info_df["Crop Name"].str.lower() == crop_name.lower()]
        if not row.empty:
            c = row.iloc[0].to_dict()
            resp = (
                f"🌾 Agronomic Advisory for {c['Crop Name']} ({c.get('Category', 'Crop')}):\n\n"
                f"• Suitable Indian States: {c.get('Suitable Indian States', 'N/A')}\n"
                f"• Suitable Soil Types: {c.get('Suitable Soil Types', 'N/A')}\n"
                f"• Optimal Sowing Season & Duration: {c.get('Sowing & Harvesting Seasons', 'N/A')} ({c.get('Crop Duration', 'N/A')})\n"
                f"• Climate Requirements: Optimal Temperature: {c.get('Optimal Temperature Range (°C)', 'N/A')}°C | Rainfall Threshold: {c.get('Optimal Rainfall Range (mm)', 'N/A')} mm | Soil pH: {c.get('Soil pH Range', 'N/A')}\n"
                f"• Fertilizer / NPK Dose: {c.get('Standard NPK (kg/ha)', 'N/A')} kg/ha\n"
                f"• Irrigation Need: {c.get('Irrigation Requirements', 'N/A')}\n"
                f"• Expected Yield: {c.get('Expected Yield', 'N/A')}\n\n"
                f"💡 Best Cultivation Practice:\n{c.get('Cultivation Guide', 'Maintain proper spacing and seed treatment.')}\n\n"
                f"🛡️ Pest & Disease Protection (IPM):\n{c.get('Pest & Disease Advisory', 'Follow integrated pest management and timely weeding.')}"
            )
            return resp

    # Scenario 2: Regional / State Recommendation Query
    if matched_states:
        state = matched_states[0]
        season = matched_seasons[0] if matched_seasons else "Kharif"
        
        # Filter dataset for top crops in this state & season
        if not crop_dataset_df.empty:
            subset = crop_dataset_df[(crop_dataset_df["State"].str.lower() == state.lower()) & (crop_dataset_df["Season"].str.lower() == season.lower())]
            if subset.empty:
                subset = crop_dataset_df[crop_dataset_df["State"].str.lower() == state.lower()]
            
            top_crops = subset["Crop Name"].value_counts().head(5).index.tolist()
            if not top_crops and not crop_info_df.empty:
                top_crops = ["Rice", "Wheat", "Cotton", "Sugarcane", "Soybean"]
                
            resp = (
                f"📍 Crop Recommendation for {state} ({season} Season):\n\n"
                f"Based on agro-climatic conditions and ICAR research guidelines, the most suitable crops for {state} during {season} are:\n"
                + "\n".join([f"{i+1}. {crop}" for i, crop in enumerate(top_crops)])
                + f"\n\n📋 Key Agronomic Tips for {state}:\n"
                f"• Sowing Window: Ensure sowing aligns with local monsoon onset or winter irrigation availability.\n"
                f"• Soil Preparation: Perform soil testing to apply balanced NPK doses and avoid excess nitrogen.\n"
                f"• Water Management: Adopt drip or furrow irrigation to optimize water productivity."
            )
            return resp

    # Scenario 3: Soil-Based Recommendation Query
    if matched_soils:
        soil = matched_soils[0]
        from modules.soil_advisory import get_soil_advisory
        soil_adv = get_soil_advisory(soil + " soil")
        top_crops_soil = [c["name"] for c in soil_adv["crops"][:5]]
        
        resp = (
            f"🪴 Soil Advisory for {soil} Soil:\n\n"
            f"• Typical Characteristics: {soil_adv['description']}\n"
            f"• Optimal Soil pH: {soil_adv['typical_ph']}\n"
            f"• Top Suitable Crops: {', '.join(top_crops_soil)}\n"
            f"• Water Management: {soil_adv['irrigation_guidance']}\n\n"
            f"🌿 Soil Health Restoration Tips:\n"
            + "\n".join([f"• {imp}" for imp in soil_adv["improvement_suggestions"][:3]])
        )
        return resp

    # Scenario 4: Seasonal Recommendation Query
    if matched_seasons:
        season = matched_seasons[0]
        from modules.regional_advisory import SEASON_OVERVIEWS
        s_info = SEASON_OVERVIEWS.get(season, {})
        resp = (
            f"🗓️ Advisory for {season} Cropping Season:\n\n"
            f"• Duration & Calendar: {s_info.get('period', 'Seasonal Window')}\n"
            f"• Weather Characteristics: {s_info.get('weather_profile', 'Favorable conditions')}\n"
            f"• Cropping Strategy: {s_info.get('cropping_strategy', 'Select certified regional seeds')}\n\n"
            f"📋 Recommended Field Practices:\n"
            + "\n".join([f"• {p}" for p in s_info.get('key_practices', [])[:3]])
        )
        return resp

    # Scenario 5: General Farming / Irrigation / Fertilizer Query
    if "water" in q or "irrigation" in q:
        return (
            "💧 General Agricultural Water Management Advisory:\n\n"
            "1. Drip and Micro-Sprinkler Irrigation: Highly recommended for vegetables, cotton, and orchards, reducing water consumption by 40-50%.\n"
            "2. Critical Growth Stages: Ensure adequate moisture during germination, tillering/flowering, and grain/pod development.\n"
            "3. Drainage Management: During heavy monsoon spells, install drainage channels to prevent waterlogging and root rot."
        )

    if "fertilizer" in q or "npk" in q or "urea" in q or "manure" in q:
        return (
            "🧪 Balanced Soil Nutrient & Fertilizer Advisory:\n\n"
            "1. Standard Benchmark Ratio: For cereal crops (Rice, Wheat, Maize), general recommended ratio is 4:2:1 (N:P:K); for pulses, 1:2:1.\n"
            "2. Split Nitrogen Application: Apply 50% basal at sowing, 25% at tillering, and 25% at panicle emergence.\n"
            "3. Organic Amendments: Incorporate 5-10 tons/ha of Farmyard Manure (FYM) or vermicompost to boost soil biological fertility."
        )

    if "pest" in q or "disease" in q or "insect" in q or "spray" in q:
        return (
            "🛡️ Integrated Pest & Disease Management Advisory (IPM):\n\n"
            "1. Cultural Control: Deep summer ploughing, clean cultivation, and rotating crops with leguminous pulses.\n"
            "2. Biological Monitoring: Install pheromone traps (5 traps/acre) and yellow sticky traps for early detection.\n"
            "3. Seed Treatment: Treat seeds with Trichoderma viride (4g/kg) or certified bio-fungicides prior to sowing."
        )

    # Default Comprehensive Farming Advisory
    return (
        "🌾 Comprehensive Indian Farmer Advisory:\n\n"
        "To provide the most tailored recommendations, you can ask about:\n"
        "1. Best crops for your specific State (e.g., 'What to plant in Maharashtra?')\n"
        "2. Best crops for your Soil type (e.g., 'Crops for black soil')\n"
        "3. Seasonal cultivation advice (e.g., 'Kharif sowing guide')\n"
        "4. Specific crop management (e.g., 'Cotton water requirements' or 'Wheat NPK dose')\n\n"
        "Key Principle: Always conduct soil testing through your nearest Krishi Vigyan Kendra (KVK) and use certified ICAR/State Agriculture University seed varieties."
    )

def generate_voice_response(text, lang_code):
    """
    Synthesizes native language audio from response text using gTTS.
    Returns path to the generated MP3 file, or None if unavailable.
    """
    if not text or not str(text).strip():
        return None
        
    gtts_lang = "en"
    for k, v in INDIAN_LANGUAGES.items():
        if v["code"] == lang_code:
            gtts_lang = v["tts"]
            break
            
    try:
        clean_text = str(text)
        # Clean markdown formatting symbols for natural vocal speech
        clean_text = re.sub(r'[*#_•`\d+\.]', '', clean_text)
        clean_text = " ".join(clean_text.split())
        
        # Keep speech concise (first 350 characters) so audio generates instantly
        if len(clean_text) > 350:
            clean_text = clean_text[:350] + "..."
            
        tts = gTTS(text=clean_text, lang=gtts_lang, slow=False)
        temp_audio = tempfile.mktemp(suffix=".mp3")
        tts.save(temp_audio)
        return temp_audio
    except Exception:
        # Fallback to English if native TTS language fails
        try:
            tts = gTTS(text="Here is your agricultural advisory.", lang="en", slow=False)
            temp_audio = tempfile.mktemp(suffix=".mp3")
            tts.save(temp_audio)
            return temp_audio
        except Exception:
            return None

def process_voice_or_text_query(audio_input, text_input, selected_lang_key):
    """
    Main pipeline function:
    1. Obtains query from voice transcription or text box.
    2. Identifies language.
    3. Translates query to English.
    4. Retrieves expert agronomic answer.
    5. Translates answer back to user's native language.
    6. Generates voice speech output.
    """
    query_text = ""
    transcription_status = ""
    
    # Check if voice input provided
    if audio_input is not None:
        transcribed, err = transcribe_audio(audio_input, selected_lang_key)
        if err:
            transcription_status = f"<div style='color: #b91c1c; background: #fef2f2; border: 1.5px solid #ef4444; padding: 10px 14px; border-radius: 8px; margin-bottom: 12px; font-weight: 600;'>🎙️ Voice Notice: {err}</div>"
        else:
            query_text = transcribed
            transcription_status = f"<div style='color: #15803d; background: #f0fdf4; border: 1.5px solid #22c55e; padding: 10px 14px; border-radius: 8px; margin-bottom: 12px; font-weight: 700;'>🎙️ Voice Detected: <i>\"{transcribed}\"</i></div>"
            
    # If no audio or audio empty, check text input
    if not query_text and text_input and str(text_input).strip():
        query_text = str(text_input).strip()
        transcription_status = f"<div style='color: #1e40af; background: #eff6ff; border: 1.5px solid #3b82f6; padding: 10px 14px; border-radius: 8px; margin-bottom: 12px; font-weight: 700;'>📝 Written Query: <i>\"{query_text}\"</i></div>"
        
    if not query_text:
        return (
            "<div style='background: #fffbeb; border: 2px solid #f59e0b; padding: 18px; border-radius: 10px; color: #78350f; font-weight: 700;'>"
            "⚠️ Please speak into the microphone or type your question in any Indian language above."
            "</div>",
            None
        )
        
    # Determine language
    target_lang_code = "auto"
    if selected_lang_key in INDIAN_LANGUAGES and selected_lang_key != "Auto-Detect":
        target_lang_code = INDIAN_LANGUAGES[selected_lang_key]["code"]
        
    # Translate question to English for agronomic reasoning
    query_en, detected_code = translate_text(query_text, source_lang=target_lang_code, target_lang='en')
    
    # Actual language code for reply
    reply_lang_code = detected_code if target_lang_code == "auto" else target_lang_code
    lang_display_name = CODE_TO_LANG_NAME.get(reply_lang_code, reply_lang_code.upper())
    
    # Generate agronomic answer in English
    advisory_en = answer_agricultural_query(query_en)
    
    # Translate answer back to user's native Indian language
    if reply_lang_code != "en":
        advisory_native, _ = translate_text(advisory_en, source_lang='en', target_lang=reply_lang_code)
    else:
        advisory_native = advisory_en
        
    # Generate Voice Output in native language
    audio_response_path = generate_voice_response(advisory_native, reply_lang_code)
    
    # Format high-visibility HTML response card
    formatted_advisory_html = advisory_native.replace("\n", "<br>")
    
    response_html = f"""
    <div style='background: #ffffff; border: 2px solid #2563eb; border-radius: 12px; padding: 24px; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.1); margin-top: 15px;'>
        {transcription_status}
        
        <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #cbd5e1; padding-bottom: 12px; margin-bottom: 16px;'>
            <div>
                <h3 style='margin: 0; color: #1e3a8a; font-size: 1.5em; font-weight: 900;'>🗣️ Agricultural Voice Advisory</h3>
                <div style='font-size: 0.95em; color: #000000; font-weight: 600; margin-top: 4px;'>
                    Language: <b style='color: #1d4ed8;'>{lang_display_name}</b> | Mode: <b>Bilingual Voice & Text</b>
                </div>
            </div>
            <div>
                <span style='background: #dbeafe; color: #1e40af; border: 1.5px solid #2563eb; padding: 6px 16px; border-radius: 16px; font-weight: 800; font-size: 0.88em;'>
                    🇮🇳 Native Indian Language
                </span>
            </div>
        </div>
        
        <div style='background: #f8fafc; border: 2px solid #93c5fd; border-radius: 10px; padding: 18px 22px; margin-bottom: 18px;'>
            <div style='font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.5px; color: #1e3a8a; font-weight: 800; margin-bottom: 6px;'>
                🌾 Advisory in {lang_display_name}:
            </div>
            <div style='font-size: 1.05em; color: #000000; font-weight: 600; line-height: 1.7;'>
                {formatted_advisory_html}
            </div>
        </div>
        
        <details style='background: #eff6ff; border: 1.5px solid #bfdbfe; border-radius: 8px; padding: 12px 16px; color: #1e3a8a;'>
            <summary style='font-weight: 800; cursor: pointer;'>📄 View English Agronomic Reference</summary>
            <div style='font-size: 0.94em; color: #0f172a; font-weight: 500; margin-top: 10px; line-height: 1.6;'>
                {advisory_en.replace(chr(10), "<br>")}
            </div>
        </details>
    </div>
    """
    
    return response_html, audio_response_path
