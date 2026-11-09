# AI FARMING ADVISORY – SMART AGRICULTURE FOR INDIA 🌾🇮🇳

An agentic, machine learning-powered decision support system designed specifically for Indian agriculture. The application assists farmers, agronomists, and agricultural extension officers in identifying suitable crops for their specific geographical region, soil conditions, seasonal windows, and ambient weather.

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Technology Stack](#technology-stack)
4. [Dataset Documentation & Sources](#dataset-documentation--sources)
5. [Machine Learning Pipeline & Metrics](#machine-learning-pipeline--metrics)
6. [Project Structure](#project-structure)
7. [Installation & Setup](#installation--setup)
8. [Usage Guide](#usage-guide)
9. [Scientific Advisory & Limitations](#scientific-advisory--limitations)

---

## 🌟 Overview

Agriculture in India is characterized by diverse agro-climatic zones, varying monsoon dynamics, and heterogeneous soil profiles. The **AI Farming Advisory** bridges the gap between regional agricultural science and everyday farming decisions.

Instead of predicting crop choices solely based on generic NPK values, this system evaluates:
- **Geographic & Agro-Climatic Suitability** across 17 Indian states and their representative regional zones.
- **Soil Classification & Chemistry** across 7 major Indian soil types (Black, Red, Alluvial, Laterite, Sandy, Clay, Loamy).
- **Cropping Seasons** (Kharif, Rabi, Zaid) and sowing calendars.
- **Ambient & Historical Weather Parameters** (temperature ranges, humidity, rainfall).
- **Localized Regional Explanations** detailing why a crop is recommended or should be avoided.

---

## 🚀 Key Features

### 1. 🏠 Home Dashboard
- Real-time summary metrics: 23 supported crops, 17 agricultural states, 7 soil types, 3 seasons, 13+ Indian languages.
- Core Indian Good Agricultural Practices (GAP) advisory card: Soil testing, micro-irrigation, legume crop rotation, and IPM.
- Quick feature navigation and overview of agro-climatic advisory modules.

### 2. 🎙️ Multilingual Voice Advisory (All Major Indian Languages)
- **Voice Microphone & File Upload**: Farmers can speak directly into their device microphone or upload audio recordings.
- **Support for 13+ Indian Languages**: Hindi (हिन्दी), Bengali (বাংলা), Telugu (తెలుగు), Marathi (मराठी), Tamil (தமிழ்), Gujarati (ગુજરાતી), Kannada (ಕನ್ನಡ), Malayalam (മലയാളം), Punjabi (ਪੰਜਾਬੀ), Odia (ଓଡ଼ିଆ), Urdu (اردو), Assamese (অসমীয়া), and English.
- **Automatic Language Detection**: Detects the spoken or typed language automatically.
- **Language-Synchronized Response**: The answer is returned in the **exact same Indian language** the question was asked in.
- **Spoken Audio Output (Text-to-Speech)**: Synthesizes high-fidelity voice audio so farmers can listen to the advisory in their native tongue as well as read the text.

### 3. 🌾 Region-Based Crop Recommendation (Primary Module)
- Reactive dropdowns: Selecting a state dynamically filters the available agro-ecological districts.
- Comprehensive inputs: State, District, Soil Type, Season, Temperature, Humidity, Expected Rainfall, Soil pH, and optional NPK inputs.
- Outputs **Top 5 Recommended Crops** with:
  - Estimated Model Suitability Score (%)
  - Best Growing Season & Duration
  - Optimal Temperature and Rainfall thresholds
  - Irrigation Requirements & Expected Yield
  - **In-depth Regional Agronomic Explanation** articulating why the crop matches that state, soil, and season.

### 3. 🌦️ Weather-Based Crop Advisory
- **Real-Time Live Weather Integration**: Instant one-click fetching of live ambient temperature, relative humidity, and precipitation via the Open-Meteo satellite feed for any selected district (no API keys required).
- **Manual Input Fallback**: Ability to input or adjust weather data manually with clear source labeling.
- **Advisory Deliverables**:
  - Weather-appropriate crop candidates.
  - **Crops to Avoid** under extreme weather stress (e.g., heat stress >35°C, cold shocks <15°C, waterlogging risks >1200mm, fungal blight risks >80% humidity).
  - Water & drainage management guidelines.
  - Agro-meteorological alerts and seasonal planting windows.

### 4. 🪴 Soil-Based Crop Advisory
- In-depth profiling of 7 Indian soil classes: **Black soil (Regur), Red soil, Alluvial soil, Laterite soil, Sandy soil, Clay soil, and Loamy soil**.
- Provides typical pH thresholds, nutrient profiles (N-P-K & micronutrient status), irrigation guidelines, and concrete soil improvement strategies (organic matter, green manuring, biofertilizers, gypsum/lime, broad-bed furrowing).

### 5. 🗓️ Seasonal & Regional Agro-Climatic Advisory
- Dedicated seasonal calendars and agronomic protocols for **Kharif (Monsoon)**, **Rabi (Winter)**, and **Zaid (Summer)**.
- Detailed agro-climatic zone profiles and irrigation infrastructure overview for each of the 17 Indian states.

### 6. 📖 Searchable Crop Encyclopedia
- Complete reference database for all 23 crops.
- Detailed crop profiles including standard NPK ratio, pH ranges, seedbed preparation, cultivation guides, and Integrated Pest Management (IPM) protocols.

### 7. 📊 Interactive Agricultural Visualizations (Plotly)
- **Crop Diversity by State**: Stacked distribution across Indian states.
- **Seasonal Suitability Chart**: Kharif vs. Rabi vs. Zaid crop representation.
- **Temperature Tolerances**: Thermal bounds across crops.
- **Rainfall vs. Humidity Dynamics**: Multi-variable scatter distribution.
- **Soil Compatibility Matrix**: Interactive cross-tab heatmap.

---

## 🛠️ Technology Stack

- **Core Language**: Python 3.10+
- **User Interface**: [Gradio](https://www.gradio.app/) (v6.x) with custom responsive agricultural CSS
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Random Forest Classifier & Decision Tree benchmark)
- **Interactive Visualizations**: [Plotly](https://plotly.com/python/)
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/)
- **Live Weather Data**: Open-Meteo API (WMO/IMD Global Forecast System)

---

## 📊 Dataset Documentation & Sources

### Datasets Created
1. `datasets/indian_crop_dataset.csv`: 3,454 curated observation records spanning all combinations of states, districts, soil types, and crop agronomic bounds.
2. `datasets/crop_information.csv`: Comprehensive agronomic profiles, cultivation guides, and IPM strategies for 23 major Indian crops.

### Features & Units
| Column | Type | Units / Format | Description |
|---|---|---|---|
| `State` | Categorical | String | Indian State (17 major agricultural states) |
| `District or Region` | Categorical | String | Representative agro-ecological district |
| `Soil Type` | Categorical | String | Soil class (Black, Red, Alluvial, Laterite, Sandy, Clay, Loamy) |
| `Crop Name` | Categorical | Target Class | 23 major Indian agricultural crops |
| `Season` | Categorical | Kharif, Rabi, Zaid | Sowing and harvest cropping season |
| `Temperature` | Numerical | °C | Mean ambient growing temperature |
| `Minimum Temperature`| Numerical | °C | Night/minimum threshold temperature |
| `Maximum Temperature`| Numerical | °C | Peak daytime threshold temperature |
| `Humidity` | Numerical | % | Relative humidity |
| `Rainfall` | Numerical | mm | Seasonal precipitation / water availability |
| `Soil pH` | Numerical | pH scale (4.0 - 9.0) | Soil acidity or alkalinity |
| `Nitrogen (N)` | Numerical | kg/ha | Soil available nitrogen |
| `Phosphorus (P)` | Numerical | kg/ha | Soil available phosphorus (P₂O₅) |
| `Potassium (K)` | Numerical | kg/ha | Soil available potassium (K₂O) |
| `Irrigation Requirement` | Categorical | Low, Medium, High, Very High | Water dependency |
| `Suitable Growing Period`| String | Calendar window | Sowing to harvest calendar |
| `Crop Duration` | String | Days | Total crop cycle length |
| `Expected Yield` | String | Quintals / ha | Typical harvest yield under standard management |

### Grounding & Benchmark Sources
Parameters and crop environmental thresholds are grounded in public agricultural research guidelines published by:
- **ICAR** (Indian Council of Agricultural Research) Crop Production Handbooks.
- **IMD** (India Meteorological Department) Agro-Meteorological Advisory Bulletins.
- **Directorate of Economics and Statistics (DES)**, Ministry of Agriculture & Farmers Welfare, Government of India.
- State Agricultural Universities (SAUs) package of practices (e.g., PAU Ludhiana, MPKV Rahuri, TNAU Coimbatore, UAS Bengaluru).

---

## 🧠 Machine Learning Pipeline & Metrics

### Preprocessing Architecture
- **Numerical Features**: Median imputation + `StandardScaler`.
- **Categorical Features**: Mode imputation + `OneHotEncoder(handle_unknown='ignore')`.
- Packaged seamlessly into an end-to-end `scikit-learn.pipeline.Pipeline`.

### Empirical Model Performance
Evaluated on an 80-20 stratified train-test split (2,763 training samples, 691 holdout test samples):

| Metric | Random Forest (Selected Model) | Decision Tree (Alternative) |
|---|---|---|
| **Accuracy** | **94.79%** | 88.13% |
| **F1-Score (Weighted)** | **0.9470** | 0.8820 |
| **Precision (Weighted)** | **0.9500** | 0.8860 |
| **Recall (Weighted)** | **0.9500** | 0.8813 |

Artifacts are saved in `models/crop_recommendation_model.pkl` and `models/model_metrics.json`.

---

## 📁 Project Structure

```
farmer_advisory/
│
├── app.py                         # Main Gradio interactive web application
├── requirements.txt               # Project dependencies
├── README.md                      # Complete system documentation
│
├── datasets/
│   ├── create_dataset.py          # Grounded dataset generation utility
│   ├── indian_crop_dataset.csv    # 3,454 agricultural records
│   └── crop_information.csv       # Encyclopedic crop knowledge base
│
├── models/
│   ├── crop_recommendation_model.pkl  # Trained ML pipeline
│   └── model_metrics.json         # Evaluation metrics and confusion matrix
│
├── modules/
│   ├── __init__.py
│   ├── crop_recommendation.py     # Top-5 recommendation & explanation engine
│   ├── regional_advisory.py       # State and seasonal agro-climatic profiles
│   ├── weather_advisory.py        # Weather stress analysis & planting windows
│   ├── soil_advisory.py           # 7 Indian soil types & nutrient management
│   ├── crop_information.py        # Searchable crop encyclopedia
│   └── visualizations.py          # Interactive Plotly chart builders
│
├── training/
│   └── train_model.py             # Model training, validation, and benchmarking
│
└── utils/
    ├── __init__.py
    ├── preprocessing.py           # Feature transformations & imputers
    └── weather_api.py             # Open-Meteo live weather client
```

---

## ⚡ Installation & Setup

### 1. Clone or Navigate to the Directory
```bash
cd C:\Users\Predator\.gemini\antigravity\scratch\farmer_advisory
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train the Model
```bash
python training/train_model.py
```

### 4. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to the local URL (typically `http://127.0.0.1:7860`).

---

## 🌾 Usage Guide

1. **Getting Crop Recommendations**:
   - Navigate to the **"Region-Based Crop Recommendation"** tab.
   - Pick your State (e.g. *Maharashtra*). The district dropdown will automatically update.
   - Select your District, Soil Type, and current Cropping Season.
   - Adjust Temperature, Humidity, and Expected Rainfall (or inspect the preset values).
   - Click **"RECOMMEND SUITABLE CROPS"**.
   - Review the Top 5 ranked crops, their suitability index, and localized explanations.

2. **Fetching Live Weather**:
   - Go to the **"Weather-Based Crop Advisory"** tab.
   - Select your State and District.
   - Click **"Fetch Live Weather"** to retrieve current satellite observations.
   - Click **"ANALYZE WEATHER & GENERATE ADVISORY"** to view weather-suitable crops, crops to avoid, and irrigation protocols.

3. **Exploring Soils & Seasons**:
   - Access the **"Soil-Based Crop Advisory"** and **"Seasonal Crop Advisory"** tabs for tailored soil improvement guides and regional calendars.

4. **Exploring Data Trends**:
   - View the **"Data Visualizations"** tab to interact with dynamic Plotly figures.

---

## ⚠️ Scientific Advisory & Limitations

> [!NOTE]
> 1. **Model Suitability Index**: The percentage score provided is an algorithmic suitability estimate derived from machine learning classification and verified agro-climatic boundaries. It is not an absolute biological guarantee of yield or harvest success.
> 2. **Soil Testing Recommendation**: Farmers are strongly encouraged to cross-reference recommendations with their official Government Soil Health Card (SHC) and consult local Krishi Vigyan Kendra (KVK) extension specialists before making major seed purchases.
> 3. **Market & Economic Factors**: Crop choices should additionally take into account local market mandi prices, storage infrastructure, and minimum support price (MSP) availability.
