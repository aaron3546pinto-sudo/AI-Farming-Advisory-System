"""
Crop Information Encyclopedia Module
Provides a comprehensive searchable database of all supported Indian crops
including agronomic requirements, soil/weather preferences, pest management,
and best cultivation practices.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFO_PATH = os.path.join(BASE_DIR, "datasets", "crop_information.csv")

def get_crop_list():
    """
    Returns list of all available crop names in alphabetical order.
    """
    if os.path.exists(INFO_PATH):
        df = pd.read_csv(INFO_PATH)
        return sorted(df["Crop Name"].dropna().unique().tolist())
    return []

def get_crop_details(crop_name):
    """
    Retrieves full encyclopedic profile for a selected crop.
    """
    if not os.path.exists(INFO_PATH):
        return None
        
    df = pd.read_csv(INFO_PATH)
    matching = df[df["Crop Name"].str.lower() == str(crop_name).strip().lower()]
    if matching.empty:
        return None
        
    return matching.iloc[0].to_dict()

def search_crops_by_keyword(query):
    """
    Searches crops matching query in name, state, category, or soil.
    """
    if not os.path.exists(INFO_PATH):
        return pd.DataFrame()
        
    df = pd.read_csv(INFO_PATH)
    if not query:
        return df
        
    q = str(query).lower()
    mask = (
        df["Crop Name"].str.lower().str.contains(q, na=False) |
        df["Category"].str.lower().str.contains(q, na=False) |
        df["Suitable Indian States"].str.lower().str.contains(q, na=False) |
        df["Suitable Soil Types"].str.lower().str.contains(q, na=False)
    )
    return df[mask]
