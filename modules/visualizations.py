"""
Interactive Agricultural Data Visualizations (Light Blue & Navy Theme)
Generates Plotly interactive charts for:
1. Crop Distribution by Indian State
2. Crop Distribution by Season (Kharif, Rabi, Zaid)
3. Temperature vs Crop Optimal Ranges
4. Rainfall vs Humidity Dynamics by Crop
5. Soil Type vs Suitable Crops Heatmap
Styled in a crisp light background with deep royal blue and navy accents.
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "indian_crop_dataset.csv")

def get_dataset():
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame()

def apply_light_blue_layout(fig, title_text):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        title=dict(text=title_text, font=dict(color="#1e3a8a", size=15, family="Arial, sans-serif")),
        font=dict(family="Arial, sans-serif", color="#0f172a", size=12),
        xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1"),
        yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")
    )
    return fig

def plot_crop_distribution_by_state():
    """
    1. Crop distribution by state: Interactive stacked/grouped bar chart.
    """
    df = get_dataset()
    if df.empty:
        return go.Figure()
        
    state_crop_counts = df.groupby(["State", "Crop Name"]).size().reset_index(name="Observations")
    
    fig = px.bar(
        state_crop_counts,
        x="State",
        y="Observations",
        color="Crop Name",
        labels={"State": "Indian State", "Observations": "Cultivation Records"},
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig = apply_light_blue_layout(fig, "<b>Crop Diversity & Observation Distribution Across Indian States</b>")
    fig.update_layout(
        xaxis_tickangle=-45,
        barmode="stack",
        margin=dict(l=40, r=40, t=60, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5, font=dict(color="#0f172a"))
    )
    return fig

def plot_crop_suitability_by_season():
    """
    2. Crop suitability by season (Kharif, Rabi, Zaid).
    """
    df = get_dataset()
    if df.empty:
        return go.Figure()
        
    season_counts = df.groupby(["Season", "Crop Name"]).size().reset_index(name="Count")
    
    fig = px.bar(
        season_counts,
        x="Crop Name",
        y="Count",
        color="Season",
        labels={"Crop Name": "Crop", "Count": "Number of Seasonal Profiles"},
        color_discrete_map={"Kharif": "#0284c7", "Rabi": "#1d4ed8", "Zaid": "#0ea5e9"},
        barmode="group"
    )
    fig = apply_light_blue_layout(fig, "<b>Crop Suitability & Representation by Season (Kharif, Rabi, Zaid)</b>")
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=60, b=80),
        legend=dict(font=dict(color="#0f172a"))
    )
    return fig

def plot_temp_vs_crop():
    """
    3. Temperature versus crop suitability: Box plot showing temperature tolerances.
    """
    df = get_dataset()
    if df.empty:
        return go.Figure()
        
    fig = px.box(
        df,
        x="Crop Name",
        y="Temperature",
        color="Crop Name",
        labels={"Crop Name": "Crop", "Temperature": "Mean Growing Temperature (°C)"},
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig = apply_light_blue_layout(fig, "<b>Temperature Requirements & Thermal Tolerances Across Indian Crops (°C)</b>")
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=60, b=80)
    )
    return fig

def plot_rainfall_vs_crop():
    """
    4. Rainfall versus humidity & crop suitability: Scatter/Bubble chart.
    """
    df = get_dataset()
    if df.empty:
        return go.Figure()
        
    sample_df = df.sample(n=min(1200, len(df)), random_state=42)
    
    fig = px.scatter(
        sample_df,
        x="Rainfall",
        y="Humidity",
        color="Crop Name",
        hover_data=["State", "Season", "Temperature", "Soil Type"],
        labels={"Rainfall": "Expected Rainfall (mm)", "Humidity": "Relative Humidity (%)"},
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )
    fig = apply_light_blue_layout(fig, "<b>Rainfall (mm) vs. Relative Humidity (%) Distribution for Indian Crops</b>")
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=-0.45, xanchor="center", x=0.5, font=dict(color="#0f172a"))
    )
    return fig

def plot_soil_type_vs_crop():
    """
    5. Soil type versus suitable crops: Heatmap of occurrence.
    """
    df = get_dataset()
    if df.empty:
        return go.Figure()
        
    cross_tab = pd.crosstab(df["Crop Name"], df["Soil Type"])
    
    fig = px.imshow(
        cross_tab,
        labels=dict(x="Soil Type", y="Crop Name", color="Suitability Index"),
        x=cross_tab.columns.tolist(),
        y=cross_tab.index.tolist(),
        color_continuous_scale="Blues",
        aspect="auto"
    )
    fig = apply_light_blue_layout(fig, "<b>Soil Type vs Suitable Crops Compatibility Matrix</b>")
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=80)
    )
    return fig
