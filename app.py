import streamlit as st

st.set_page_config(
    page_title="Marathwada Water Resource Analysis Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

import geemap.foliumap as geemap
import datetime
import pandas as pd
from streamlit_folium import st_folium
from folium import LayerControl
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from streamlit_echarts import st_echarts
import google.generativeai as genai
import json
import os
import ee
import base64
from io import BytesIO
import zipfile
from dotenv import load_dotenv
load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME")

ee.Authenticate()
ee.Initialize(project=PROJECT_NAME)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

def create_zones(image, num_zones=5):
    stats = image.reduceRegion(
        reducer=ee.Reducer.percentile([0, 20, 40, 60, 80, 100]),
        geometry=image.geometry(),
        scale=1000,
        maxPixels=1e9
    )
    band_name = ee.String(image.bandNames().get(0))
    thresholds = ee.List([
        stats.get(band_name.cat('_p0')),
        stats.get(band_name.cat('_p20')),
        stats.get(band_name.cat('_p40')),
        stats.get(band_name.cat('_p60')),
        stats.get(band_name.cat('_p80')),
        stats.get(band_name.cat('_p100'))
    ])
    classified = image.multiply(0)
    for i in range(num_zones):
        lower = ee.Number(thresholds.get(i))
        upper = ee.Number(thresholds.get(i + 1))
        mask = image.gte(lower).And(image.lt(upper))
        classified = classified.where(mask, ee.Number(i + 1))
    return classified, thresholds

def get_time_series(collection, band, region, start_date, end_date):
    # Filter out images without a valid timestamp
    collection = collection.filter(ee.Filter.notNull(['system:time_start']))

    def calculate_stats(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=1000,
            maxPixels=1e9
        )
        return ee.Feature(None, {
            'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
            'value': stats.get(band)
        })

    features = collection.map(calculate_stats).getInfo()['features']

    return [
        {'date': f['properties']['date'], 'value': f['properties']['value']}
        for f in features
        if f['properties'].get('date') and f['properties'].get('value') is not None
    ]

@st.cache_data
def gemini_insights(parameter_name, df):
    prompt = f"""
    You are a hydrology and remote sensing expert. Analyze the following time series data for {parameter_name} in the Marathwada region of Maharashtra, India.\nProvide a concise summary of trends, anomalies, and actionable insights for water resource management.\nData:\n{df.to_csv(index=False)}
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

def add_zoned_layer(Map, image, name, palette):
    zones, thresholds = create_zones(image)
    Map.addLayer(zones, {'min': 1, 'max': 5, 'palette': palette}, f'{name} Zones')
    return zones, thresholds

def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href

def get_all_data_download_link(data_dict):
    """Create a zip file with all datasets and return a download link"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_name, df in data_dict.items():
            csv_data = df.to_csv(index=False)
            zip_file.writestr(f"{file_name}.csv", csv_data)

    zip_buffer.seek(0)
    b64 = base64.b64encode(zip_buffer.getvalue()).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="marathwada_data.zip">Download All Datasets</a>'
    return href

def create_monthly_aggregation(df):
    """Aggregate data by month"""
    if df.empty:
        return pd.DataFrame()

    df['month'] = df['date'].dt.month
    monthly_data = df.groupby('month')['value'].agg(['mean', 'min', 'max', 'std']).reset_index()
    monthly_data['month_name'] = monthly_data['month'].apply(lambda x: calendar.month_abbr[x])
    return monthly_data

def create_seasonal_aggregation(df):
    """Aggregate data by season"""
    if df.empty:
        return pd.DataFrame()

    # Define seasons: Winter (Dec-Feb), Summer (Mar-May), Monsoon (Jun-Sep), Post-Monsoon (Oct-Nov)
    season_map = {
        1: 'Winter', 2: 'Winter',
        3: 'Summer', 4: 'Summer', 5: 'Summer',
        6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
        10: 'Post-Monsoon', 11: 'Post-Monsoon',
        12: 'Winter'
    }

    df['season'] = df['date'].dt.month.map(season_map)
    seasonal_data = df.groupby('season')['value'].agg(['mean', 'min', 'max', 'std']).reset_index()
    # Reorder seasons for display
    season_order = {'Winter': 0, 'Summer': 1, 'Monsoon': 2, 'Post-Monsoon': 3}
    seasonal_data['order'] = seasonal_data['season'].map(season_order)
    seasonal_data = seasonal_data.sort_values('order').drop('order', axis=1)

    return seasonal_data

def render_metrics(df, parameter):
    """Render metric cards for key statistics"""
    if df.empty:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label=f"Average {parameter}", value=f"{df['value'].mean():.3f}")
    with col2:
        st.metric(label=f"Maximum {parameter}", value=f"{df['value'].max():.3f}")
    with col3:
        st.metric(label=f"Minimum {parameter}", value=f"{df['value'].min():.3f}")
    with col4:
        st.metric(label=f"Standard Deviation", value=f"{df['value'].std():.3f}")

    style_metric_cards()

@st.cache_data
def load_data(start_date, end_date):
    # Define region
    india_admin = ee.FeatureCollection("FAO/GAUL/2015/level2")
    maharashtra = india_admin.filter(ee.Filter.eq('ADM1_NAME', 'Maharashtra'))
    marathwada_districts = maharashtra.filter(ee.Filter.inList('ADM2_NAME', [
        'Aurangabad', 'Bid', 'Jalna', 'Latur', 'Nanded', 'Osmanabad', 'Parbhani', 'Hingoli'
    ]))
    region = marathwada_districts.geometry()

    # DEM & Slope
    elev = ee.Image('USGS/SRTMGL1_003').clip(region)
    dem = elev.select('elevation')
    slope = ee.Terrain.slope(elev).unmask(0)

    # LULC (ESA WorldCover)
    lulc = ee.ImageCollection("ESA/WorldCover/v100") \
        .filterDate("2020-01-01", "2020-12-31") \
        .first() \
        .select('Map') \
        .clip(region) \
        .unmask(60)

    # Soil Texture
    soil = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02') \
        .select('b0') \
        .clip(region) \
        .unmask(10)

    # Geomorphology
    geom = ee.Image('CSP/ERGo/1_0/Global/ALOS_landforms') \
        .select('constant') \
        .clip(region) \
        .unmask(11)

    # Rainfall
    rainfall = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(start_date, end_date)
    rainfall_mean = rainfall.mean().clip(region)

    # Sentinel-2 NDVI & NDWI
    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterDate(start_date, end_date) \
        .filterBounds(region) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  # Filter out cloudy images

    # Calculate NDVI and NDWI with error handling
    def calculate_ndvi(img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
        return img.addBands(ndvi)

    def calculate_ndwi(img):
        ndwi = img.normalizedDifference(['B3', 'B8']).rename('ndwi')
        return img.addBands(ndwi)

    s2_with_indices = s2.map(calculate_ndvi).map(calculate_ndwi)
    
    ndvi = s2_with_indices.select('ndvi')
    ndvi_mean = ndvi.mean().clip(region)

    ndwi = s2_with_indices.select('ndwi')
    ndwi_mean = ndwi.mean().clip(region)

    # Soil Moisture
    soil_moisture = ee.ImageCollection('NASA/SMAP/SPL3SMP_E/005') \
        .filterDate(start_date, end_date) \
        .select('soil_moisture_am')
    soil_moisture_mean = soil_moisture.mean().clip(region)

    # Distance to Water
    ndwi_median = ndwi_mean
    water_mask = ndwi_median.gt(0.2)
    distance_to_water = water_mask.fastDistanceTransform(30).sqrt().clip(region).rename('dist_to_water')

    # Runoff Coefficient (example weighted)
    rc = lulc.multiply(0.3).add(soil.multiply(0.3)).add(slope.multiply(0.4)).rename('runoff_coeff')

    # Time Series DataFrames with error handling
    try:
        ndvi_ts = get_time_series(ndvi, 'ndvi', region, start_date, end_date)
        ndvi_df = pd.DataFrame(ndvi_ts)
        if not ndvi_df.empty:
            ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
            print(f"NDVI data points: {len(ndvi_df)}")  # Debug print
    except Exception as e:
        print(f"Error processing NDVI time series: {str(e)}")
        ndvi_df = pd.DataFrame()

    try:
        ndwi_ts = get_time_series(ndwi, 'ndwi', region, start_date, end_date)
        ndwi_df = pd.DataFrame(ndwi_ts)
        if not ndwi_df.empty:
            ndwi_df['date'] = pd.to_datetime(ndwi_df['date'])
            print(f"NDWI data points: {len(ndwi_df)}")  # Debug print
    except Exception as e:
        print(f"Error processing NDWI time series: {str(e)}")
        ndwi_df = pd.DataFrame()

    rainfall_ts = get_time_series(rainfall, 'precipitation', region, start_date, end_date)
    rainfall_df = pd.DataFrame(rainfall_ts)
    if not rainfall_df.empty:
        rainfall_df['date'] = pd.to_datetime(rainfall_df['date'])

    soil_moisture_ts = get_time_series(soil_moisture, 'soil_moisture_am', region, start_date, end_date)
    soil_moisture_df = pd.DataFrame(soil_moisture_ts)
    if not soil_moisture_df.empty:
        soil_moisture_df['date'] = pd.to_datetime(soil_moisture_df['date'])

    # Return all layers and time series
    return {
        'region': region,
        'rainfall_mean': rainfall_mean,
        'ndvi_mean': ndvi_mean,
        'ndwi_mean': ndwi_mean,
        'soil_moisture_mean': soil_moisture_mean,
        'elevation': dem,
        'slope': slope,
        'lulc': lulc,
        'soil': soil,
        'geom': geom,
        'dist_to_water': distance_to_water,
        'runoff_coeff': rc,
        'rainfall_df': rainfall_df,
        'ndvi_df': ndvi_df,
        'ndwi_df': ndwi_df,
        'soil_moisture_df': soil_moisture_df
    }

palette_dict = {
    'rainfall': ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'],
    'ndvi': ['#ffffcc', '#c2e699', '#78c679', '#31a354', '#006837'],
    'ndwi': ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#225ea8'],
    'soil_moisture': ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'],
    'elevation': ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32'],
    'slope': ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#f03b20', '#bd0026'],
    'lulc': ['#d9f0a3', '#addd8e', '#78c679', '#31a354', '#006837'],
    'soil': ['#f7fcb9', '#addd8e', '#31a354'],
    'geom': ['#f7fcf0', '#e0f3db', '#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3', '#2b8cbe', '#0868ac', '#084081'],
    'dist_to_water': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'],
    'runoff_coeff': ['#ffffcc', '#ffeda0', '#feb24c', '#fd8d3c', '#f03b20', '#bd0026']
}

color_map = {
    "Rainfall": '#1E88E5',
    "NDVI": '#43A047',
    "NDWI": '#00ACC1',
    "Soil Moisture": '#FB8C00',
    "Elevation": '#7B1FA2',
    "Slope": '#F4511E',
    "LULC": '#00897B',
    "Soil Texture": '#6D4C41',
    "Geomorphology": '#3949AB',
    "Distance to Water": '#0288D1',
    "Runoff Coefficient": '#C62828'
}

# Add this mapping for zone meanings at the top of your file
zone_meanings = {
    "Rainfall": [
        "Very Low Rainfall", "Low Rainfall", "Moderate Rainfall", "High Rainfall", "Very High Rainfall"
    ],
    "NDVI": [
        "Very Poor Vegetation", "Poor Vegetation", "Moderate Vegetation", "Good Vegetation", "Very Good Vegetation"
    ],
    "NDWI": [
        "Very Low Water Content", "Low Water Content", "Moderate Water Content", "High Water Content", "Very High Water Content"
    ],
    "Soil Moisture": [
        "Very Dry Soil", "Dry Soil", "Moderate Moisture", "Moist Soil", "Very Moist Soil"
    ],
    "Elevation": [
        "Very Low Elevation", "Low Elevation", "Moderate Elevation", "High Elevation", "Very High Elevation"
    ],
    "DEM": [
        "Very Low DEM", "Low DEM", "Moderate DEM", "High DEM", "Very High DEM"
    ],
    "Slope": [
        "Very Gentle Slope", "Gentle Slope", "Moderate Slope", "Steep Slope", "Very Steep Slope"
    ],
    "LULC": [
        "Mostly Water/Urban", "Mostly Cropland", "Mixed Land Use", "Mostly Forest", "Dense Forest/Natural"
    ],
    "Soil Texture": [
        "Very Fine Texture", "Fine Texture", "Moderate Texture", "Coarse Texture", "Very Coarse Texture"
    ],
    "Geomorphology": [
        "Plains", "Low Hills", "Moderate Hills", "High Hills", "Mountains"
    ]
}

def show_legend(name, thresholds, palette):
    st.markdown(f"**{name} Zones**")
    meanings = zone_meanings.get(name, ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5"])
    for i in range(5):
        st.markdown(
            f'<div style="display: flex; align-items: center;">'
            f'<div style="width: 20px; height: 20px; background: {palette[i]}; margin-right: 8px; border: 1px solid #888;"></div>'
            f'Zone {i+1}: {thresholds[i]:.2f} - {thresholds[i+1]:.2f} <br>'
            f'<span style="font-size:0.9em;color:#555;">{meanings[i]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #f0f2f6;
        box-shadow: 0 0.15rem 0.5rem rgba(0, 0, 0, 0.1);
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 0.3rem solid #1976D2;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">Marathwada Water Resource Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="highlight">
This dashboard provides comprehensive analysis of water resources in the Marathwada region of Maharashtra, India using Earth Engine satellite data and AI-powered insights.
</div>
""", unsafe_allow_html=True)

# Create sidebar for controls
with st.sidebar:
    st.title("Dashboard Controls")

    # Date range selection
    st.header("Analysis Period")
    default_start = datetime.date(2023, 1, 1)
    default_end = datetime.date(2023, 12, 31)

    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", default_end)

    # Convert to string format for Earth Engine
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    st.header("Dataset Information")
    st.markdown("""
    - **Rainfall**: CHIRPS daily precipitation
    - **NDVI**: Normalized Difference Vegetation Index (Sentinel-2)
    - **NDWI**: Normalized Difference Water Index (Sentinel-2)
    - **Soil Moisture**: NASA SMAP soil moisture product
    - **Elevation**: SRTM Digital Elevation Model
    """)

    # Button to load data
    data_load = st.button("Load/Refresh Data", type="primary")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This dashboard visualizes key water resource indicators for the Marathwada region using Earth Engine data and provides AI-generated insights through Google's Gemini model.
    """)

# Initialize or load data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if data_load or not st.session_state.data_loaded:
    with st.spinner("Loading data from Earth Engine... This may take a moment."):
        data = load_data(start_date_str, end_date_str)
        st.session_state.data = data
        st.session_state.data_loaded = True

        # Generate AI insights for each parameter
        with st.spinner("Generating AI insights..."):
            st.session_state.rainfall_insight = gemini_insights('Rainfall', data['rainfall_df'])
            st.session_state.ndvi_insight = gemini_insights('NDVI', data['ndvi_df'])
            st.session_state.ndwi_insight = gemini_insights('NDWI', data['ndwi_df'])
            st.session_state.soil_moisture_insight = gemini_insights('Soil Moisture', data['soil_moisture_df'])

    st.success("Data loaded successfully!")

# Check if data is loaded before proceeding
if not st.session_state.data_loaded:
    st.info("Please load data using the sidebar controls.")
    st.stop()

# Access loaded data
data = st.session_state.data

# Create tabs for organization
tabs = st.tabs([
    "📊 Dashboard Overview",
    "🗺️ Geographic Analysis",
    "📈 Time Series Analysis",
    "🔍 Seasonal Patterns",
    "🤖 AI Insights",
    "💾 Data Download"
])

with tabs[0]:
    st.markdown('<h2 class="sub-header">Water Resource Dashboard Overview</h2>', unsafe_allow_html=True)

    # Key metrics overview
    st.markdown("### Key Metrics")
    metrics_col1, metrics_col2 = st.columns(2)

    with metrics_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Rainfall")
        render_metrics(data['rainfall_df'], "Rainfall (mm)")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### NDVI (Vegetation Health)")
        render_metrics(data['ndvi_df'], "NDVI")
        st.markdown('</div>', unsafe_allow_html=True)

    with metrics_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### NDWI (Water Content)")
        render_metrics(data['ndwi_df'], "NDWI")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Soil Moisture")
        render_metrics(data['soil_moisture_df'], "Soil Moisture")
        st.markdown('</div>', unsafe_allow_html=True)

    # Summary charts
    st.markdown("### Summary of Water Resource Indicators")

    # Combined line chart for all parameters
    fig = go.Figure()

    # Normalize the data for better comparison (0-1 scale)
    def normalize_series(df):
        if df.empty:
            return df
        min_val = df['value'].min()
        max_val = df['value'].max()
        df['normalized'] = (df['value'] - min_val) / (max_val - min_val) if max_val > min_val else df['value']
        return df

    rainfall_norm = normalize_series(data['rainfall_df'].copy())
    ndvi_norm = normalize_series(data['ndvi_df'].copy())
    ndwi_norm = normalize_series(data['ndwi_df'].copy())
    soil_moisture_norm = normalize_series(data['soil_moisture_df'].copy())

    # Add traces with normalized values for comparison
    if not rainfall_norm.empty:
        fig.add_trace(go.Scatter(
            x=rainfall_norm['date'],
            y=rainfall_norm['normalized'],
            name='Rainfall',
            line=dict(color='#1E88E5', width=2)
        ))

    if not ndvi_norm.empty:
        fig.add_trace(go.Scatter(
            x=ndvi_norm['date'],
            y=ndvi_norm['normalized'],
            name='NDVI',
            line=dict(color='#43A047', width=2)
        ))

    if not ndwi_norm.empty:
        fig.add_trace(go.Scatter(
            x=ndwi_norm['date'],
            y=ndwi_norm['normalized'],
            name='NDWI',
            line=dict(color='#00ACC1', width=2)
        ))

    if not soil_moisture_norm.empty:
        fig.add_trace(go.Scatter(
            x=soil_moisture_norm['date'],
            y=soil_moisture_norm['normalized'],
            name='Soil Moisture',
            line=dict(color='#FB8C00', width=2)
        ))

    # Update layout
    fig.update_layout(
        title="Normalized Time Series Comparison (0-1 scale)",
        xaxis_title="Date",
        yaxis_title="Normalized Value",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Quick insights from AI
    st.markdown("### Key Insights")
    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### NDVI vs Rainfall")

        # Create scatter plot of NDVI vs Rainfall
        if not data['ndvi_df'].empty and not data['rainfall_df'].empty:
            # Merge dataframes
            merged_df = pd.merge(
                data['ndvi_df'],
                data['rainfall_df'],
                left_on='date',
                right_on='date',
                suffixes=('_ndvi', '_rainfall')
            )

            fig = px.scatter(
                merged_df,
                x='value_rainfall',
                y='value_ndvi',
                trendline="ols",
                labels={
                    'value_rainfall': 'Rainfall (mm)',
                    'value_ndvi': 'NDVI'
                },
                title="Vegetation Health vs Rainfall Correlation"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for correlation analysis.")

        st.markdown('</div>', unsafe_allow_html=True)

    with insight_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Water Availability Trends")

        # Create composite water index
        if not data['ndwi_df'].empty and not data['soil_moisture_df'].empty:
            # Merge dataframes
            water_df = pd.merge(
                data['ndwi_df'],
                data['soil_moisture_df'],
                left_on='date',
                right_on='date',
                suffixes=('_ndwi', '_soil')
            )

            # Create monthly aggregates
            water_df['month'] = water_df['date'].dt.month
            water_df['month_name'] = water_df['date'].dt.month.apply(lambda x: calendar.month_abbr[x])

            # Create monthly average
            monthly_water = water_df.groupby('month').agg({
                'value_ndwi': 'mean',
                'value_soil': 'mean'
            }).reset_index()

            monthly_water['month_name'] = monthly_water['month'].apply(lambda x: calendar.month_abbr[x])

            # Order by month
            monthly_water = monthly_water.sort_values('month')

            # Create side-by-side bar chart
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=monthly_water['month_name'],
                y=monthly_water['value_ndwi'],
                name='NDWI',
                marker_color='#00ACC1'
            ))

            fig.add_trace(go.Bar(
                x=monthly_water['month_name'],
                y=monthly_water['value_soil'],
                name='Soil Moisture',
                marker_color='#FB8C00'
            ))

            fig.update_layout(
                title="Monthly Water Availability Indicators",
                xaxis_title="Month",
                yaxis_title="Value",
                barmode='group'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for water availability analysis.")

        st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<h2 class="sub-header">Geographic Analysis</h2>', unsafe_allow_html=True)
    st.markdown("""
    This map shows the distribution of key water resource indicators across the Marathwada region.
    Use the layer control on the top right to toggle different indicators.
    """)

    geo_parameters = [
        ("Rainfall", 'rainfall_mean', palette_dict['rainfall']),
        ("NDVI", 'ndvi_mean', palette_dict['ndvi']),
        ("NDWI", 'ndwi_mean', palette_dict['ndwi']),
        ("Soil Moisture", 'soil_moisture_mean', palette_dict['soil_moisture']),
        ("Elevation", 'elevation', palette_dict['elevation']),
        ("DEM", 'elevation', palette_dict['elevation']),
        ("Slope", 'slope', palette_dict['slope']),
        ("LULC", 'lulc', palette_dict['lulc']),
        ("Soil Texture", 'soil', palette_dict['soil']),
        ("Geomorphology", 'geom', palette_dict['geom'])
    ]
    geo_param_names = [p[0] for p in geo_parameters]
    selected_geo_param = st.selectbox("Select Parameter to Display on Map", geo_param_names)

    overview_map = geemap.Map(center=[19.0, 76.5], zoom=7)

    # Only add the selected layer
    for name, key, palette in geo_parameters:
        if name == selected_geo_param:
            zones, thresholds = add_zoned_layer(overview_map, data[key], name, palette)
            break

    map_col, legend_col = st.columns([4, 1])
    with map_col:
        overview_map.to_streamlit(height=600)
    with legend_col:
        show_legend(selected_geo_param, thresholds.getInfo(), palette)

    # Simplified insights section
    st.markdown("### Geographic Insights")
    geo_col1, geo_col2 = st.columns(2)

    with geo_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        #### Zone Analysis Guide

        The map displays five zones (1-5) for each parameter:
        - **Zone 1**: Lowest values (0-20th percentile)
        - **Zone 2**: Low values (20-40th percentile)
        - **Zone 3**: Medium values (40-60th percentile)
        - **Zone 4**: High values (60-80th percentile)
        - **Zone 5**: Highest values (80-100th percentile)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with geo_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        #### Interpretation Tips

        When analyzing the map:
        - **NDVI**: Higher values indicate healthier vegetation
        - **NDWI**: Higher values indicate more surface water content
        - **Rainfall**: Higher values indicate greater precipitation
        - **Soil Moisture**: Higher values indicate wetter soils
        """)
        st.markdown('</div>', unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<h2 class="sub-header">Time Series Analysis</h2>', unsafe_allow_html=True)

    # Parameter selection
    # Parameter selection
    parameters = [
        "All Parameters",
        "Rainfall",
        "NDVI",
        "NDWI",
        "Soil Moisture",
        "Elevation",
        "Slope",
        "LULC",
        "Soil Texture",
        "Geomorphology",
        "Distance to Water",
        "Runoff Coefficient"
    ]
    selected_param = st.selectbox("Select Parameter to Analyze", parameters)

    # Time series visualization
    if selected_param == "All Parameters":
        # Create time series plots for all parameters
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=("Rainfall", "NDVI", "NDWI", "Soil Moisture"),
            vertical_spacing=0.1,
            shared_xaxes=True
        )

        # Add traces for each parameter
        if not data['rainfall_df'].empty:
            fig.add_trace(
                go.Scatter(
                    x=data['rainfall_df']['date'],
                    y=data['rainfall_df']['value'],
                    name='Rainfall',
                    line=dict(color='#1E88E5', width=2)
                ),
                row=1, col=1
            )

        if not data['ndvi_df'].empty:
            fig.add_trace(
                go.Scatter(
                    x=data['ndvi_df']['date'],
                    y=data['ndvi_df']['value'],
                    name='NDVI',
                    line=dict(color='#43A047', width=2)
                ),
                row=2, col=1
            )

        if not data['ndwi_df'].empty:
            fig.add_trace(
                go.Scatter(
                    x=data['ndwi_df']['date'],
                    y=data['ndwi_df']['value'],
                    name='NDWI',
                    line=dict(color='#00ACC1', width=2)
                ),
                row=3, col=1
            )

        if not data['soil_moisture_df'].empty:
            fig.add_trace(
                go.Scatter(
                    x=data['soil_moisture_df']['date'],
                    y=data['soil_moisture_df']['value'],
                    name='Soil Moisture',
                    line=dict(color='#FB8C00', width=2)
                ),
                row=4, col=1
            )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="Time Series Analysis of All Parameters",
            showlegend=False
        )

        # Update y-axis labels
        fig.update_yaxes(title_text="Rainfall (mm)", row=1, col=1)
        fig.update_yaxes(title_text="NDVI", row=2, col=1)
        fig.update_yaxes(title_text="NDWI", row=3, col=1)
        fig.update_yaxes(title_text="Soil Moisture", row=4, col=1)

        # Update x-axis label
        fig.update_xaxes(title_text="Date", row=4, col=1)

        st.plotly_chart(fig, use_container_width=True)

    else:
        # Single parameter plot
        # Single parameter plot
        param_map = {
            "Rainfall": data['rainfall_df'],
            "NDVI": data['ndvi_df'],
            "NDWI": data['ndwi_df'],
            "Soil Moisture": data['soil_moisture_df'],
            "Elevation": pd.DataFrame({
                'date': pd.Timestamp('today'),
                'value': data['elevation'].reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=data['region'],
                    scale=1000,
                    maxPixels=1e9
                ).getInfo()['elevation']
            }, index=[0]),
            "Slope": pd.DataFrame({
                'date': pd.Timestamp('today'),
                'value': data['slope'].reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=data['region'],
                    scale=1000,
                    maxPixels=1e9
                ).getInfo()['slope']
            }, index=[0]),
            "LULC": pd.DataFrame({
                'date': pd.Timestamp('today'),
                'value': data['lulc'].reduceRegion(
                    reducer=ee.Reducer.mode(),
                    geometry=data['region'],
                    scale=1000,
                    maxPixels=1e9
                ).getInfo()['Map']
            }, index=[0]),
            "Soil Texture": pd.DataFrame({
                'date': pd.Timestamp('today'),
                'value': data['soil'].reduceRegion(
                    reducer=ee.Reducer.mode(),
                    geometry=data['region'],
                    scale=1000,
                    maxPixels=1e9
                ).getInfo()['b0']
            }, index=[0]),
            "Geomorphology": pd.DataFrame({
                'date': pd.Timestamp('today'),
                'value': data['geom'].reduceRegion(
                    reducer=ee.Reducer.mode(),
                    geometry=data['region'],
                    scale=1000,
                    maxPixels=1e9
                ).getInfo()['constant']
            }, index=[0]),
            "Distance to Water": pd.DataFrame({
                'date': pd.Timestamp('today'),
                'value': data['dist_to_water'].reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=data['region'],
                    scale=1000,
                    maxPixels=1e9
                ).getInfo()['dist_to_water']
            }, index=[0]),
            "Runoff Coefficient": pd.DataFrame({
                'date': pd.Timestamp('today'),
                'value': data['runoff_coeff'].reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=data['region'],
                    scale=1000,
                    maxPixels=1e9
                ).getInfo()['runoff_coeff']
            }, index=[0])
        }

        color_map = {
            "Rainfall": '#1E88E5',
            "NDVI": '#43A047',
            "NDWI": '#00ACC1',
            "Soil Moisture": '#FB8C00',
            "Elevation": '#7B1FA2',
            "Slope": '#F4511E',
            "LULC": '#00897B',
            "Soil Texture": '#6D4C41',
            "Geomorphology": '#3949AB',
            "Distance to Water": '#0288D1',
            "Runoff Coefficient": '#C62828'
        }

        df = param_map[selected_param]

        if not df.empty:
            # Create detailed time series plot with moving average
            fig = go.Figure()

            # Add raw data
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['value'],
                    name=selected_param,
                    line=dict(color=color_map[selected_param], width=2)
                )
            )

            # Add 7-day moving average if enough data points
            if len(df) > 7:
                df['rolling_avg'] = df['value'].rolling(window=7, min_periods=1).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['rolling_avg'],
                        name='7-day Moving Average',
                        line=dict(color='#D81B60', width=2, dash='dash')
                    )
                )

            # Update layout
            fig.update_layout(
                title=f"{selected_param} Time Series Analysis",
                xaxis_title="Date",
                yaxis_title=selected_param,
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Statistical analysis
            st.markdown("### Statistical Analysis")
            stat_col1, stat_col2 = st.columns(2)

            with stat_col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"#### {selected_param} Distribution")

                # Create histogram
                fig = px.histogram(
                    df,
                    x='value',
                    nbins=20,
                    title=f"{selected_param} Distribution",
                    color_discrete_sequence=[color_map[selected_param]]
                )

                fig.update_layout(
                    xaxis_title=selected_param,
                    yaxis_title="Frequency",
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with stat_col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"#### {selected_param} Summary Statistics")

                # Calculate summary statistics
                stats = df['value'].describe()

                # Create a DataFrame for display
                stats_df = pd.DataFrame({
                    'Statistic': stats.index,
                    'Value': stats.values
                })

                # Format values to 3 decimal places
                stats_df['Value'] = stats_df['Value'].apply(lambda x: f"{x:.3f}")

                # Display statistics
                st.dataframe(stats_df, use_container_width=True)

                # Calculate additional metrics
                if len(df) > 1:
                    trend = df['value'].iloc[-1] - df['value'].iloc[0]
                    trend_percent = (trend / df['value'].iloc[0]) * 100 if df['value'].iloc[0] != 0 else 0

                    st.markdown(f"**Overall Trend:** {trend:.3f} ({trend_percent:.2f}%)")

                    # Calculate coefficient of variation
                    cv = (stats['std'] / stats['mean']) * 100 if stats['mean'] != 0 else 0
                    st.markdown(f"**Coefficient of Variation:** {cv:.2f}%")

                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info(f"No {selected_param} data available for the selected time period.")

with tabs[3]:
    st.markdown('<h2 class="sub-header">Seasonal Analysis</h2>', unsafe_allow_html=True)

    # Create seasonal analysis for each parameter
    st.markdown("### Monthly and Seasonal Patterns")

    # Parameter selection for seasonal analysis
    # Parameter selection for seasonal analysis
    seasonal_param = st.selectbox(
        "Select Parameter for Seasonal Analysis",
        ["Rainfall", "NDVI", "NDWI", "Soil Moisture", "Runoff Coefficient"]
    )

    # Map parameters to dataframes
    seasonal_param_map = {
        "Rainfall": data['rainfall_df'],
        "NDVI": data['ndvi_df'],
        "NDWI": data['ndwi_df'],
        "Soil Moisture": data['soil_moisture_df']
    }

    df = seasonal_param_map[seasonal_param]

    if not df.empty:
        # Monthly aggregation
        monthly_data = create_monthly_aggregation(df)

        # Seasonal aggregation
        seasonal_data = create_seasonal_aggregation(df)

        # Create visualizations
        seasonal_col1, seasonal_col2 = st.columns(2)

        with seasonal_col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"#### Monthly {seasonal_param} Patterns")

            # Create monthly bar chart
            fig = go.Figure()

            # Add mean values with error bars
            fig.add_trace(go.Bar(
                x=monthly_data['month_name'],
                y=monthly_data['mean'],
                name='Mean',
                error_y=dict(
                    type='data',
                    array=monthly_data['std'],
                    visible=True
                ),
                marker_color=color_map[seasonal_param]
            ))

            # Update layout
            fig.update_layout(
                title=f"Monthly {seasonal_param} Patterns",
                xaxis=dict(
                    title="Month",
                    categoryorder='array',
                    categoryarray=[calendar.month_abbr[i] for i in range(1, 13)]
                ),
                yaxis_title=seasonal_param,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with seasonal_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"#### Seasonal {seasonal_param} Patterns")

            # Create seasonal bar chart
            fig = go.Figure()

            # Add mean values with error bars
            fig.add_trace(go.Bar(
                x=seasonal_data['season'],
                y=seasonal_data['mean'],
                name='Mean',
                error_y=dict(
                    type='data',
                    array=seasonal_data['std'],
                    visible=True
                ),
                marker_color=color_map[seasonal_param]
            ))

            # Update layout
            fig.update_layout(
                title=f"Seasonal {seasonal_param} Patterns",
                xaxis_title="Season",
                yaxis_title=seasonal_param,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Seasonal insights
        st.markdown("### Seasonal Insights")

        # Month with highest and lowest values
        if not monthly_data.empty:
            max_month_idx = monthly_data['mean'].idxmax()
            min_month_idx = monthly_data['mean'].idxmin()

            max_month = monthly_data.loc[max_month_idx, 'month_name']
            min_month = monthly_data.loc[min_month_idx, 'month_name']

            max_val = monthly_data.loc[max_month_idx, 'mean']
            min_val = monthly_data.loc[min_month_idx, 'mean']

            insight_col1, insight_col2 = st.columns(2)

            with insight_col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Monthly Highlights")
                st.markdown(f"""
                - **Highest {seasonal_param}**: {max_month} ({max_val:.3f})
                - **Lowest {seasonal_param}**: {min_month} ({min_val:.3f})
                - **Variability**: {(max_val - min_val):.3f} range throughout the year
                """)
                st.markdown('</div>', unsafe_allow_html=True)

            with insight_col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Seasonal Variability")

                # Calculate coefficient of variation for each season
                if not seasonal_data.empty:
                    seasonal_data['cv'] = (seasonal_data['std'] / seasonal_data['mean']) * 100

                    # Create horizontal bar chart for CV
                    fig = px.bar(
                        seasonal_data,
                        y='season',
                        x='cv',
                        orientation='h',
                        title=f"Seasonal Variability ({seasonal_param})",
                        labels={'cv': 'Coefficient of Variation (%)', 'season': 'Season'},
                        color='cv',
                        color_continuous_scale=px.colors.sequential.Viridis
                    )

                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info(f"No {seasonal_param} data available for the selected time period.")

with tabs[4]:
    st.markdown('<h2 class="sub-header">AI-Powered Insights</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight">
    This section provides AI-generated insights based on the analysis of water resource data for the Marathwada region.
    Insights are generated using Google's Gemini AI model and are refreshed each time data is loaded.
    </div>
    """, unsafe_allow_html=True)

    # Parameter selection for insights
    # Parameter selection for insights
    insight_param = st.selectbox(
        "Select Parameter for AI Insights",
        ["Rainfall", "NDVI", "NDWI", "Soil Moisture", "Runoff Coefficient", "Land Use & Terrain", "Comprehensive Analysis"]
    )

    if insight_param == "Rainfall":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Rainfall Insights")
        st.markdown(st.session_state.rainfall_insight)
        st.markdown('</div>', unsafe_allow_html=True)

    elif insight_param == "NDVI":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### NDVI (Vegetation Health) Insights")
        st.markdown(st.session_state.ndvi_insight)
        st.markdown('</div>', unsafe_allow_html=True)

    elif insight_param == "NDWI":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### NDWI (Water Content) Insights")
        st.markdown(st.session_state.ndwi_insight)
        st.markdown('</div>', unsafe_allow_html=True)

    elif insight_param == "Soil Moisture":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Soil Moisture Insights")
        st.markdown(st.session_state.soil_moisture_insight)
        st.markdown('</div>', unsafe_allow_html=True)

    elif insight_param == "Runoff Coefficient":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Runoff Coefficient Insights")
        # Generate runoff coefficient insights
        runoff_prompt = f"""
        You are a hydrology and remote sensing expert. Analyze the runoff coefficient for the Marathwada region of Maharashtra, India.
        Provide insights on water harvesting potential, flood risks, and water conservation strategies based on the runoff values.
        The average runoff coefficient for the region is approximately {data['runoff_coeff'].reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=data['region'],
            scale=1000,
            maxPixels=1e9
        ).getInfo()['runoff_coeff']}.
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        runoff_insights = model.generate_content(runoff_prompt).text
        st.markdown(runoff_insights)
        st.markdown('</div>', unsafe_allow_html=True)

    elif insight_param == "Land Use & Terrain":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Land Use & Terrain Analysis")
        # Generate land use and terrain insights
        terrain_prompt = f"""
        You are a hydrology and remote sensing expert. Analyze how land use patterns and terrain characteristics
        (elevation, slope, soil texture, geomorphology) affect water resources in the Marathwada region of Maharashtra, India.
        Provide insights on water conservation strategies based on these physical parameters.
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        terrain_insights = model.generate_content(terrain_prompt).text
        st.markdown(terrain_insights)
        st.markdown('</div>', unsafe_allow_html=True)

    else:  # Comprehensive Analysis
        # Generate integrated insights
        with st.spinner("Generating comprehensive analysis..."):
            # Prepare data summary for AI
            summary_data = {
                'rainfall_stats': data['rainfall_df']['value'].describe().to_dict() if not data['rainfall_df'].empty else {},
                'ndvi_stats': data['ndvi_df']['value'].describe().to_dict() if not data['ndvi_df'].empty else {},
                'ndwi_stats': data['ndwi_df']['value'].describe().to_dict() if not data['ndwi_df'].empty else {},
                'soil_moisture_stats': data['soil_moisture_df']['value'].describe().to_dict() if not data['soil_moisture_df'].empty else {}
            }

            # Convert to JSON for prompt
            summary_json = json.dumps(summary_data)

            # Generate comprehensive insights
            comprehensive_prompt = f"""
            You are a hydrology and remote sensing expert. Analyze the following summary statistics for water resource indicators in the Marathwada region of Maharashtra, India.

            Provide a comprehensive analysis that integrates these indicators to assess the overall water resource situation. Focus on:
            1. Water scarcity/abundance assessment
            2. Vegetation health in relation to water availability
            3. Recommendations for water resource management
            4. Potential areas of concern

            Statistics Summary:
            {summary_json}

            The analysis period is from {start_date} to {end_date}.
            """

            model = genai.GenerativeModel('gemini-1.5-flash')
            comprehensive_response = model.generate_content(comprehensive_prompt)
            comprehensive_insight = comprehensive_response.text

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Comprehensive Water Resource Analysis")
        st.markdown(comprehensive_insight)
        st.markdown('</div>', unsafe_allow_html=True)

    # Show methodology explanation
    with st.expander("How are these insights generated?"):
        st.markdown("""
        These insights are generated using Google's Gemini AI model, which analyzes the time series data
        from Earth Engine for patterns, trends, and anomalies. The AI considers:

        1. **Statistical patterns** - Averages, trends, variances, and seasonality
        2. **Correlations** - Relationships between different water indicators
        3. **Known hydrology principles** - How these indicators relate to water resource management
        4. **Regional context** - The specific challenges of the Marathwada region

        The insights are regenerated each time you load new data, ensuring they're specific to your selected time period.
        These insights should be used to complement scientific analysis and local knowledge, not replace them.
        """)

with tabs[5]:
    st.markdown('<h2 class="sub-header">Data Download</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight">
    Download the data used in this analysis for further exploration or integration with other tools.
    </div>
    """, unsafe_allow_html=True)

    # Create data dictionary for downloads
    download_data = {
        'rainfall': data['rainfall_df'],
        'ndvi': data['ndvi_df'],
        'ndwi': data['ndwi_df'],
        'soil_moisture': data['soil_moisture_df'],
        'elevation': pd.DataFrame({'parameter': 'elevation', 'mean': data['elevation'].reduceRegion(reducer=ee.Reducer.mean(), geometry=data['region'], scale=1000, maxPixels=1e9).getInfo()['elevation']}, index=[0]),
        'dem': pd.DataFrame({'parameter': 'dem', 'mean': data['elevation'].reduceRegion(reducer=ee.Reducer.mean(), geometry=data['region'], scale=1000, maxPixels=1e9).getInfo()['elevation']}, index=[0]),
        'slope': pd.DataFrame({'parameter': 'slope', 'mean': data['slope'].reduceRegion(reducer=ee.Reducer.mean(), geometry=data['region'], scale=1000, maxPixels=1e9).getInfo()['slope']}, index=[0]),
        'lulc': pd.DataFrame({'parameter': 'lulc', 'mode': data['lulc'].reduceRegion(reducer=ee.Reducer.mode(), geometry=data['region'], scale=1000, maxPixels=1e9).getInfo()['Map']}, index=[0]),
        'soil_texture': pd.DataFrame({'parameter': 'soil_texture', 'mode': data['soil'].reduceRegion(reducer=ee.Reducer.mode(), geometry=data['region'], scale=1000, maxPixels=1e9).getInfo()['b0']}, index=[0]),
        'geomorphology': pd.DataFrame({'parameter': 'geomorphology', 'mode': data['geom'].reduceRegion(reducer=ee.Reducer.mode(), geometry=data['region'], scale=1000, maxPixels=1e9).getInfo()['constant']}, index=[0]),
        'runoff_coefficient': pd.DataFrame({'parameter': 'runoff_coefficient', 'mean': data['runoff_coeff'].reduceRegion(reducer=ee.Reducer.mean(), geometry=data['region'], scale=1000, maxPixels=1e9).getInfo()['runoff_coeff']}, index=[0])
    }

    st.markdown("### Individual Datasets")

    # Row 1: Rainfall and Soil Moisture
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.markdown("#### Rainfall Data")
        if not data['rainfall_df'].empty:
            st.dataframe(data['rainfall_df'].head(10), use_container_width=True)
            st.markdown(get_table_download_link(data['rainfall_df'], "rainfall_data", "Download Rainfall Data"), unsafe_allow_html=True)
        else:
            st.info("No Rainfall data available for the selected period.")
    with row1_col2:
        st.markdown("#### Soil Moisture Data")
        if not data['soil_moisture_df'].empty:
            st.dataframe(data['soil_moisture_df'].head(10), use_container_width=True)
            st.markdown(get_table_download_link(data['soil_moisture_df'], "soil_moisture_data", "Download Soil Moisture Data"), unsafe_allow_html=True)
        else:
            st.info("No Soil Moisture data available for the selected period.")

    # Row 2: NDVI and NDWI
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.markdown("#### NDVI Data")
        if not data['ndvi_df'].empty:
            st.dataframe(data['ndvi_df'].head(10), use_container_width=True)
            st.markdown(get_table_download_link(data['ndvi_df'], "ndvi_data", "Download NDVI Data"), unsafe_allow_html=True)
        else:
            st.info("No NDVI data available for the selected period.")
    with row2_col2:
        st.markdown("#### NDWI Data")
        if not data['ndwi_df'].empty:
            st.dataframe(data['ndwi_df'].head(10), use_container_width=True)
            st.markdown(get_table_download_link(data['ndwi_df'], "ndwi_data", "Download NDWI Data"), unsafe_allow_html=True)
        else:
            st.info("No NDWI data available for the selected period.")

    # Row 3: Elevation and DEM
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        st.markdown("#### Elevation Data")
        elev_val = data['elevation'].reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=data['region'],
            scale=1000,
            maxPixels=1e9
        ).getInfo().get('elevation', None)
        if elev_val is not None:
            st.markdown(f"<div class='card'><b>Elevation (mean):</b> {elev_val:.3f}</div>", unsafe_allow_html=True)
            elev_df = pd.DataFrame({'parameter': ['elevation'], 'value': [elev_val]})
            st.markdown(get_table_download_link(elev_df, "elevation_data", "Download Elevation Data"), unsafe_allow_html=True)
        else:
            st.info("No Elevation data available for the selected period.")
    with row3_col2:
        st.markdown("#### DEM Data")
        dem_val = data['elevation'].reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=data['region'],
            scale=1000,
            maxPixels=1e9
        ).getInfo().get('elevation', None)
        if dem_val is not None:
            st.markdown(f"<div class='card'><b>DEM (mean):</b> {dem_val:.3f}</div>", unsafe_allow_html=True)
            dem_df = pd.DataFrame({'parameter': ['dem'], 'value': [dem_val]})
            st.markdown(get_table_download_link(dem_df, "dem_data", "Download DEM Data"), unsafe_allow_html=True)
        else:
            st.info("No DEM data available for the selected period.")

    # Row 4: Slope and LULC
    row4_col1, row4_col2 = st.columns(2)
    with row4_col1:
        st.markdown("#### Slope Data")
        slope_val = data['slope'].reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=data['region'],
            scale=1000,
            maxPixels=1e9
        ).getInfo().get('slope', None)
        if slope_val is not None:
            st.markdown(f"<div class='card'><b>Slope (mean):</b> {slope_val:.3f}</div>", unsafe_allow_html=True)
            slope_df = pd.DataFrame({'parameter': ['slope'], 'value': [slope_val]})
            st.markdown(get_table_download_link(slope_df, "slope_data", "Download Slope Data"), unsafe_allow_html=True)
        else:
            st.info("No Slope data available for the selected period.")
    with row4_col2:
        st.markdown("#### LULC Data")
        lulc_val = data['lulc'].reduceRegion(
            reducer=ee.Reducer.mode(),
            geometry=data['region'],
            scale=1000,
            maxPixels=1e9
        ).getInfo().get('Map', None)
        if lulc_val is not None:
            st.markdown(f"<div class='card'><b>LULC (mode):</b> {lulc_val}</div>", unsafe_allow_html=True)
            lulc_df = pd.DataFrame({'parameter': ['lulc'], 'value': [lulc_val]})
            st.markdown(get_table_download_link(lulc_df, "lulc_data", "Download LULC Data"), unsafe_allow_html=True)
        else:
            st.info("No LULC data available for the selected period.")

    # Row 5: Soil Texture and Geomorphology
    row5_col1, row5_col2 = st.columns(2)
    with row5_col1:
        st.markdown("#### Soil Texture Data")
        soil_val = data['soil'].reduceRegion(
            reducer=ee.Reducer.mode(),
            geometry=data['region'],
            scale=1000,
            maxPixels=1e9
        ).getInfo().get('b0', None)
        if soil_val is not None:
            st.markdown(f"<div class='card'><b>Soil Texture (mode):</b> {soil_val}</div>", unsafe_allow_html=True)
            soil_df = pd.DataFrame({'parameter': ['soil_texture'], 'value': [soil_val]})
            st.markdown(get_table_download_link(soil_df, "soil_texture_data", "Download Soil Texture Data"), unsafe_allow_html=True)
        else:
            st.info("No Soil Texture data available for the selected period.")
    with row5_col2:
        st.markdown("#### Geomorphology Data")
        geom_val = data['geom'].reduceRegion(
            reducer=ee.Reducer.mode(),
            geometry=data['region'],
            scale=1000,
            maxPixels=1e9
        ).getInfo().get('constant', None)
        if geom_val is not None:
            st.markdown(f"<div class='card'><b>Geomorphology (mode):</b> {geom_val}</div>", unsafe_allow_html=True)
            geom_df = pd.DataFrame({'parameter': ['geomorphology'], 'value': [geom_val]})
            st.markdown(get_table_download_link(geom_df, "geomorphology_data", "Download Geomorphology Data"), unsafe_allow_html=True)
        else:
            st.info("No Geomorphology data available for the selected period.")

    # Row 6: Runoff Coefficient (single block)
    st.markdown("#### Runoff Coefficient Data")
    runoff_val = data['runoff_coeff'].reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=data['region'],
        scale=1000,
        maxPixels=1e9
    ).getInfo().get('runoff_coeff', None)
    if runoff_val is not None:
        st.markdown(f"<div class='card'><b>Runoff Coefficient (mean):</b> {runoff_val:.3f}</div>", unsafe_allow_html=True)
        runoff_df = pd.DataFrame({'parameter': ['runoff_coefficient'], 'value': [runoff_val]})
        st.markdown(get_table_download_link(runoff_df, "runoff_coefficient_data", "Download Runoff Coefficient Data"), unsafe_allow_html=True)
    else:
        st.info("No Runoff Coefficient data available for the selected period.")

    # Download all datasets
    st.markdown("### Download All Datasets")
    st.markdown(get_all_data_download_link(download_data), unsafe_allow_html=True)

    # Metadata information
    st.markdown("### Dataset Metadata")
    st.markdown("""
    <div class="card">
    <h4>Data Sources:</h4>
    <ul>
        <li><strong>Rainfall:</strong> CHIRPS Daily Precipitation (mm) - 0.05° resolution</li>
        <li><strong>NDVI:</strong> Normalized Difference Vegetation Index from Sentinel-2 (Range: -1 to 1)</li>
        <li><strong>NDWI:</strong> Normalized Difference Water Index from Sentinel-2 (Range: -1 to 1)</li>
        <li><strong>Soil Moisture:</strong> NASA SMAP Enhanced L3 Radiometer Global Daily 9 km (cm³/cm³)</li>
        <li><strong>Elevation:</strong> SRTM Digital Elevation Model (meters)</li>
    </ul>

    <h4>Analysis Period:</h4>
    <p>From {start_date} to {end_date}</p>

    <h4>Spatial Extent:</h4>
    <p>Marathwada region of Maharashtra, India (8 districts: Aurangabad, Bid, Jalna, Latur, Nanded, Osmanabad, Parbhani, Hingoli)</p>

    <h4>Processing Notes:</h4>
    <ul>
        <li>All datasets are spatially averaged over the region for time series analysis</li>
        <li>Missing values may occur due to cloud cover or satellite data gaps</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
Marathwada Water Resource Analysis Dashboard | Created with Earth Engine & Streamlit | Last updated: April 2025
</div>
""", unsafe_allow_html=True)
