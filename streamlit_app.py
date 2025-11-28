import json
import os
from datetime import datetime, timezone, timedelta, date

import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="IND320 Project", layout="wide")
import pandas as pd
import numpy as np
import requests

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import streamlit-plotly-events for click detection
try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except ImportError:
    PLOTLY_EVENTS_AVAILABLE = False

# extra imports required for analysis and plotting
from pathlib import Path
from scipy.fftpack import dct, idct
from scipy.signal import spectrogram as _scipy_spectrogram
from statsmodels.tsa.seasonal import STL
from sklearn.neighbors import LocalOutlierFactor

# --- Mongo helpers (kept) ---
# Helper function to retrieve MongoDB settings from secrets or environment variables
def _mongo_settings():
    # prefer Streamlit secrets, fall back to env vars for local testing
    uri = st.secrets.get("mongo", {}).get("uri") if hasattr(st, "secrets") else None
    db = st.secrets.get("mongo", {}).get("db") if hasattr(st, "secrets") else None
    coll = st.secrets.get("mongo", {}).get("coll") if hasattr(st, "secrets") else None
    uri = uri or os.getenv("MONGODB_URI", "")
    db = db or os.getenv("MONGODB_DB", "power")
    coll = coll or os.getenv("MONGODB_COLL", "production_2021")
    return uri, db, coll

# Cache the MongoDB client for efficient reuse
@st.cache_resource
def _mongo_client(uri: str):
    from pymongo import MongoClient
    # Add SSL options to handle certificate verification issues on macOS
    return MongoClient(
        uri,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=10000
    )

# Retrieve distinct price areas from the MongoDB collection
@st.cache_data(ttl=300)
def get_price_areas(uri: str, db: str, coll: str):
    cli = _mongo_client(uri)
    return sorted(cli[db][coll].distinct("priceArea"))

# Retrieve production groups for a specific price area
@st.cache_data(ttl=300)
def get_groups_for_area(uri: str, db: str, coll: str, price_area: str):
    cli = _mongo_client(uri)
    # Try 'group' field first (new schema), fallback to 'productionGroup' (old schema)
    groups = sorted(cli[db][coll].distinct("group", {"priceArea": price_area}))
    if not groups:
        groups = sorted(cli[db][coll].distinct("productionGroup", {"priceArea": price_area}))
    return groups

# Retrieve groups for production or consumption
@st.cache_data(ttl=300)
def get_groups_for_area_by_type(uri: str, db: str, coll: str, price_area: str, data_type: str = "production"):
    cli = _mongo_client(uri)
    field = "productionGroup" if data_type == "production" else "consumptionGroup"
    try:
        groups = sorted(cli[db][coll].distinct(field, {"priceArea": price_area}))
        return groups if groups else []
    except:
        return []


@st.cache_data(ttl=300)
def get_mean_by_price_area(uri: str, db: str, coll: str, production_group: str, start_dt, end_dt) -> pd.DataFrame:
    """Aggregate mean quantityKwh per priceArea for given group and time range."""
    if not uri:
        return pd.DataFrame(columns=["priceArea", "meanKwh"])
    cli = _mongo_client(uri)
    pipe = [
        {"$addFields": {"ts": {"$cond": [{"$eq": [{"$type": "$startTime"}, "string"]}, {"$dateFromString": {"dateString": "$startTime"}}, "$startTime"]}}},
        {"$match": {"ts": {"$gte": start_dt, "$lt": end_dt}, "productionGroup": production_group}},
        {"$addFields": {"qty": {"$cond": [{"$in": [{"$type": "$quantityKwh"}, ["int","long","double","decimal"]]}, "$quantityKwh", {"$toDouble": "$quantityKwh"}]}}},
        {"$group": {"_id": "$priceArea", "meanKwh": {"$avg": "$qty"}}},
        {"$sort": {"_id": 1}}
    ]
    rows = list(cli[db][coll].aggregate(pipe, allowDiskUse=True))
    if not rows:
        return pd.DataFrame(columns=["priceArea", "meanKwh"])
    df = pd.DataFrame(rows).rename(columns={"_id": "priceArea"})
    df["meanKwh"] = pd.to_numeric(df["meanKwh"], errors="coerce").fillna(0.0)
    return df

# Retrieve yearly totals for a specific price area
@st.cache_data(ttl=300)
def get_year_totals(uri: str, db: str, coll: str, price_area: str, year: int = 2021) -> pd.DataFrame:
    cli = _mongo_client(uri)

    dt_start = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    dt_end   = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    pipe = [
        # area first
        {"$match": {"priceArea": price_area}},
        # make ts a Date no matter if startTime is string or already Date
        {"$addFields": {
            "ts": {
                "$cond": [
                    {"$eq": [{"$type": "$startTime"}, "string"]},
                    {"$dateFromString": {"dateString": "$startTime"}},
                    "$startTime"
                ]
            }
        }},
        # filter by year on ts
        {"$match": {"ts": {"$gte": dt_start, "$lt": dt_end}}},
        # coerce quantityKwh to double robustly
        {"$addFields": {
            "qty": {
                "$cond": [
                    {"$in": [{"$type": "$quantityKwh"}, ["int", "long", "double", "decimal"]]},
                    "$quantityKwh",
                    {"$toDouble": "$quantityKwh"}
                ]
            },
            "grp": {"$ifNull": ["$group", "$productionGroup"]}
        }},
        # group
        {"$group": {"_id": "$grp", "totalKwh": {"$sum": "$qty"}}},
        {"$sort": {"totalKwh": -1}}
    ]

    rows = list(cli[db][coll].aggregate(pipe, allowDiskUse=True))
    if not rows:
        return pd.DataFrame(columns=["productionGroup", "totalKwh"])

    df = pd.DataFrame(rows).rename(columns={"_id": "productionGroup"})
    df["totalKwh"] = pd.to_numeric(df["totalKwh"], errors="coerce").fillna(0.0)
    return df[df["totalKwh"] > 0]

# Retrieve monthly series data for specific groups in a price area
@st.cache_data(ttl=300)
def get_month_series(uri: str, db: str, coll: str, price_area: str, year: int, month: int, groups: list[str]) -> pd.DataFrame:
    cli = _mongo_client(uri)

    # build UTC bounds for the month
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC").to_pydatetime()
    end = (pd.Timestamp(year=year, month=month, day=1, tz="UTC") + pd.offsets.MonthBegin(1)).to_pydatetime()

    match = {"priceArea": price_area}
    if groups:
        # Match either 'group' or 'productionGroup' field
        match["$or"] = [{"group": {"$in": groups}}, {"productionGroup": {"$in": groups}}]

    pipe = [
        {"$match": match},
        {"$addFields": {
            "ts": {
                "$cond": [
                    {"$eq": [{"$type": "$startTime"}, "string"]},
                    {"$dateFromString": {"dateString": "$startTime"}},
                    "$startTime"
                ]
            },
            "grp": {"$ifNull": ["$group", "$productionGroup"]}
        }},
        {"$match": {"ts": {"$gte": start, "$lt": end}}},
        {"$addFields": {
            "qty": {
                "$cond": [
                    {"$in": [{"$type": "$quantityKwh"}, ["int", "long", "double", "decimal"]]},
                    "$quantityKwh",
                    {"$toDouble": "$quantityKwh"}
                ]
            }
        }},
        {"$project": {"_id": 0, "productionGroup": "$grp", "startTime": "$ts", "quantityKwh": "$qty"}}
    ]

    rows = list(cli[db][coll].aggregate(pipe, allowDiskUse=True))
    if not rows:
        return pd.DataFrame(columns=["startTime", "productionGroup", "quantityKwh"])

    df = pd.DataFrame(rows)
    # ts is already Date, ensure tz aware and sorted
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df["quantityKwh"] = pd.to_numeric(df["quantityKwh"], errors="coerce").fillna(0.0)
    return df.sort_values("startTime")

# --- Area map for ERA5 download ---
AREAS = [
    {"price_area": "NO1", "city": "Oslo",         "lat": 59.9139,  "lon": 10.7522},
    {"price_area": "NO2", "city": "Kristiansand", "lat": 58.1467,  "lon": 7.9956},
    {"price_area": "NO3", "city": "Trondheim",    "lat": 63.4305,  "lon": 10.3951},
    {"price_area": "NO4", "city": "Tromsø",       "lat": 69.6492,  "lon": 18.9553},
    {"price_area": "NO5", "city": "Bergen",       "lat": 60.39299, "lon": 5.32415},
]
AREA_DF = pd.DataFrame(AREAS)

def _coords_for_area(area_code: str):
    row = AREA_DF.loc[AREA_DF["price_area"] == area_code].iloc[0]
    return float(row["lon"]), float(row["lat"]), str(row["city"])

# --- ERA5 hourly download for 2021 via Open Meteo archive API ---
@st.cache_data(ttl=3600)
def download_era5_hourly_for_area(area_code: str, year: int = 2021,
                                  hourly_vars=None, tz: str = "Europe/Oslo") -> pd.DataFrame:
    default_vars = ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]
    if hourly_vars is None:
        hourly_vars = default_vars
    lon, lat, _ = _coords_for_area(area_code)
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date":   f"{year}-12-31",
        "hourly":     ",".join(hourly_vars),
        "timezone":   tz
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    time = pd.to_datetime(data["hourly"]["time"])
    df = pd.DataFrame(index=time)
    for v in hourly_vars:
        df[v] = data["hourly"].get(v, [np.nan] * len(time))
    df.index.name = "time"
    return df.sort_index()

# --- ERA5 download for arbitrary coordinates ---
@st.cache_data(ttl=3600)
def download_era5_hourly_coords(lat: float, lon: float, year_start: int, year_end: int,
                                hourly_vars=None, tz: str = "Europe/Oslo") -> pd.DataFrame:
    """Download ERA5 data for arbitrary coordinates and year range"""
    default_vars = ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]
    if hourly_vars is None:
        hourly_vars = default_vars
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": f"{year_start}-01-01",
        "end_date":   f"{year_end}-12-31",
        "hourly":     ",".join(hourly_vars),
        "timezone":   tz
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    time = pd.to_datetime(data["hourly"]["time"])
    df = pd.DataFrame(index=time)
    for v in hourly_vars:
        df[v] = data["hourly"].get(v, [np.nan] * len(time))
    df.index.name = "time"
    return df.sort_index()

# --- Shared selection state for area ---
if "area_code" not in st.session_state:
    st.session_state.area_code = "NO5"
if "selected_coord" not in st.session_state:
    st.session_state.selected_coord = None
if "selected_area" not in st.session_state:
    st.session_state.selected_area = None
if "clicked_coord" not in st.session_state:
    st.session_state.clicked_coord = None

# --- Sidebar Navigation ---
st.sidebar.title("🗺️ IND320 Energy Analytics")

# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Home button
if st.sidebar.button("🏠 Home", use_container_width=True, type="primary" if st.session_state.page == "Home" else "secondary"):
    st.session_state.page = "Home"
    st.rerun()

st.sidebar.markdown("---")

# Page navigation with buttons organized by section
st.sidebar.markdown("### 📊 Explorative Analysis")
if st.sidebar.button("🗺️ Map & Regions", use_container_width=True, type="primary" if st.session_state.page == "Map & Regions" else "secondary"):
    st.session_state.page = "Map & Regions"
    st.rerun()
if st.sidebar.button("⚡ Elhub Production", use_container_width=True, type="primary" if st.session_state.page == "Elhub Production" else "secondary"):
    st.session_state.page = "Elhub Production"
    st.rerun()
if st.sidebar.button("📊 Data Overview", use_container_width=True, type="primary" if st.session_state.page == "Data Overview" else "secondary"):
    st.session_state.page = "Data Overview"
    st.rerun()
if st.sidebar.button("❄️ Snow Drift", use_container_width=True, type="primary" if st.session_state.page == "Snow Drift" else "secondary"):
    st.session_state.page = "Snow Drift"
    st.rerun()

st.sidebar.markdown("### 🔍 Signal Analysis")
if st.sidebar.button("📈 STL & Spectrogram", use_container_width=True, type="primary" if st.session_state.page == "STL & Spectrogram" else "secondary"):
    st.session_state.page = "STL & Spectrogram"
    st.rerun()
if st.sidebar.button("🔄 Sliding Window Correlation", use_container_width=True, type="primary" if st.session_state.page == "Sliding Window Correlation" else "secondary"):
    st.session_state.page = "Sliding Window Correlation"
    st.rerun()

st.sidebar.markdown("### ⚠️ Anomaly Detection")
if st.sidebar.button("🚨 Outliers & Anomalies", use_container_width=True, type="primary" if st.session_state.page == "Outliers & Anomalies" else "secondary"):
    st.session_state.page = "Outliers & Anomalies"
    st.rerun()

st.sidebar.markdown("### 🔮 Predictive Analysis")
if st.sidebar.button("📉 SARIMAX Forecasting", use_container_width=True, type="primary" if st.session_state.page == "SARIMAX Forecasting" else "secondary"):
    st.session_state.page = "SARIMAX Forecasting"
    st.rerun()

page = st.session_state.page

# --- Page 1: Home ---
if page == "Home":
    st.title("🗺️ IND320 Project - Weather and Energy Analytics")
    st.markdown("""
    ## Welcome to the Energy Analytics Dashboard
    
    This application provides comprehensive analysis of Norwegian energy production data combined with meteorological insights.
    
    ### 📊 Features:
    
    **Explorative Analysis:**
    - **Interactive clickable map** with GeoJSON price area polygons (NO1-NO5)
    - Click on regions or markers to select price areas
    - Production and consumption data analysis
    - Choropleth coloring by mean production values
    - Snow drift calculations for selected coordinates with wind rose visualization
    
    **Signal Analysis:**
    - STL decomposition (trend, seasonal, residual components)
    - Spectrograms for frequency content over time
    - Sliding window correlation with lag analysis
    
    **Anomaly Detection:**
    - Statistical Process Control (SPC) with DCT high-pass filtering
    - Local Outlier Factor (LOF) density-based detection
    
    **Predictive Analysis:**
    - SARIMAX forecasting with seasonal components for future predictions
    
    ### 🚀 Getting Started:
    1. **Click on the map** in **Map & Regions** to select a price area
    2. Explore production/consumption data in **Elhub Production**
    3. Analyze signals with **STL & Spectrogram**
    4. Detect anomalies in **Outliers & Anomalies**
    5. Perform correlation analysis in **Sliding Window Correlation**
    6. Forecast with **SARIMAX Forecasting**
    7. Calculate snow drift in **Snow Drift**
    
    Use the sidebar to navigate between different analysis tools.
    """)

# --- Page: Map & Regions ---
elif page == "Map & Regions":
    import folium
    from streamlit_folium import st_folium
    from shapely.geometry import shape, Point
    
    st.title("🗺️ Map - Price Areas with Interactive Selection")
    st.caption("📋 Click anywhere on the map to store coordinates")

    uri, dbname, collname = _mongo_settings()
    
    # Load GeoJSON from workspace
    geojson_path = Path("file.geojson")
    geojson_data = None
    if geojson_path.exists():
        try:
            with open(geojson_path) as f:
                geojson_data = json.load(f)
        except Exception as e:
            st.error(f"Failed to load GeoJSON: {e}")
    else:
        st.error("file.geojson not found in workspace. Please ensure GeoJSON file is present.")
    
    # Build ID to name mapping
    @st.cache_data
    def build_id_to_name(gj):
        out = {}
        for f in gj.get("features", []):
            fid = f.get("id") or (f.get("properties") or {}).get("id")
            if fid is None:
                continue
            name = (f.get("properties") or {}).get("ElSpotOmr")
            if name:
                out[fid] = str(name)
        return out
    
    # Price area name to GeoJSON ID mapping
    AREA_ID_MAP = {
        "NO1": 6,
        "NO2": 7,
        "NO3": 8,
        "NO4": 9,
        "NO5": 10
    }
    
    id_to_name = build_id_to_name(geojson_data) if geojson_data else {}
    
    # Build polygons for coordinate lookup
    if "polygons" not in st.session_state and geojson_data:
        polys = []
        for feat in geojson_data.get("features", []):
            fid = feat.get("id") or (feat.get("properties") or {}).get("id")
            if not fid:
                continue
            try:
                geom = shape(feat["geometry"])
            except Exception:
                continue
            polys.append((fid, geom))
        st.session_state.polygons = polys
    
    def find_feature_id(lon: float, lat: float):
        """Find which GeoJSON feature contains this coordinate."""
        if "polygons" not in st.session_state:
            return None
        pt = Point(lon, lat)
        for fid, geom in st.session_state.polygons:
            if geom.covers(pt):
                return fid
        return None
    
    # Map controls
    col1, col2 = st.columns(2)
    
    with col1:
        data_type = st.radio(
            "Dataset",
            ["Production", "Consumption"],
            horizontal=True
        )
    
    with col2:
        # Get available groups based on data type
        if uri:
            from pymongo import MongoClient
            client = MongoClient(uri, tls=True, tlsAllowInvalidCertificates=True)
            db = client[dbname]
            
            if data_type == "Production":
                # Production collection - uses field "group"
                active_collection = "production_2021_2024"
                groups = sorted(db[active_collection].distinct("group"))
                col_group = "group"
                col_time = "startTime"
                col_area = "priceArea"
                col_kwh = "quantityKwh"
            else:
                # Consumption collection - also uses field "group"
                active_collection = "consumption_2021_2024"
                groups = sorted(db[active_collection].distinct("group"))
                col_group = "group"
                col_time = "startTime"
                col_area = "priceArea"
                col_kwh = "quantityKwh"
        else:
            groups = []
            active_collection = collname
        
        if groups:
            group_select = st.selectbox(
                f"{data_type} Group",
                options=groups,
                help=f"Select a {data_type.lower()} group to visualize"
            )
        else:
            st.warning(f"⚠️ No {data_type.lower()} groups found in database")
            group_select = None
    
    # Date selection - query actual min/max from database
    col3, col4 = st.columns(2)
    
    # Get actual date range from database
    @st.cache_data(ttl=3600)
    def get_date_range(uri, dbname, collection_name):
        from pymongo import MongoClient
        from datetime import date
        client = MongoClient(uri, tls=True, tlsAllowInvalidCertificates=True)
        db = client[dbname]
        coll = db[collection_name]
        
        # Get min and max startTime
        min_doc = list(coll.find().sort("startTime", 1).limit(1))
        max_doc = list(coll.find().sort("startTime", -1).limit(1))
        
        if min_doc and max_doc:
            min_dt = pd.to_datetime(min_doc[0]["startTime"])
            max_dt = pd.to_datetime(max_doc[0]["startTime"])
            return date(min_dt.year, min_dt.month, min_dt.day), date(max_dt.year, max_dt.month, max_dt.day)
        return date(2021, 1, 1), date(2024, 12, 31)
    
    if uri and active_collection:
        MIN_DATE, MAX_DATE = get_date_range(uri, dbname, active_collection)
    else:
        MIN_DATE, MAX_DATE = date(2021, 1, 1), date(2024, 12, 31)
    
    with col3:
        start_date = st.date_input(
            "Start date",
            value=MIN_DATE,
            min_value=MIN_DATE,
            max_value=MAX_DATE
        )
    
    with col4:
        end_date = st.date_input(
            "End date",
            value=MAX_DATE,
            min_value=MIN_DATE,
            max_value=MAX_DATE
        )
    
    if start_date > end_date:
        st.error("❌ Start date must be before end date.")
        st.stop()
    
    # Query MongoDB for choropleth data
    @st.cache_data
    def query_data(data_type, group, start, end, col_group, col_time, col_area, col_kwh, _uri, _dbname, _collname):
        from pymongo import MongoClient
        from datetime import datetime
        
        client = MongoClient(_uri, tls=True, tlsAllowInvalidCertificates=True)
        db = client[_dbname]
        col = db[_collname]
        
        pipeline = [
            {"$match": {
                col_group: group,
                col_time: {
                    "$gte": datetime.combine(start, datetime.min.time()),
                    "$lte": datetime.combine(end, datetime.max.time())
                }
            }},
            {"$group": {
                "_id": f"${col_area}",
                "mean_value": {"$avg": f"${col_kwh}"}
            }}
        ]
        
        df = pd.DataFrame(list(col.aggregate(pipeline)))
        if df.empty:
            return pd.DataFrame({"id": [], "value": []})
        
        # Convert NO1 -> 6 etc.
        df["id"] = df["_id"].map(AREA_ID_MAP)
        df["value"] = df["mean_value"]
        df = df.dropna(subset=["id"])
        
        return df[["id", "value"]]
    
    df_vals = pd.DataFrame({"id": [], "value": []})
    if group_select and uri:
        with st.spinner(f"Loading {data_type.lower()} data..."):
            try:
                df_vals = query_data(
                    data_type,
                    group_select,
                    start_date,
                    end_date,
                    col_group,
                    col_time,
                    col_area,
                    col_kwh,
                    uri,
                    dbname,
                    active_collection
                )
            except Exception as e:
                st.warning(f"Could not load data: {e}")
    
    # Initialize session state for pin and selected feature
    if "last_pin" not in st.session_state:
        st.session_state.last_pin = [63.0, 10.0]  # Center of Norway
    if "selected_feature_id" not in st.session_state:
        st.session_state.selected_feature_id = None
    
    # Auto-detect feature from initial pin
    if st.session_state.selected_feature_id is None and geojson_data:
        lat, lon = st.session_state.last_pin
        st.session_state.selected_feature_id = find_feature_id(lon, lat)
    
    # Display selection info above map
    fid = st.session_state.selected_feature_id
    if fid is not None:
        area_name = id_to_name.get(fid, f"ID {fid}")
        value = df_vals.loc[df_vals["id"] == fid, "value"] if not df_vals.empty else []
        value_display = f"{float(value.iloc[0]):,.0f}" if len(value) else "No data"
        
        st.info(f"📍 **{area_name}** | Coordinates: {st.session_state.last_pin[0]:.4f}°N, {st.session_state.last_pin[1]:.4f}°E | Mean kWh: {value_display}")
    else:
        st.info(f"📍 Click on map to select area | Current: {st.session_state.last_pin[0]:.4f}°N, {st.session_state.last_pin[1]:.4f}°E")
    
    # Create folium map
    m = folium.Map(
        location=st.session_state.last_pin,
        zoom_start=5,
        tiles="OpenStreetMap"
    )
    
    # Choropleth layer
    if geojson_data and not df_vals.empty:
        folium.Choropleth(
            geo_data=geojson_data,
            data=df_vals,
            columns=["id", "value"],
            key_on="feature.id",
            fill_color="YlOrRd",
            fill_opacity=0.5,
            line_opacity=0.9,
            line_color="black",
            line_weight=1,
            nan_fill_opacity=0.2,
            legend_name=f"Mean {data_type} ({group_select})",
            highlight=True
        ).add_to(m)
    
    # Highlight selected polygon with thicker border
    sel_id = st.session_state.selected_feature_id
    if sel_id is not None and geojson_data:
        selected_feats = [
            f for f in geojson_data.get("features", [])
            if f.get("id") == sel_id
        ]
        if selected_feats:
            folium.GeoJson(
                {"type": "FeatureCollection", "features": selected_feats},
                style_function=lambda f: {
                    "fillOpacity": 0,
                    "color": "#FF4444",
                    "weight": 5
                }
            ).add_to(m)
    
    # Pin marker at clicked location
    folium.Marker(
        location=st.session_state.last_pin,
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
    # Render map with click detection
    out = st_folium(m, key="folium_map", height=650, width=None)
    
    # Handle map clicks
    if out and out.get("last_clicked"):
        lat = out["last_clicked"]["lat"]
        lon = out["last_clicked"]["lng"]
        new_coord = [lat, lon]
        
        if new_coord != st.session_state.last_pin:
            st.session_state.last_pin = new_coord
            st.session_state.selected_feature_id = find_feature_id(lon, lat)
            
            # Update area selection if within a price area
            if st.session_state.selected_feature_id:
                # Map feature ID back to area code
                reverse_map = {v: k for k, v in AREA_ID_MAP.items()}
                area_code = reverse_map.get(st.session_state.selected_feature_id)
                if area_code:
                    st.session_state.area_code = area_code
                    st.session_state.selected_area = area_code
                    area_row = AREA_DF[AREA_DF["price_area"] == area_code].iloc[0]
                    st.session_state.selected_coord = (float(area_row["lat"]), float(area_row["lon"]))
            
            st.rerun()

    with st.expander("📖 Map Guide"):
        st.markdown("""
        **How to use:**
        - Click anywhere on the map to select that location and price area
        - The info bar above shows your selection details and mean energy values
        - Red marker indicates your selected point
        - Selected price area is highlighted with a red border
        
        **What you see:**
        - Color intensity represents mean energy values for the selected group and date range
        - Darker colors = higher values
        - Price areas: NO1 (Oslo), NO2 (Kristiansand), NO3 (Trondheim), NO4 (Tromsø), NO5 (Bergen)
        
        **Data:** GeoJSON polygons from NVE | Energy data from Elhub via MongoDB
        """)

# --- Page: Elhub Production ---
elif page == "Elhub Production":
    st.title("⚡ Elhub - Energy Production Analysis")

    uri, dbname, _ = _mongo_settings()
    collname = "production_2021_2024"  # Use the correct collection with 2021-2024 data
    if not uri:
        st.error("MongoDB URI is missing. Add it to st.secrets['mongo']['uri'] or set MONGODB_URI.")
        st.stop()
    
    # Year selector
    selected_year = st.selectbox("Year", options=[2021, 2022, 2023, 2024], index=0, key="elhub_year")
    
    # Area selector with radio buttons
    areas = get_price_areas(uri, dbname, collname)
    if not areas:
        st.warning("No price areas found.")
        st.stop()
    
    # Use radio buttons - let it maintain its own state via key
    selected_area_elhub = st.radio(
        "Price Area",
        options=areas,
        horizontal=True,
        key="elhub_area_radio"
    )
    
    # Update global session state to keep it in sync
    st.session_state.area_code = selected_area_elhub
    area = selected_area_elhub

    # split into two columns
    left, right = st.columns(2)

    # left side - pie chart for selected year and area
    with left:
        st.subheader("Totals pie")

        with st.spinner(f"Loading production data for {area} ({selected_year})..."):
            pie_df = get_year_totals(uri, dbname, collname, area, selected_year)
        if pie_df.empty:
            st.info(f"No data for {area} in {selected_year}.")
        else:
            total_sum = pie_df["totalKwh"].sum()
            pie_df["pct"] = (100 * pie_df["totalKwh"] / total_sum).round(1)
            fig_p = px.pie(pie_df, values="totalKwh", names="productionGroup",
                           hover_data={"totalKwh": True, "pct": True},
                           title=f"Production groups share in {area} ({selected_year})")
            fig_p.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_p, use_container_width=True)

    # right side - group selector, month selector, line plot
    with right:
        st.subheader("Monthly lines")
        all_groups = get_groups_for_area(uri, dbname, collname, area)

        pills_fn = getattr(st, "pills", None)
        if pills_fn is not None:
            selected_groups = pills_fn("Production groups", options=all_groups, selection_mode="multi")
        else:
            selected_groups = st.multiselect("Production groups", options=all_groups, default=all_groups)

        month_str = st.selectbox("Month", options=[f"{selected_year}-{m:02d}" for m in range(1, 13)], index=0)
        year = selected_year
        month = int(month_str.split("-")[1])

        mdf = get_month_series(uri, dbname, collname, area, year, month, selected_groups)
        if mdf.empty:
            st.info("No rows for this selection.")
        else:
            # aggregate to daily totals
            mdf["day"] = mdf["startTime"].dt.tz_convert("UTC").dt.tz_localize(None).dt.date
            daily = mdf.groupby(["day", "productionGroup"], as_index=False)["quantityKwh"].sum()
            daily["day"] = pd.to_datetime(daily["day"])
            fig_l = px.line(daily, x="day", y="quantityKwh", color="productionGroup",
                            title=f"Daily kWh in {area} for {month_str}", labels={"quantityKwh": "kWh per day"})
            fig_l.update_layout(legend_title_text='Production group')
            st.plotly_chart(fig_l, use_container_width=True)

    # documentation
    with st.expander("Data source"):
        st.markdown(
            "Source: Elhub Energy Data API, dataset PRODUCTION_PER_GROUP_MBA_HOUR, year 2021. "
            "Data was fetched in a notebook, stored to Cassandra, then inserted into MongoDB. "
            "This page reads from MongoDB."
        )

# --- Page new A: STL and Spectrogram on Elhub ---
elif page == "STL & Spectrogram":
    st.title("STL and Spectrogram on Elhub production")
    
    uri, dbname, _ = _mongo_settings()
    collname = "production_2021_2024"  # Use the correct collection with 2021-2024 data
    if not uri:
        st.error("MongoDB URI is missing. Add it to st.secrets['mongo']['uri'] or set MONGODB_URI.")
        st.stop()
    
    areas = get_price_areas(uri, dbname, collname)
    if not areas:
        st.warning("No price areas found.")
        st.stop()
    area_idx = areas.index(st.session_state.area_code) if st.session_state.area_code in areas else 0
    
    # Year and area selectors
    col1, col2 = st.columns(2)
    with col1:
        stl_year = st.selectbox("Year", options=[2021, 2022, 2023, 2024], index=0, key="stl_year")
    with col2:
        area = st.selectbox("Price Area", options=areas, index=area_idx, key="stl_area")
    
    with st.expander("📚 Theoretical Background", expanded=False):
        st.markdown("""
        ### STL Decomposition
        **STL (Seasonal and Trend decomposition using Loess)** separates a time series into three components:
        - **Trend**: Long-term progression of the series (low-frequency variation)
        - **Seasonal**: Repeating patterns at fixed intervals (daily, weekly cycles)
        - **Residual**: Remaining variation after removing trend and seasonal components
        
        **Applications**: Identify underlying patterns, detect anomalies in residuals, forecast future values.
        
        ### Spectrogram Analysis
        **Spectrogram** shows how the frequency content of a signal changes over time using Short-Time Fourier Transform (STFT).
        - **X-axis**: Time
        - **Y-axis**: Frequency (cycles per hour)
        - **Color**: Power/Energy at each time-frequency point
        
        **Applications**: Detect changing periodicities, identify transient events, analyze non-stationary signals.
        """)
    all_groups = get_groups_for_area(uri, dbname, collname, area)
    if not all_groups:
        st.warning("No production groups found for this area.")
        st.stop()

    tabs = st.tabs(["STL", "Spectrogram"])

    def _series_for_group(grp: str) -> pd.Series:
        from datetime import datetime, timezone
        cli = _mongo_client(uri)

        dt_start = datetime(stl_year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dt_end   = datetime(stl_year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        pipe = [
            # Match either 'group' or 'productionGroup' field
            {"$match": {"priceArea": area, "$or": [{"group": grp}, {"productionGroup": grp}]}},
            {"$addFields": {
                "ts": {
                    "$cond": [
                        {"$eq": [{"$type": "$startTime"}, "string"]},
                        {"$dateFromString": {"dateString": "$startTime"}},
                        "$startTime"
                    ]
                }
            }},
            {"$addFields": {
                "qty": {
                    "$cond": [
                        {"$in": [{"$type": "$quantityKwh"}, ["int", "long", "double", "decimal"]]},
                        "$quantityKwh",
                        {"$toDouble": "$quantityKwh"}
                    ]
                }
            }},
            {"$match": {"ts": {"$gte": dt_start, "$lt": dt_end}}},
            {"$project": {"_id": 0, "startTime": "$ts", "quantityKwh": "$qty"}},
            {"$sort": {"startTime": 1}}
        ]

        rows = list(cli[dbname][collname].aggregate(pipe, allowDiskUse=True))
        if not rows:
            return pd.Series(dtype=float)

        dd = pd.DataFrame(rows)
        dd["time"] = pd.to_datetime(dd["startTime"], utc=True).dt.tz_convert("Europe/Oslo")
        dd["quantityKwh"] = pd.to_numeric(dd["quantityKwh"], errors="coerce").fillna(0.0)
        dd = dd.sort_values("time")

        return pd.Series(dd["quantityKwh"].values, index=dd["time"])

    # STL tab
    with tabs[0]:

        with st.expander("Quick help", expanded=False):
            st.markdown(
                "- Cycle length is the repeating pattern in samples. For hourly data, 24 means daily cycle, 24*7 means weekly.\n"
                "- Seasonal window and Trend window are smoothing sizes. They must be odd numbers. We auto-adjust to the next odd value if needed.\n"
                "- Robust fit makes the method less sensitive to spikes."
            )

        grp = st.selectbox(
            "Choose production group",
            options=all_groups,
            index=0,
            key="stl_grp",
            help="Pick which Elhub production series to analyze"
        )

        # dynamic title, e.g. "STL for hydro in NO5"
        st.subheader(f"STL for {grp} in {area}")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            period = st.number_input(
                "Cycle length, samples",
                min_value=2,
                value=24,
                step=1,
                help="24 for daily pattern with hourly data, 168 for weekly"
            )
        with c2:
            seasonal = st.number_input(
                "Seasonal window",
                min_value=3,
                value=13,
                step=1,
                help="Must be odd, controls smoothness of the seasonal curve"
            )
        with c3:
            trend = st.number_input(
                "Trend window",
                min_value=5,
                value=365,
                step=1,
                help="Must be odd, controls smoothness of the long term trend"
            )
        with c4:
            robust = st.checkbox(
                "Robust fit",
                value=True,
                help="Downweights outliers during fitting"
            )

        # enforce odd windows
        seasonal_eff = int(seasonal) if int(seasonal) % 2 == 1 else int(seasonal) + 1
        trend_eff = int(trend) if int(trend) % 2 == 1 else int(trend) + 1

        series = _series_for_group(grp)
        if series.empty:
            st.info("No rows for this selection.")
        else:
            start, end = series.index.min(), series.index.max()
            idx = pd.date_range(start=start, end=end, freq="h", tz=series.index.tz)
            y = series.reindex(idx).interpolate(limit=6).bfill().ffill()

            # safety guard so windows are not larger than the series
            if len(y) < max(int(period), seasonal_eff, trend_eff) + 10:
                st.warning(
                    f"Series too short for cycle={period}, seasonal={seasonal_eff}, trend={trend_eff} "
                    f"(length {len(y)}). Try smaller windows."
                )
            else:
                @st.cache_data
                def compute_stl(series_data, per, seas, tr, rob):
                    return STL(series_data, period=per, seasonal=seas, trend=tr, robust=rob).fit()
                
                with st.spinner("Performing STL decomposition..."):
                    stl_fit = compute_stl(y, int(period), seasonal_eff, trend_eff, bool(robust))

                # Use Plotly for interactive STL component visualization
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=(f"Observed {area} {grp}", "Trend", "Seasonal", "Remainder"))
                fig.add_trace(go.Scatter(x=stl_fit.observed.index, y=stl_fit.observed.values, name="Observed"), row=1, col=1)
                fig.add_trace(go.Scatter(x=stl_fit.trend.index, y=stl_fit.trend.values, name="Trend", line=dict(color="orange")), row=2, col=1)
                fig.add_trace(go.Scatter(x=stl_fit.seasonal.index, y=stl_fit.seasonal.values, name="Seasonal", line=dict(color="green")), row=3, col=1)
                fig.add_trace(go.Scatter(x=stl_fit.resid.index, y=stl_fit.resid.values, name="Remainder", line=dict(color="red")), row=4, col=1)
                fig.update_layout(height=900, showlegend=False, title_text=f"STL decomposition for {grp} in {area}")
                st.plotly_chart(fig, use_container_width=True)

    # Spectrogram tab


    
    with tabs[1]:

        with st.expander("Quick help", expanded=False):
            st.markdown(
                "- Window length controls frequency detail. Larger window gives better frequency detail.\n"
                "- Overlap is how much neighbor windows share. Half of the window length is common."
            )

        grp = st.selectbox(
            "Choose production group",
            options=all_groups,
            index=0,
            key="spec_grp",
            help="Pick which Elhub production series to analyze"
        )

        # dynamic title, e.g. "Spectrogram for hydro in NO5"
        st.subheader(f"Spectrogram for {grp} in {area}")

        c1, c2 = st.columns(2)
        with c1:
            window_len = st.number_input(
                "Window length, samples",
                min_value=32,
                value=256,
                step=8,
                help="STFT window size in samples"
            )
        with c2:
            overlap = st.number_input(
                "Overlap, samples",
                min_value=0,
                value=128,
                step=8,
                help="Usually half of the window length"
            )

        series = _series_for_group(grp)
        if series.empty:
            st.info("No rows for this selection.")
        else:
            start, end = series.index.min(), series.index.max()
            idx = pd.date_range(start=start, end=end, freq="h", tz=series.index.tz)
            y = series.reindex(idx).interpolate(limit=6).bfill().ffill().values

            nperseg = int(window_len)
            noverlap = min(int(overlap), nperseg // 2)

            if nperseg <= 1:
                st.warning("Window length must be at least 2.")
            else:
                with st.spinner("Computing spectrogram..."):
                    @st.cache_data
                    def compute_spectrogram(sig, nperseg_val, noverlap_val):
                        return _scipy_spectrogram(sig, fs=1.0, nperseg=nperseg_val, noverlap=noverlap_val,
                                                scaling="density", mode="magnitude")
                    f, t, Sxx = compute_spectrogram(y, nperseg, noverlap)

                # Use Plotly to render spectrogram as an image heatmap
                # t is time bins (index into windows), f is frequency bins
                Sxx_log = np.log1p(Sxx)
                fig = go.Figure(data=go.Heatmap(z=Sxx_log, x=t, y=f, colorscale="Viridis", colorbar=dict(title="log(1+mag)")))
                fig.update_layout(title=f"Spectrogram for {grp} in {area}", xaxis_title="Time (window index)", yaxis_title="Frequency (cycles/hour)", height=500)
                st.plotly_chart(fig, use_container_width=True)

# --- Page: Sliding Window Correlation ---
elif page == "Sliding Window Correlation":
    st.title("🔍 Sliding Window Correlation - Weather vs Energy")
    
    uri, dbname, _ = _mongo_settings()
    collname = "production_2021_2024"  # Use the correct collection with 2021-2024 data
    if not uri:
        st.error("MongoDB URI is missing for energy data.")
        st.stop()
    
    areas = get_price_areas(uri, dbname, collname)
    if not areas:
        st.warning("No price areas found.")
        st.stop()
    area_idx = areas.index(st.session_state.area_code) if st.session_state.area_code in areas else 0
    
    # Parameters
    area = st.session_state.area_code
    
    col1, col2, col3 = st.columns(3)
    with col1:
        data_type = st.radio("Dataset", ["Production", "Consumption"], horizontal=True)
    with col2:
        corr_year = st.selectbox("Year", options=[2021, 2022, 2023, 2024], index=0, key="corr_year")
    with col3:
        window_days = st.slider("Window (days)", min_value=7, max_value=90, value=30)
    
    # Update collection based on data type
    if data_type == "Production":
        collname = "production_2021_2024"
    else:
        collname = "consumption_2021_2024"
    
    col4, col5 = st.columns(2)
    with col4:
        met_var = st.selectbox(
            "Meteorological Variable",
            ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]
        )
    with col5:
        all_groups = get_groups_for_area(uri, dbname, collname, area)
        if not all_groups:
            st.warning(f"No {data_type.lower()} groups found.")
            st.stop()
        energy_group = st.selectbox(f"{data_type} Group", all_groups, index=0)
    
    # Fetch data
    with st.spinner(f"Loading data for {area} - {energy_group} ({corr_year})..."):
        # Get weather data for selected year
        df_weather = download_era5_hourly_for_area(area, corr_year)
        if met_var not in df_weather.columns:
            st.error(f"Variable {met_var} not found in weather data.")
            st.stop()
        
        # Get energy data for selected year
        cli = _mongo_client(uri)
        dt_start = datetime(corr_year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dt_end = datetime(corr_year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    pipe = [
        # Match either 'group' or 'productionGroup' field
        {"$match": {"priceArea": area, "$or": [{"group": energy_group}, {"productionGroup": energy_group}]}},
        {"$addFields": {
            "ts": {
                "$cond": [
                    {"$eq": [{"$type": "$startTime"}, "string"]},
                    {"$dateFromString": {"dateString": "$startTime"}},
                    "$startTime"
                ]
            }
        }},
        {"$match": {"ts": {"$gte": dt_start, "$lt": dt_end}}},
        {"$addFields": {
            "qty": {
                "$cond": [
                    {"$in": [{"$type": "$quantityKwh"}, ["int", "long", "double", "decimal"]]},
                    "$quantityKwh",
                    {"$toDouble": "$quantityKwh"}
                ]
            }
        }},
        {"$sort": {"ts": 1}}
    ]
    
    energy_docs = list(cli[dbname][collname].aggregate(pipe, allowDiskUse=True))
    if not energy_docs:
        st.error("No energy data found for selected group and area.")
        st.stop()
    
    df_energy = pd.DataFrame(energy_docs)
    df_energy["time"] = pd.to_datetime(df_energy["ts"]).dt.tz_localize("UTC")
    df_energy = df_energy.set_index("time").sort_index()
    df_energy = df_energy[["qty"]].rename(columns={"qty": "energy_kwh"})
    
    # Align weather and energy data
    df_weather_reset = df_weather.reset_index()
    df_weather_reset["time"] = pd.to_datetime(df_weather_reset["time"]).dt.tz_localize("UTC")
    df_weather_reset = df_weather_reset.set_index("time")
    
    # Merge on common time index
    df_combined = df_weather_reset[[met_var]].join(df_energy[["energy_kwh"]], how="inner")
    df_combined = df_combined.dropna()
    
    if df_combined.empty:
        st.error("No overlapping time data between weather and energy.")
        st.stop()
    
    st.success(f"Loaded {len(df_combined)} overlapping hourly records.")
    
    # Compute sliding window correlation with lag
    window_hours = window_days * 24
    max_lag_hours = 48
    lag_step = 6
    lags = list(range(0, max_lag_hours + 1, lag_step))
    
    correlation_results = []
    progress_bar = st.progress(0)
    
    for i, lag in enumerate(lags):
        # Shift energy data by lag hours
        df_lagged = df_combined.copy()
        df_lagged["energy_lagged"] = df_lagged["energy_kwh"].shift(lag)
        df_lagged = df_lagged.dropna()
        
        # Compute rolling correlation
        rolling_corr = df_lagged[met_var].rolling(window=window_hours).corr(df_lagged["energy_lagged"])
        
        # Store mean correlation for this lag
        mean_corr = rolling_corr.mean()
        correlation_results.append({"lag_hours": lag, "mean_correlation": mean_corr})
        progress_bar.progress((i + 1) / len(lags))
    
    progress_bar.empty()
    
    corr_df = pd.DataFrame(correlation_results)
    
    # Plot 1: Lag vs Correlation
    fig1 = px.line(corr_df, x="lag_hours", y="mean_correlation", 
                   title=f"Mean Correlation vs Lag: {met_var} → {energy_group}",
                   labels={"lag_hours": "Lag (hours)", "mean_correlation": "Mean Correlation"},
                   markers=True)
    fig1.add_hline(y=0, line_dash="dash", line_color="gray")
    fig1.update_layout(height=400)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Plot 2: Time series of rolling correlation at selected lag
    selected_lag = st.slider("Select Lag for Time Series Plot", min_value=0, max_value=max_lag_hours, value=0, step=lag_step)
    
    df_lag_selected = df_combined.copy()
    df_lag_selected["energy_lagged"] = df_lag_selected["energy_kwh"].shift(selected_lag)
    df_lag_selected = df_lag_selected.dropna()
    
    rolling_corr_ts = df_lag_selected[met_var].rolling(window=window_hours).corr(df_lag_selected["energy_lagged"])
    rolling_corr_ts = rolling_corr_ts.dropna()
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=rolling_corr_ts.index, y=rolling_corr_ts.values, mode="lines", name=f"Correlation (lag={selected_lag}h)"))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(title=f"Rolling Correlation Over Time (window={window_days}d, lag={selected_lag}h)",
                       xaxis_title="Time", yaxis_title="Correlation", height=400)
    st.plotly_chart(fig2, use_container_width=True)
    
    with st.expander("ℹ️ About Sliding Window Correlation"):
        st.markdown("""
        **Sliding Window Correlation** computes the Pearson correlation between a meteorological variable and energy production/consumption over a moving time window.
        
        **Parameters:**
        - **Window Length**: Size of the rolling window (in days) over which correlation is computed
        - **Lag**: Time shift (in hours) applied to the energy data to test lagged relationships
        - **Lag Step**: Increment for testing multiple lag values
        
        **Interpretation:**
        - Positive correlation: higher meteorological values → higher energy production
        - Negative correlation: higher meteorological values → lower energy production
        - Lag effects: correlation may peak at non-zero lags due to delayed impacts
        
        **Use Cases:**
        - Detect changes in correlation during extreme weather events
        - Identify optimal lag for predictive modeling
        - Understand weather-energy relationships in different seasons
        """)

# --- Page: Data Overview ---
elif page == "Data Overview":
    st.title("📊 Data Overview - ERA5 Meteorological Data")
    
    # Price area and year selectors
    col1, col2 = st.columns(2)
    with col1:
        area = st.selectbox("Price Area", options=["NO1", "NO2", "NO3", "NO4", "NO5"], 
                           index=["NO1", "NO2", "NO3", "NO4", "NO5"].index(st.session_state.area_code) if st.session_state.area_code in ["NO1", "NO2", "NO3", "NO4", "NO5"] else 4,
                           key="overview_area")
    with col2:
        overview_year = st.selectbox("Year", options=[2021, 2022, 2023, 2024], index=0, key="overview_year")
    
    with st.spinner(f"Loading ERA5 data for {area} ({overview_year})..."):
        df_weather = download_era5_hourly_for_area(area, overview_year)
    # Display first 50 rows of weather data
    st.write(f"Area {area}, first 50 rows:")
    st.dataframe(df_weather.head(50), use_container_width=True)

    # First month overview
    st.subheader("First month overview")
    first_month = df_weather.loc[df_weather.index.to_period('M') == df_weather.index.min().to_period('M')]
    value_cols = [c for c in df_weather.columns if c in
              ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]]
    rows = [{"Variable": col, "First Month": first_month[col].tolist()} for col in value_cols]
    chart_table = pd.DataFrame(rows)

    # Render first month trend as line chart
    st.dataframe(
        chart_table,
        column_config={
            "Variable": st.column_config.TextColumn("Variable"),
            "First Month": st.column_config.LineChartColumn("First month trend"),
        },
        hide_index=True,
        use_container_width=True,
    )

# --- Page new B: Outlier or SPC and Anomaly or LOF on weather ---
elif page == "Outliers & Anomalies":
    st.title("Outliers and anomalies, ERA5 weather")
    
    # Use current area selection
    area = st.session_state.area_code
    anomaly_year = st.selectbox("Year", options=[2021, 2022, 2023, 2024], index=0, key="anomaly_year")
    
    with st.expander("📚 Theoretical Background", expanded=False):
        st.markdown("""
        ### Statistical Process Control (SPC)
        **SPC** monitors processes to detect unusual variation. For outlier detection:
        - Apply **DCT high-pass filter** to remove low-frequency trends
        - Compute **robust statistics** (median and MAD) to estimate center and spread
        - Flag points beyond **k×σ** threshold as outliers
        
        **Applications**: Detect sensor errors, identify extreme weather events, monitor data quality.
        
        ### Local Outlier Factor (LOF)
        **LOF** is a density-based anomaly detection algorithm that:
        - Compares local density of each point to its neighbors
        - Identifies points in lower-density regions as anomalies
        - Works well for multivariate data with irregular cluster shapes
        
        **Applications**: Detect unusual weather patterns, identify rare event combinations, multivariate anomaly detection.
        
        **Key difference**: SPC is univariate and threshold-based; LOF is multivariate and density-based.
        """)

    with st.spinner(f"Loading ERA5 data for {area} ({anomaly_year})..."):
        df_weather = download_era5_hourly_for_area(area, anomaly_year)

    # DCT high-pass filter helper function
    def _dct_highpass(series: pd.Series, cutoff=30):
        x = series.values.astype(float)
        n = len(x)
        if np.isnan(x).any():
            idx = np.arange(n)
            ok = np.isfinite(x)
            x = x.copy()
            x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
        X = dct(x, norm="ortho")
        X[:cutoff] = 0.0
        x_hp = idct(X, norm="ortho")
        return pd.Series(x_hp, index=series.index)

    # Tabs for different anomaly detection methods
    tabs = st.tabs(["SPC Outliers - Temperature", "LOF Anomalies - Precipitation"])

    # Outlier or SPC tab - Temperature only
    with tabs[0]:
        st.subheader("SPC Outliers on Temperature")

        with st.expander("Quick help", expanded=False):
            st.markdown(
                "- Adjust High pass strength and Sigma threshold.\n"
                "- Baseline window defines a rolling mean used to draw the SPC bands."
            )

        var_spc = "temperature_2m"

        # SPC parameters
        c1, c2, c3 = st.columns(3)
        with c1:
            cutoff = st.slider(
                "High pass strength",
                min_value=5, max_value=200, value=30, step=1,
                help="Larger value removes more slow variation and highlights spikes"
            )
        with c2:
            k_sigma = st.slider(
                "Sigma threshold",
                min_value=2.0, max_value=6.0, value=3.5, step=0.1,
                help="Points outside baseline ± k times sigma are flagged"
            )
        with c3:
            base_win = st.slider(
                "Baseline window, hours",
                min_value=24, max_value=24*60, value=24*14, step=24,
                help="Rolling window size for the baseline"
            )

        # Process the selected variable
        series = pd.to_numeric(df_weather[var_spc], errors="coerce")
        if series.isna().all():
            st.info("Selected variable contains only missing values.")
        else:
            satv = _dct_highpass(series, cutoff=int(cutoff))

            med = np.median(satv.values)
            mad = np.median(np.abs(satv.values - med))
            sigma = 1.4826 * mad if mad > 0 else float(np.nanstd(satv.values))

            lower = med - float(k_sigma) * sigma
            upper = med + float(k_sigma) * sigma
            is_out = (satv < lower) | (satv > upper)

            baseline = series.rolling(window=int(base_win), min_periods=1, center=True).mean()
            b_lower = baseline + (lower - med)
            b_upper = baseline + (upper - med)

            # plot the SPC bands and mark outliers
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Temperature"))
            fig.add_trace(go.Scatter(x=b_lower.index, y=b_lower.values, mode="lines", name="SPC lower", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=b_upper.index, y=b_upper.values, mode="lines", name="SPC upper", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=series.index[is_out], y=series.values[is_out], mode="markers", name="Outliers", marker=dict(color="red", size=6)))
            fig.update_layout(title="SPC Outliers - Temperature", xaxis_title="Time", yaxis_title="Celsius", height=450)
            st.plotly_chart(fig, use_container_width=True)

            # summary statistics table for SPC
            n_total = int(series.notna().sum())
            n_out = int(is_out.sum())
            share = (100.0 * n_out / n_total) if n_total else 0.0
            satv_vals = satv.values[np.isfinite(satv.values)]

            spc_summary = pd.DataFrame(
                {
                    "metric": [
                        "points analyzed",
                        "outliers detected",
                        "outlier share, percent",
                        "robust center, SATV",
                        "robust sigma, SATV",
                        "lower threshold, SATV",
                        "upper threshold, SATV",
                        "series min",
                        "series max",
                        "series mean",
                        "series std",
                    ],
                    "value": [
                        n_total,
                        n_out,
                        round(share, 3),
                        round(float(med), 4),
                        round(float(sigma), 4),
                        round(float(lower), 4),
                        round(float(upper), 4),
                        round(float(np.nanmin(series.values)), 4),
                        round(float(np.nanmax(series.values)), 4),
                        round(float(np.nanmean(series.values)), 4),
                        round(float(np.nanstd(series.values)), 4),
                    ],
                }
            )
            st.subheader("SPC summary")
            st.dataframe(spc_summary, use_container_width=True, hide_index=True)


    # Anomaly or LOF tab - Precipitation only
    with tabs[1]:
        st.subheader("LOF Anomalies on Precipitation")

        with st.expander("Quick help", expanded=False):
            st.markdown(
                "- Multivariate LOF analysis uses raw precipitation value plus rolling windows.\n"
                "- Contamination is the expected share of anomalies.\n"
                "- Neighbors controls how local the detector is."
            )

        var_lof = "precipitation"

        # LOF parameters
        contamination = st.slider(
            "Contamination, share of anomalies",
            min_value=0.001, max_value=0.05, value=0.01, step=0.001,
            help="Typical values 0.005 to 0.02",
            key="lof_contamination"
        )
        n_neighbors = st.slider(
            "Neighbors",
            min_value=10, max_value=100, value=35, step=1,
            help="More neighbors gives a smoother detector",
            key="lof_neighbors"
        )

        # Prepare data for LOF
        lof_series = pd.to_numeric(df_weather[var_lof], errors="coerce").fillna(0.0)
        roll3 = lof_series.rolling(3, min_periods=1).mean()
        roll24 = lof_series.rolling(24, min_periods=1).mean()
        X = np.c_[lof_series.values, roll3.values, roll24.values]

        lof = LocalOutlierFactor(n_neighbors=int(n_neighbors), contamination=float(contamination))
        y_pred = lof.fit_predict(X)
        is_out_lof = y_pred == -1

        # plot LOF anomalies
        fig_lof = go.Figure()
        fig_lof.add_trace(go.Scatter(x=lof_series.index, y=lof_series.values, mode="lines", name="Precipitation"))
        fig_lof.add_trace(go.Scatter(x=lof_series.index[is_out_lof], y=lof_series.values[is_out_lof], mode="markers", name="Anomalies", marker=dict(color="red", size=6)))
        fig_lof.update_layout(title="LOF Anomalies - Precipitation", xaxis_title="Time", yaxis_title="mm per hour", height=450)
        st.plotly_chart(fig_lof, use_container_width=True)

        # summary statistics table for LOF
        n_total_lof = int(len(lof_series))
        n_anom = int(is_out_lof.sum())
        share_lof = (100.0 * n_anom / n_total_lof) if n_total_lof else 0.0

        lof_summary = pd.DataFrame(
            {
                "metric": [
                    "points analyzed",
                    "anomalies detected",
                    "anomaly share, percent",
                    "contamination setting",
                    "neighbors used",
                    "variable min",
                    "variable max",
                    "variable mean",
                    "variable std",
                ],
                "value": [
                    n_total_lof,
                    n_anom,
                    round(share_lof, 3),
                    float(contamination),
                    int(n_neighbors),
                    round(float(np.nanmin(lof_series.values)), 4),
                    round(float(np.nanmax(lof_series.values)), 4),
                    round(float(np.nanmean(lof_series.values)), 4),
                    round(float(np.nanstd(lof_series.values)), 4),
                ],
            }
        )
        st.subheader("LOF summary")
        st.dataframe(lof_summary, use_container_width=True, hide_index=True)

# --- Page: SARIMAX Forecasting ---
elif page == "SARIMAX Forecasting":
    st.title("🔮 SARIMAX Forecasting - Energy Production/Consumption")
    
    # Need statsmodels SARIMAX
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        sarimax_available = True
    except ImportError:
        st.error("statsmodels not available.")
        st.stop()
    
    uri, dbname, _ = _mongo_settings()
    if not uri:
        st.error("MongoDB URI is missing for energy data.")
        st.stop()
    
    # Dataset selection
    col1, col2 = st.columns(2)
    with col1:
        data_type = st.radio("Dataset", ["Production", "Consumption"], horizontal=True, key="sarimax_data_type")
    with col2:
        forecast_days = st.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30)
    
    # Update collection based on data type
    if data_type == "Production":
        collname = "production_2021_2024"
    else:
        collname = "consumption_2021_2024"
    
    areas = get_price_areas(uri, dbname, collname)
    if not areas:
        st.warning("No price areas found.")
        st.stop()
    
    # Price area selection
    area_idx = areas.index(st.session_state.area_code) if st.session_state.area_code in areas else 0
    area = st.selectbox("Price Area", areas, index=area_idx, key="sarimax_area")
    st.session_state.area_code = area
    
    all_groups = get_groups_for_area(uri, dbname, collname, area)
    if not all_groups:
        st.warning(f"No {data_type.lower()} groups found.")
        st.stop()
    
    # Group and exogenous variable selection
    col3, col4 = st.columns(2)
    with col3:
        target_group = st.selectbox(f"{data_type} Group", all_groups, index=0)
    with col4:
        use_exog = st.checkbox("Include temperature as exogenous variable", value=False)
    
    # Get actual date range from database
    @st.cache_data(ttl=3600)
    def get_date_range_sarimax(uri: str, db: str, coll: str):
        cli = _mongo_client(uri)
        pipe = [
            {"$addFields": {"ts": {"$cond": [{"$eq": [{"$type": "$startTime"}, "string"]}, {"$dateFromString": {"dateString": "$startTime"}}, "$startTime"]}}},
            {"$group": {"_id": None, "min": {"$min": "$ts"}, "max": {"$max": "$ts"}}}
        ]
        res = list(cli[db][coll].aggregate(pipe, allowDiskUse=True))
        if res and res[0].get("min") and res[0].get("max"):
            return res[0]["min"].date(), res[0]["max"].date()
        return date(2021, 1, 1), date(2024, 12, 31)
    
    min_date, max_date = get_date_range_sarimax(uri, dbname, collname)
    
    # Training period selection with date pickers
    st.subheader("Training Period")
    col5, col6 = st.columns(2)
    with col5:
        train_start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key="sarimax_start")
    with col6:
        train_end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key="sarimax_end")
    
    # Validate training period
    if train_start_date >= train_end_date:
        st.error("❌ Training start date must be before end date.")
        st.stop()
    
    # SARIMAX parameters
    with st.expander("⚙️ SARIMAX Model Parameters", expanded=False):
        st.markdown("""
        **SARIMAX(p,d,q)(P,D,Q,S)** parameters:
        - **p, d, q**: Non-seasonal AR, differencing, MA orders
        - **P, D, Q**: Seasonal AR, differencing, MA orders  
        - **S**: Seasonal period (e.g., 24 for daily patterns in hourly data)
        """)
        col_a, col_b = st.columns(2)
        with col_a:
            p = st.slider("p (AR order)", 0, 3, 1)
            d = st.slider("d (Differencing)", 0, 2, 1)
            q = st.slider("q (MA order)", 0, 3, 1)
        with col_b:
            P = st.slider("P (Seasonal AR)", 0, 2, 1)
            D = st.slider("D (Seasonal Diff)", 0, 1, 1)
            Q = st.slider("Q (Seasonal MA)", 0, 2, 1)
            S = st.slider("S (Seasonal period)", 12, 48, 24)
    
    exog_vars = ["temperature_2m"] if use_exog else []
    
    # Fetch energy data
    cli = _mongo_client(uri)
    dt_start = datetime.combine(train_start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    dt_end = datetime.combine(train_end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
    
    pipe = [
        # Match either 'group' or 'productionGroup' field
        {"$match": {"priceArea": area, "$or": [{"group": target_group}, {"productionGroup": target_group}]}},
        {"$addFields": {
            "ts": {
                "$cond": [
                    {"$eq": [{"$type": "$startTime"}, "string"]},
                    {"$dateFromString": {"dateString": "$startTime"}},
                    "$startTime"
                ]
            }
        }},
        {"$match": {"ts": {"$gte": dt_start, "$lt": dt_end}}},
        {"$addFields": {
            "qty": {
                "$cond": [
                    {"$in": [{"$type": "$quantityKwh"}, ["int", "long", "double", "decimal"]]},
                    "$quantityKwh",
                    {"$toDouble": "$quantityKwh"}
                ]
            }
        }},
        {"$sort": {"ts": 1}}
    ]
    
    energy_docs = list(cli[dbname][collname].aggregate(pipe, allowDiskUse=True))
    if not energy_docs:
        st.error("No energy data found for selected group and training period.")
        st.stop()
    
    df_energy = pd.DataFrame(energy_docs)
    df_energy["time"] = pd.to_datetime(df_energy["ts"]).dt.tz_localize("UTC")
    df_energy = df_energy.set_index("time").sort_index()
    df_energy = df_energy[["qty"]].rename(columns={"qty": "energy_kwh"})
    
    # Resample to daily for easier modeling
    df_daily = df_energy.resample("D").sum()
    
    st.success(f"Loaded {len(df_daily)} daily records for training.")
    
    # Prepare exogenous variables if selected
    exog_train = None
    exog_forecast = None
    if use_exog:
        # Download weather data for all years in training period
        df_weather_list = []
        for year in range(train_start_date.year, train_end_date.year + 1):
            df_year = download_era5_hourly_for_area(area, year)
            df_weather_list.append(df_year)
        df_weather = pd.concat(df_weather_list).sort_index()
        
        weather_daily = df_weather["temperature_2m"].resample("D").mean().to_frame()
        exog_train = weather_daily.reindex(df_daily.index).dropna()
        
        # Ensure we have matching data
        if len(exog_train) != len(df_daily):
            common_idx = df_daily.index.intersection(exog_train.index)
            df_daily = df_daily.loc[common_idx]
            exog_train = exog_train.loc[common_idx]
        
        last_val = exog_train.iloc[-1]
        exog_forecast = pd.DataFrame([last_val] * forecast_days, columns=["temperature_2m"])
        exog_forecast.index = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    
    # Fit SARIMAX model
    if st.button("🚀 Run SARIMAX Forecast", type="primary"):
        with st.spinner("Fitting SARIMAX model..."):
            try:
                model = SARIMAX(
                    df_daily["energy_kwh"],
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, S),
                    exog=exog_train if use_exog else None,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                results = model.fit(disp=False)
                
                st.success("✅ Model fitted successfully!")
                
                # Forecast
                forecast_obj = results.get_forecast(steps=forecast_days, exog=exog_forecast if use_exog else None)
                forecast_mean = forecast_obj.predicted_mean
                forecast_ci = forecast_obj.conf_int()
                
                # Plot
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=df_daily.index,
                    y=df_daily["energy_kwh"],
                    mode="lines",
                    name="Historical",
                    line=dict(color="blue")
                ))
                
                # Forecast mean
                fig.add_trace(go.Scatter(
                    x=forecast_mean.index,
                    y=forecast_mean.values,
                    mode="lines",
                    name="Forecast",
                    line=dict(color="red", dash="dash")
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_ci.index.tolist() + forecast_ci.index.tolist()[::-1],
                    y=forecast_ci.iloc[:, 1].tolist() + forecast_ci.iloc[:, 0].tolist()[::-1],
                    fill="toself",
                    fillcolor="rgba(255,0,0,0.2)",
                    line=dict(color="rgba(255,0,0,0)"),
                    name="95% CI"
                ))
                
                fig.update_layout(
                    title=f"SARIMAX Forecast: {target_group} in {area}",
                    xaxis_title="Date",
                    yaxis_title="Energy (kWh/day)",
                    height=500,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                st.subheader("📊 Forecast Values")
                forecast_df = pd.DataFrame({
                    "Date": forecast_mean.index.date,
                    "Forecast (kWh)": forecast_mean.values.round(2),
                    "Lower 95% CI": forecast_ci.iloc[:, 0].values.round(2),
                    "Upper 95% CI": forecast_ci.iloc[:, 1].values.round(2)
                })
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Model fitting failed: {str(e)}")
                st.info("Try adjusting parameters or check data quality.")
    
    with st.expander("ℹ️ About SARIMAX"):
        st.markdown("""
        **SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) is a powerful time series forecasting model.
        
        **Parameters:**
        - **(p, d, q)**: Non-seasonal AR, differencing, and MA orders
        - **(P, D, Q, S)**: Seasonal AR, differencing, MA orders, and seasonal period
        - **Exogenous variables**: External factors (e.g., weather) that may influence the target
        
        **Model Selection:**
        - Start with simple models (e.g., (1,1,1) and (1,1,1,24))
        - Use AIC/BIC for model comparison
        - Validate with residual diagnostics
        
        **Interpretation:**
        - Forecast line shows predicted mean values
        - Confidence intervals reflect prediction uncertainty
        - Wider intervals indicate higher uncertainty
        
        **Use Cases:**
        - Short to medium-term energy production forecasting
        - Incorporate weather forecasts as exogenous variables
        - Seasonal pattern modeling (daily, weekly cycles)
        """)

# --- Page: Snow Drift ---
elif page == "Snow Drift":
    st.title("❄️ Snow Drift Calculations")

    # Try to import the provided Snow_drift module
    try:
        import Snow_drift as sd
        snow_module_available = True
    except Exception as e:
        snow_module_available = False
        sd = None
        st.warning(f"Snow_drift.py not available: {e}")

    st.markdown("""
    Calculate snow drift for selected coordinates. Select a location from the Map page.
    
    **Note:** Seasonal years run from **July 1 to June 30** (e.g., season 2021 = July 1, 2021 to June 30, 2022).
    """)

    # Check for selected coordinates
    sel_coord = st.session_state.get("selected_coord", None)
    sel_area = st.session_state.get("selected_area", None)
    
    if sel_coord:
        st.info(f"📍 Using selected coordinates: {sel_coord[0]:.4f}, {sel_coord[1]:.4f}" + (f" (Area: {sel_area})" if sel_area else ""))
    else:
        st.warning("⚠️ No location selected. Please use the **Map & Regions** page to select coordinates first.")

    st.markdown("**Year Range**")
    
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=1950, max_value=2023, value=2020, step=1, help="First seasonal year")
    with col2:
        end_year = st.number_input("End Year", min_value=1950, max_value=2023, value=2021, step=1, help="Last seasonal year")
    
    if start_year > end_year:
        st.error("Start year must be ≤ end year")
        st.stop()
    
    # Use default parameters from Snow_drift.py
    T = 3000.0
    F = 30000.0
    theta = 0.5

    # Fetch ERA5 data for selected location
    df_snow = None
    area_code = st.session_state.get("area_code", None)
    
    if st.button("🌍 Fetch ERA5 Data", type="primary", use_container_width=True):
        if not sel_coord:
            st.error("⚠️ No location selected. Please select a location on the **Map & Regions** page first.")
        else:
            try:
                # Use actual selected coordinates
                lat, lon = sel_coord[0], sel_coord[1]
                
                # Fetch data for seasonal years using arbitrary coordinates
                with st.spinner(f"Fetching ERA5 data for coordinates ({lat:.2f}, {lon:.2f}) from {start_year}-{end_year+1}..."):
                    df_era = download_era5_hourly_coords(
                        lat=lat,
                        lon=lon,
                        year_start=start_year,
                        year_end=end_year + 1,  # Need data through June of end_year+1
                        hourly_vars=["temperature_2m", "precipitation", "wind_speed_10m", "wind_direction_10m"],
                        tz="UTC"
                    )
                    df_snow = df_era.reset_index()
                    df_snow.rename(columns={"time": "time"}, inplace=True)
                    
                    st.success(f"✅ Fetched {len(df_snow)} hourly records covering seasons {start_year}-{end_year}")
                    
            except Exception as e:
                st.error(f"ERA5 fetch failed: {e}")
                df_snow = None

    if df_snow is None:
        st.info("👆 Click **Fetch ERA5 Data** above to load meteorological data for snow drift calculations.")
        st.stop()

    # Normalize column names for Snow_drift expectations
    colmap = {
        "precipitation": "precipitation (mm)",
        "temperature_2m": "temperature_2m (°C)",
        "wind_speed_10m": "wind_speed_10m (m/s)",
        "wind_direction_10m": "wind_direction_10m (°)"
    }
    for src, dst in colmap.items():
        if src in df_snow.columns and dst not in df_snow.columns:
            df_snow[dst] = df_snow[src]
    
    # Ensure time column exists
    if "time" not in df_snow.columns:
        if df_snow.index.name == "time":
            df_snow = df_snow.reset_index()
        else:
            st.error("No time column found in the data.")
            st.stop()
    
    df_snow["time"] = pd.to_datetime(df_snow["time"]).dt.tz_localize(None)
    
    # Create season column: year if month >= 7 (July-Dec), else year - 1 (Jan-June)
    # Season 2020 = July 1, 2020 to June 30, 2021
    df_snow["season"] = df_snow["time"].apply(lambda dt: dt.year if dt.month >= 7 else dt.year - 1)
    
    # Filter to selected seasonal years
    df_snow = df_snow[df_snow["season"].between(start_year, end_year)]
    
    if df_snow.empty:
        st.error(f"No data found for seasons {start_year}-{end_year}. Try fetching more years.")
        st.stop()

    # Call Snow_drift functions
    if not snow_module_available:
        st.error("`Snow_drift.py` not available or failed to import. Ensure the file is present and named exactly `Snow_drift.py`.")
        st.stop()

    @st.cache_data
    def compute_snow_drift_results(df, T_val, F_val, theta_val):
        yearly = sd.compute_yearly_results(df, float(T_val), float(F_val), float(theta_val))
        sectors = sd.compute_average_sector(df)
        return yearly, sectors
    
    @st.cache_data
    def compute_monthly_snow_drift(df, T_val, F_val, theta_val):
        """Compute monthly snow drift for each season"""
        monthly_results = []
        for season in sorted(df["season"].unique()):
            # Get data for this season (July 1 to June 30)
            season_start = pd.Timestamp(year=season, month=7, day=1)
            season_end = pd.Timestamp(year=season+1, month=6, day=30, hour=23, minute=59, second=59)
            season_data = df[(df['time'] >= season_start) & (df['time'] <= season_end)].copy()
            
            if season_data.empty:
                continue
            
            # Add month column
            season_data["month"] = season_data["time"].dt.month
            
            for month in sorted(season_data["month"].unique()):
                month_data = season_data[season_data["month"] == month].copy()
                if len(month_data) < 24:  # Need at least 1 day of data
                    continue
                
                # Calculate Swe for this month
                month_data['Swe_hourly'] = month_data.apply(
                    lambda row: row['precipitation (mm)'] if row['temperature_2m (°C)'] < 1 else 0, axis=1)
                total_Swe = month_data['Swe_hourly'].sum()
                wind_speeds = month_data["wind_speed_10m (m/s)"].tolist()
                
                # Compute transport for this month
                result = sd.compute_snow_transport(float(T_val), float(F_val), float(theta_val), total_Swe, wind_speeds)
                monthly_results.append({
                    "season": season,
                    "month": month,
                    "Qt (kg/m)": result["Qt (kg/m)"],
                    "Control": result["Control"]
                })
        
        return pd.DataFrame(monthly_results)

    with st.spinner("Computing snow drift results..."):
        try:
            yearly_df, avg_sectors = compute_snow_drift_results(df_snow, float(T), float(F), float(theta))
            monthly_df = compute_monthly_snow_drift(df_snow, float(T), float(F), float(theta))
        except Exception as e:
            st.error(f"Snow drift calculations failed: {e}")
            st.stop()

    if yearly_df.empty:
        st.warning("No seasonal results computed. Check if data covers the selected year range.")
    else:
        # Display results
        st.subheader("📈 Annual Snow Drift Results")
        
        # Format display dataframe
        yearly_df_disp = yearly_df.copy()
        yearly_df_disp["Qt (tonnes/m)"] = yearly_df_disp["Qt (kg/m)"] / 1000.0
        display_cols = ["season", "Qt (tonnes/m)"]
        st.dataframe(
            yearly_df_disp[display_cols], 
            use_container_width=True,
            column_config={
                "season": "Season (July-June)",
                "Qt (tonnes/m)": st.column_config.NumberColumn("Qt (tonnes/m)", format="%.2f")
            },
            hide_index=True
        )

        overall_avg = yearly_df["Qt (kg/m)"].mean()
        
        # Plot monthly and yearly together - yearly as semi-transparent background bars
        st.subheader("📊 Snow Drift - Yearly and Monthly")
        
        if not monthly_df.empty:
            # Create month names for better readability
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            monthly_df['month_name'] = monthly_df['month'].map(month_names)
            monthly_df['season_month'] = monthly_df['season'].astype(str) + '-' + monthly_df['month_name']
            
            fig_combined = go.Figure()
            
            # First add yearly totals as semi-transparent background bars
            yearly_added = False
            for idx, row in yearly_df.iterrows():
                season = row['season']
                season_year = int(season.split('-')[0])
                # Find months belonging to this season
                season_months = monthly_df[monthly_df['season'] == season_year]['season_month'].tolist()
                if season_months:
                    # Add a bar for each month in the season with the yearly total value
                    for month in season_months:
                        fig_combined.add_trace(go.Bar(
                            x=[month],
                            y=[row['Qt (kg/m)']],
                            name="Yearly" if not yearly_added else None,
                            marker_color='rgba(173, 216, 230, 0.4)',  # Semi-transparent light blue
                            showlegend=not yearly_added,
                            legendgroup="yearly"
                        ))
                        yearly_added = True
            
            # Then add monthly values on top
            fig_combined.add_trace(go.Bar(
                x=monthly_df['season_month'],
                y=monthly_df['Qt (kg/m)'],
                name="Monthly",
                marker_color='rgba(52, 152, 219, 0.8)'
            ))
            
            # Add average line
            fig_combined.add_hline(
                y=overall_avg, 
                line_dash="dash", 
                line_color="gray",
                annotation_text=f"Avg: {overall_avg:.1f} kg/m",
                annotation_position="right"
            )
            
            fig_combined.update_layout(
                title="Snow Drift Transport - Monthly and Yearly",
                xaxis_title="Season-Month",
                yaxis_title="Qt (kg/m)",
                height=500,
                showlegend=True,
                xaxis_tickangle=-45,
                barmode='overlay'
            )
            
            st.plotly_chart(fig_combined, use_container_width=True)
            
            # Summary table grouped by month
            st.subheader("📋 Monthly Statistics Across All Seasons")
            monthly_summary = monthly_df.groupby('month_name')['Qt (kg/m)'].agg(['mean', 'min', 'max', 'count']).round(2)
            monthly_summary = monthly_summary.reindex([month_names[i] for i in range(1, 13) if month_names[i] in monthly_summary.index])
            monthly_summary.columns = ['Mean Qt (kg/m)', 'Min Qt (kg/m)', 'Max Qt (kg/m)', 'Seasons']
            st.dataframe(monthly_summary, use_container_width=True)
        else:
            # If no monthly data, just show yearly
            fig_qt = px.bar(yearly_df, x="season", y="Qt (kg/m)", 
                            title="Annual Snow Drift Transport (Qt)",
                            labels={"season": "Season (July-June)", "Qt (kg/m)": "Qt (kg/m)"})
            fig_qt.update_layout(height=350, showlegend=False)
            fig_qt.add_hline(y=overall_avg, line_dash="dash", line_color="gray", 
                            annotation_text=f"Avg: {overall_avg:.1f} kg/m")
            st.plotly_chart(fig_qt, use_container_width=True)
        
        st.subheader("🌹 Wind Rose - Directional Distribution")
        st.metric(
            label="Overall Average Qt", 
            value=f"{overall_avg/1000.0:.2f} tonnes/m",
            help="Mean annual snow transport across all seasons"
        )
        
        # Show polar bar chart
        directions = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
        vals_tonnes = np.array(avg_sectors) / 1000.0
        fig = go.Figure()
        fig.add_trace(go.Barpolar(
            r=vals_tonnes.tolist(), 
            theta=directions, 
            name="Snow transport",
            marker_color="lightblue",
            marker_line_color="navy",
            marker_line_width=1
        ))
        fig.update_layout(
            title=f"Average Directional Snow Transport",
            polar=dict(
                radialaxis=dict(showticklabels=True, title="tonnes/m"),
                angularaxis=dict(direction="clockwise")
            ),
            showlegend=False,
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("ℹ️ About Snow Drift Calculations"):
        st.markdown("""
        **Snow Drift Methodology:**
        
        Snow drift calculations compute the mean annual snow transport (**Qt**) based on:
        
        - Wind-driven transport potential (using wind speed^3.8)
        - Snowfall amounts over the season
        - Transport and fetch distances
        - Relocation coefficient
        
        **Parameters:**
        - **T**: Maximum transport distance (how far snow can travel)
        - **F**: Fetch distance (upwind distance snow originates from)
        - **θ**: Relocation coefficient (fraction of snowfall that gets relocated by wind)
        
        **Seasonal Years:**
        - Defined as July 1 to June 30 (winter season spans two calendar years)
        - Season 2020 = July 1, 2020 to June 30, 2021
        
        **Wind Rose:**
        - Shows directional distribution of snow transport
        - 16 sectors (N, NNE, NE, etc.)
        - Values in tonnes/m
        """)
