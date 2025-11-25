e# streamlit_app.py - fixed

import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt
import os

#  imports
import numpy as np
import matplotlib.pyplot as plt
import requests
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
    return MongoClient(uri)

# Retrieve distinct price areas from the MongoDB collection
@st.cache_data(ttl=300)
def get_price_areas(uri: str, db: str, coll: str):
    cli = _mongo_client(uri)
    return sorted(cli[db][coll].distinct("priceArea"))

# Retrieve production groups for a specific price area
@st.cache_data(ttl=300)
def get_groups_for_area(uri: str, db: str, coll: str, price_area: str):
    cli = _mongo_client(uri)
    return sorted(cli[db][coll].distinct("productionGroup", {"priceArea": price_area}))

from datetime import datetime, timezone

# Retrieve yearly totals for a specific price area
@st.cache_data(ttl=300)
def get_year_totals(uri: str, db: str, coll: str, price_area: str) -> pd.DataFrame:
    cli = _mongo_client(uri)

    dt_start = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    dt_end   = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

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
            }
        }},
        # group
        {"$group": {"_id": "$productionGroup", "totalKwh": {"$sum": "$qty"}}},
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
        match["productionGroup"] = {"$in": groups}

    pipe = [
        {"$match": match},
        {"$addFields": {
            "ts": {
                "$cond": [
                    {"$eq": [{"$type": "$startTime"}, "string"]},
                    {"$dateFromString": {"dateString": "$startTime"}},
                    "$startTime"
                ]
            }
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
        {"$project": {"_id": 0, "productionGroup": 1, "startTime": "$ts", "quantityKwh": "$qty"}}
    ]

    rows = list(cli[db][coll].aggregate(pipe, allowDiskUse=True))
    if not rows:
        return pd.DataFrame(columns=["startTime", "productionGroup", "quantityKwh"])

    df = pd.DataFrame(rows)
    # ts is already Date, ensure tz aware and sorted
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df["quantityKwh"] = pd.to_numeric(df["quantityKwh"], errors="coerce").fillna(0.0)
    return df.sort_values("startTime")

# --- App setup ---
st.set_page_config(page_title="IND320 Project", layout="wide")
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = True

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

# --- Navigation in new order: 1, 4, new A, 2, 3, new B ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Elhub", "STL & Spectrogram", "Data Table", "Plots", "Outliers & Anomalies"]
)
# --- Shared selection state for area ---
if "area_code" not in st.session_state:
    st.session_state.area_code = "NO5"

# --- Page 1: Home ---
if page == "Home":
    st.title("IND320 Project - Weather and Power")
    st.write("Use the sidebar to navigate. Choose area on the Elhub page, then explore new A, Data Table, Plots, and new B.")

# --- Page 4 moved to front: Elhub ---
elif page == "Elhub":
    st.title("Elhub - Production per group, 2021")

    uri, dbname, collname = _mongo_settings()
    if not uri:
        st.error("MongoDB URI is missing. Add it to st.secrets['mongo']['uri'] or set MONGODB_URI.")
        st.stop()

    # split into two columns
    left, right = st.columns(2)

    # left side - radio for price area and a pie chart for full 2021
    with left:
        st.subheader("Totals pie")
        areas = get_price_areas(uri, dbname, collname)
        if not areas:
            st.warning("No price areas found.")
            st.stop()
        idx = areas.index(st.session_state.area_code) if st.session_state.area_code in areas else 0
        area = st.radio("Price area", options=areas, index=idx, horizontal=True)
        st.session_state.area_code = area

        pie_df = get_year_totals(uri, dbname, collname, area)
        if pie_df.empty:
            st.info("No data for this area in 2021.")
        else:
            total_sum = pie_df["totalKwh"].sum()
            pie_df["pct"] = (100 * pie_df["totalKwh"] / total_sum).round(1)
            pie_chart = (
                alt.Chart(pie_df)
                .mark_arc(outerRadius=120)
                .encode(
                    theta=alt.Theta("totalKwh:Q"),
                    color=alt.Color("productionGroup:N", legend=alt.Legend(title="Production group")),
                    tooltip=[
                        alt.Tooltip("productionGroup:N", title="Group"),
                        alt.Tooltip("totalKwh:Q", title="Total kWh", format=",.0f"),
                        alt.Tooltip("pct:Q", title="Share", format=".1f"),
                    ],
                )
                .properties(height=350)
            )
            st.altair_chart(pie_chart, use_container_width=True)

    # right side - group selector, month selector, line plot
    with right:
        st.subheader("Monthly lines")
        all_groups = get_groups_for_area(uri, dbname, collname, area)

        pills_fn = getattr(st, "pills", None)
        if pills_fn is not None:
            selected_groups = pills_fn("Production groups", options=all_groups, selection_mode="multi")
        else:
            selected_groups = st.multiselect("Production groups", options=all_groups, default=all_groups)

        month_str = st.selectbox("Month", options=[f"2021-{m:02d}" for m in range(1, 13)], index=0)
        year = 2021
        month = int(month_str.split("-")[1])

        mdf = get_month_series(uri, dbname, collname, area, year, month, selected_groups)
        if mdf.empty:
            st.info("No rows for this selection.")
        else:
            # aggregate to daily totals
            mdf["day"] = mdf["startTime"].dt.tz_convert("UTC").dt.tz_localize(None).dt.date
            daily = mdf.groupby(["day", "productionGroup"], as_index=False)["quantityKwh"].sum()

            line = (
                alt.Chart(daily)
                .mark_line()
                .encode(
                    x=alt.X("day:T", title="Day", axis=alt.Axis(format="%d")),
                    y=alt.Y("quantityKwh:Q", title="kWh per day"),
                    color=alt.Color("productionGroup:N", legend=alt.Legend(title="Production group")),
                    tooltip=[
                        alt.Tooltip("day:T", title="Day", format="%Y-%m-%d"),
                        alt.Tooltip("productionGroup:N", title="Group"),
                        alt.Tooltip("quantityKwh:Q", title="kWh", format=",.0f"),
                    ],
                )
                .properties(height=350)
            )
            st.altair_chart(line, use_container_width=True)

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

    uri, dbname, collname = _mongo_settings()
    if not uri:
        st.error("MongoDB URI is missing. Add it to st.secrets['mongo']['uri'] or set MONGODB_URI.")
        st.stop()

    area = st.session_state.area_code
    all_groups = get_groups_for_area(uri, dbname, collname, area)
    if not all_groups:
        st.warning("No production groups found for this area.")
        st.stop()

    tabs = st.tabs(["STL", "Spectrogram"])

    def _series_for_group(grp: str) -> pd.Series:
        from datetime import datetime, timezone
        cli = _mongo_client(uri)

        dt_start = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dt_end   = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        pipe = [
            {"$match": {"priceArea": area, "productionGroup": grp}},
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

        # enforce odd windows and tell the user what we will use
        seasonal_eff = int(seasonal) if int(seasonal) % 2 == 1 else int(seasonal) + 1
        trend_eff = int(trend) if int(trend) % 2 == 1 else int(trend) + 1
        st.caption(f"Using seasonal window {seasonal_eff} and trend window {trend_eff} (both must be odd).")

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
                stl_fit = STL(
                    y,
                    period=int(period),
                    seasonal=seasonal_eff,
                    trend=trend_eff,
                    robust=bool(robust)
                ).fit()

                fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
                ax[0].plot(stl_fit.observed); ax[0].set_title(f"Observed {area} {grp}")
                ax[1].plot(stl_fit.trend);    ax[1].set_title("Trend")
                ax[2].plot(stl_fit.seasonal); ax[2].set_title("Seasonal")
                ax[3].plot(stl_fit.resid);    ax[3].set_title("Remainder")
                fig.tight_layout()
                st.pyplot(fig)

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
                f, t, Sxx = _scipy_spectrogram(
                    y, fs=1.0, nperseg=nperseg, noverlap=noverlap,
                    scaling="density", mode="magnitude"
                )

                fig, ax = plt.subplots(figsize=(14, 5))
                im = ax.pcolormesh(t, f, Sxx, shading="auto")
                ax.set_title(f"Spectrogram for {grp} in {area}")
                ax.set_xlabel("Time index")
                ax.set_ylabel("Frequency, cycles per hour")
                fig.colorbar(im, ax=ax, label="Magnitude")
                fig.tight_layout()
                st.pyplot(fig)


# --- Page 2: Data Table now uses ERA5 API and selected area ---
elif page == "Data Table":
    st.title("Data Table - ERA5 2021 via Open Meteo")
    area = st.session_state.area_code
    df_weather = download_era5_hourly_for_area(area, 2021)
    # explanation for displaying the first 50 rows of weather data
    st.write(f"Area {area}, first 50 rows:")
    st.dataframe(df_weather.head(50), use_container_width=True)

    # explanation for first month overview
    st.subheader("First month overview")
    first_month = df_weather.loc[df_weather.index.to_period('M') == df_weather.index.min().to_period('M')]
    value_cols = [c for c in df_weather.columns if c in
              ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]]
    rows = [{"Variable": col, "First Month": first_month[col].tolist()} for col in value_cols]
    chart_table = pd.DataFrame(rows)

    # explanation for rendering the first month trend as a line chart
    st.dataframe(
        chart_table,
        column_config={
            "Variable": st.column_config.TextColumn("Variable"),
            "First Month": st.column_config.LineChartColumn("First month trend"),
        },
        hide_index=True,
        use_container_width=True,
    )

# --- Page 3: Plots now uses ERA5 API and selected area ---
elif page == "Plots":
    st.title("Plots - ERA5 2021")
    area = st.session_state.area_code
    df_weather = download_era5_hourly_for_area(area, 2021)

    value_cols = [c for c in df_weather.columns if c in
              ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]]
    # explanation for selecting a variable to plot
    col_choice = st.selectbox("Choose variable", ["All"] + value_cols)

    # explanation for selecting a range of months to display
    months = pd.Index(df_weather.index.to_period('M').unique().astype(str))
    start_m, end_m = st.select_slider(
        "Select month range",
        options=months.tolist(),
        value=(months[0], months[0])   # default: first month
    )

    # explanation for filtering data based on the selected range
    mask = (df_weather.index.to_period('M') >= pd.Period(start_m)) & (df_weather.index.to_period('M') <= pd.Period(end_m))
    dff = df_weather.loc[mask]

    st.write(f"Showing data from {start_m} to {end_m} for area {area}.")

    # explanation for rendering the selected data as a line chart
    if col_choice == "All":
        dff_reset = dff.reset_index().melt("time", value_vars=value_cols, var_name="Variable", value_name="Value")
        chart = (
            alt.Chart(dff_reset)
            .mark_line()
            .encode(
                x=alt.X("time:T", title="Time"),
                y=alt.Y("Value:Q", title="Value"),
                color="Variable:N"
            )
            .properties(width="container", height=400, title="All Variables")
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        dff_reset = dff.reset_index()[["time", col_choice]]
        chart = (
            alt.Chart(dff_reset)
            .mark_line()
            .encode(
                x=alt.X("time:T", title="Time"),
                y=alt.Y(col_choice, title=col_choice)
            )
            .properties(width="container", height=400, title=col_choice)
        )
        st.altair_chart(chart, use_container_width=True)

# --- Page new B: Outlier or SPC and Anomaly or LOF on weather ---
elif page == "Outliers & Anomalies":
    st.title("Outliers and anomalies, ERA5 weather")

    area = st.session_state.area_code
    df_weather = download_era5_hourly_for_area(area, 2021)

    # helpers
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
    tabs = st.tabs(["Outlier or SPC", "Anomaly or LOF"])

    # Outlier or SPC tab
    with tabs[0]:
        st.subheader("SPC style outliers on hourly weather")

        with st.expander("Quick help", expanded=False):
            st.markdown(
                "- Pick a variable, then adjust High pass strength and Sigma threshold.\n"
                "- Baseline window defines a rolling mean used to draw the SPC bands."
            )

        available_vars = [c for c in df_weather.columns if c in
                  ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]]
        var = st.selectbox(
            "Variable",
            options=available_vars,
            index=available_vars.index("temperature_2m") if "temperature_2m" in available_vars else 0,
            help="Series to analyze for outliers"
        )

        #  explanation for SPC parameters
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

        # explanation for processing the selected variable
        series = pd.to_numeric(df_weather[var], errors="coerce")
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

            y_label = {
                "temperature_2m": "Celsius",
                "precipitation": "mm per hour",
                "wind_speed_10m": "m per s",
                "wind_gusts_10m": "m per s",
                "wind_direction_10m": "degrees"
            }.get(var, "value")

        # plot the SPC bands and mark outliers
        fig, ax = plt.subplots()
        ax.plot(series.index, series.values, label=var)
        ax.plot(b_lower.index, b_lower.values, linewidth=1, label="SPC lower")
        ax.plot(b_upper.index, b_upper.values, linewidth=1, label="SPC upper")
        ax.scatter(series.index[is_out], series.values[is_out], s=12, label="Outliers")
        ax.set_xlabel("Time"); ax.set_ylabel(y_label)
        ax.legend(loc="best")
        fig.tight_layout()
        st.pyplot(fig)

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


    # Anomaly or LOF tab
    with tabs[1]:
        st.subheader("Anomalies with Local Outlier Factor, precipitation")

        with st.expander("Quick help", expanded=False):
            st.markdown(
                "- Contamination is the expected share of anomalies.\n"
                "- Neighbors controls how local the detector is."
            )

        # explanation for LOF parameters
        contamination = st.slider(
            "Contamination, share of anomalies",
            min_value=0.001, max_value=0.05, value=0.01, step=0.001,
            help="Typical values 0.005 to 0.02"
        )
        n_neighbors = st.slider(
            "Neighbors",
            min_value=10, max_value=100, value=35, step=1,
            help="More neighbors gives a smoother detector"
        )

        # explanation for preparing data for LOF
        p = pd.to_numeric(df_weather["precipitation"], errors="coerce").fillna(0.0)
        roll3 = p.rolling(3, min_periods=1).sum()
        roll24 = p.rolling(24, min_periods=1).sum()
        X = np.c_[p.values, roll3.values, roll24.values]

        lof = LocalOutlierFactor(n_neighbors=int(n_neighbors), contamination=float(contamination))
        y_pred = lof.fit_predict(X)
        is_out = y_pred == -1

        # plot LOF anomalies
        fig, ax = plt.subplots()
        ax.plot(p.index, p.values, label="Precipitation mm")
        ax.scatter(p.index[is_out], p.values[is_out], s=12, label="Anomalies")
        ax.set_xlabel("Time"); ax.set_ylabel("mm per hour")
        ax.legend(loc="best")
        fig.tight_layout()
        st.pyplot(fig)

        # summary statistics table for LOF
        n_total = int(len(p))
        n_anom = int(is_out.sum())
        share = (100.0 * n_anom / n_total) if n_total else 0.0

        lof_summary = pd.DataFrame(
            {
                "metric": [
                    "points analyzed",
                    "anomalies detected",
                    "anomaly share, percent",
                    "contamination setting",
                    "neighbors used",
                    "precip min",
                    "precip max",
                    "precip mean",
                    "precip std",
                ],
                "value": [
                    n_total,
                    n_anom,
                    round(share, 3),
                    float(contamination),
                    int(n_neighbors),
                    round(float(np.nanmin(p.values)), 4),
                    round(float(np.nanmax(p.values)), 4),
                    round(float(np.nanmean(p.values)), 4),
                    round(float(np.nanstd(p.values)), 4),
                ],
            }
        )
        st.subheader("LOF summary")
        st.dataframe(lof_summary, use_container_width=True, hide_index=True)
