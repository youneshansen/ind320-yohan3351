import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt
import os


def _mongo_settings():
    # prefer Streamlit secrets, fall back to env vars for local testing
    uri = st.secrets.get("mongo", {}).get("uri") if hasattr(st, "secrets") else None
    db = st.secrets.get("mongo", {}).get("db") if hasattr(st, "secrets") else None
    coll = st.secrets.get("mongo", {}).get("coll") if hasattr(st, "secrets") else None
    uri = uri or os.getenv("MONGODB_URI", "")
    db = db or os.getenv("MONGODB_DB", "power")
    coll = coll or os.getenv("MONGODB_COLL", "production_2021")
    return uri, db, coll

@st.cache_resource
def _mongo_client(uri: str):
    from pymongo import MongoClient
    return MongoClient(uri)

@st.cache_data(ttl=300)
def get_price_areas(uri: str, db: str, coll: str):
    cli = _mongo_client(uri)
    return sorted(cli[db][coll].distinct("priceArea"))

@st.cache_data(ttl=300)
def get_groups_for_area(uri: str, db: str, coll: str, price_area: str):
    cli = _mongo_client(uri)
    return sorted(cli[db][coll].distinct("productionGroup", {"priceArea": price_area}))

@st.cache_data(ttl=300)
def get_year_totals(uri: str, db: str, coll: str, price_area: str) -> pd.DataFrame:
    cli = _mongo_client(uri)
    pipe = [
        {"$match": {
            "priceArea": price_area,
            "startTime": {"$gte": "2021-01-01T00:00:00", "$lt": "2022-01-01T00:00:00"}
        }},
        {"$group": {"_id": "$productionGroup", "totalKwh": {"$sum": "$quantityKwh"}}},
        {"$sort": {"totalKwh": -1}}
    ]
    rows = list(cli[db][coll].aggregate(pipe, allowDiskUse=True))
    df = pd.DataFrame(rows, columns=["_id", "totalKwh"]).rename(columns={"_id": "productionGroup"})
    df["totalKwh"] = pd.to_numeric(df["totalKwh"], errors="coerce").fillna(0.0)
    return df[df["totalKwh"] > 0]

@st.cache_data(ttl=300)
def get_month_series(uri: str, db: str, coll: str, price_area: str, year: int, month: int, groups: list[str]) -> pd.DataFrame:
    cli = _mongo_client(uri)
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1)
    match = {"priceArea": price_area, "startTime": {"$gte": start.isoformat(), "$lt": end.isoformat()}}
    if groups:
        match["productionGroup"] = {"$in": groups}
    pipe = [
        {"$match": match},
        {"$project": {"_id": 0, "productionGroup": 1, "startTime": 1, "quantityKwh": 1}}
    ]
    rows = list(cli[db][coll].aggregate(pipe, allowDiskUse=True))
    if not rows:
        return pd.DataFrame(columns=["startTime", "productionGroup", "quantityKwh"])
    df = pd.DataFrame(rows)
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df["quantityKwh"] = pd.to_numeric(df["quantityKwh"], errors="coerce").fillna(0.0)
    return df.sort_values("startTime")


# --- Streamlit app setup ---
st.set_page_config(page_title="IND320 Project", layout="wide")

# --- Load CSV with caching ---
@st.cache_data
def load_data():
    path = Path("open-meteo-subset.csv")
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    return df

df = load_data()

# --- Variable names ---
value_cols = [
    'temperature_2m (°C)',
    'precipitation (mm)',
    'wind_speed_10m (m/s)',
    'wind_gusts_10m (m/s)',
    'wind_direction_10m (°)'
]

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Table", "Plots", "Elhub"])


# --- Page 1: Home ---
if page == "Home":
    st.title("IND320 Project – Weather Data")
    st.write("This Streamlit app is part of the compulsory project for IND320.")
    st.write("Use the sidebar to navigate to the Data Table, Plots, or About page.")

# --- Page 2: Data Table ---
elif page == "Data Table":
    st.title("Data Table")
    st.write("Preview of the dataset (first 50 rows):")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("First month overview (row-wise line charts)")
    # Get first month of data
    first_month = df.loc[df.index.to_period('M') == df.index.min().to_period('M')]

    # Build a table with one row per variable
    rows = []
    for col in value_cols:
        rows.append({
            "Variable": col,
            "First Month": first_month[col].tolist()
        })
    chart_table = pd.DataFrame(rows)

    st.dataframe(
        chart_table,
        column_config={
            "Variable": st.column_config.TextColumn("Variable"),
            "First Month": st.column_config.LineChartColumn("First month trend"),
        },
        hide_index=True,
        use_container_width=True,
    )

# --- Page 3: Plots ---
elif page == "Plots":

    st.title("Plots")

    # Dropdown to choose column
    col_choice = st.selectbox("Choose variable", ["All"] + value_cols)

    # Slider to select month range
    months = pd.Index(df.index.to_period('M').unique().astype(str))
    start_m, end_m = st.select_slider(
        "Select month range",
        options=months.tolist(),
        value=(months[0], months[0])   # default: first month
    )

    # Filter data by selected months
    mask = (df.index.to_period('M') >= pd.Period(start_m)) & (df.index.to_period('M') <= pd.Period(end_m))
    dff = df.loc[mask]

    st.write(f"Showing data from **{start_m}** to **{end_m}**.")

    # Plot with Altair for axis labels
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


# --- Page 4: Elhub ---
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
        area = st.radio("Price area", options=areas, index=0, horizontal=True)

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

    # right side - pills or multiselect for groups, month selector, line plot for that month
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

