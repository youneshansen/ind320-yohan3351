import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt

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
page = st.sidebar.radio("Go to", ["Home", "Data Table", "Plots", "About"])

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


# --- Page 4: About ---
elif page == "About":
    st.title("About")
    st.markdown("""
    **Project:** IND320 Compulsory Work (Part 1)

    **Features:**
    - Four-page app with sidebar navigation  
    - Page 2: Table with row-wise line charts (first month per variable)  
    - Page 3: Plot with variable select and month range slider  
    - Data loaded from local CSV with caching  
    """)
