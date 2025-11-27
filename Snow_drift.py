#!/usr/bin/env python3
"""
Complete script for calculating annual snow drifting using Tabler (2003)
and visualizing the average directional contributions in a 16-sector wind rose.

Assumptions:
 - Hourly meteorological input is stored in a CSV file 
   (open-meteo-60.57N7.60E1212m.csv).
 - The CSV contains two header sections: metadata in the first few rows and the
   actual data header starting on the fourth row.
 - Hourly temperature, precipitation, wind speed at 10 m, and wind direction at 10 m are provided.
 - Hourly Swe is defined as the precipitation when the temperature is below +1°C.
 - Snow drifting calculations follow Tabler (2003):
     1. Qupot (potential wind-driven transport): summed hourly contributions using u^3.8.
     2. Qspot (snowfall-limited transport): 0.5 * T * Swe.
     3. Srwe (relocated water equivalent): θ * Swe.
     4. If Qupot > Qspot then snowfall controls:
          Qinf = 0.5 * T * Srwe,
        otherwise Qinf = Qupot.
     5. Mean annual snow transport: Qt = Qinf * (1 - 0.14 ** (F/T)).
 - The meteorological data is treated seasonally. In this script the season starts on July 1
   and runs for 12 months (until June 30 of the following year).
 - The rose plot displays the average yearly directional breakdown, and the overall average
   yearly snow transport is shown in tonnes/m (one decimal).
 - The script also computes the necessary fence height for storing the drift.
   For a given fence type, the required height is computed as:
       H = ( (Qt_tonnes) / (Qc/H^2.2) )^(1/2.2)
   where the storage capacity factor (Qc/H^2.2) is taken from Table 3.3:
       - Wyoming: 8.5
       - Slat-and-wire: 7.7
       - Solid: 2.9
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_Qupot(hourly_wind_speeds, dt=3600):
    """
    Compute the potential wind-driven snow transport (Qupot) [kg/m]
    by summing hourly contributions using u^3.8.
    
    Formula:
       Qupot = sum((u^3.8) * dt) / 233847
    """
    total = sum((u ** 3.8) * dt for u in hourly_wind_speeds) / 233847
    return total

def sector_index(direction):
    """
    Given a wind direction in degrees, returns the index (0-15)
    corresponding to a 16-sector division.
    """
    # Center the bin by adding 11.25° then modulo 360 and divide by 22.5°
    return int(((direction + 11.25) % 360) // 22.5)

def compute_sector_transport(hourly_wind_speeds, hourly_wind_dirs, dt=3600):
    """
    Compute the cumulative transport for each of 16 wind sectors.
    
    Parameters:
      hourly_wind_speeds: list of wind speeds [m/s]
      hourly_wind_dirs: list of wind directions [degrees]
      dt: time step in seconds
      
    Returns:
      A list of 16 transport values (kg/m) corresponding to the sectors.
    """
    sectors = [0.0] * 16
    for u, d in zip(hourly_wind_speeds, hourly_wind_dirs):
        idx = sector_index(d)
        sectors[idx] += ((u ** 3.8) * dt) / 233847
    return sectors

def compute_snow_transport(T, F, theta, Swe, hourly_wind_speeds, dt=3600):
    """
    Compute various components of the snow drifting transport according to Tabler (2003).
    
    Parameters:
      T: Maximum transport distance (m)
      F: Fetch distance (m)
      theta: Relocation coefficient
      Swe: Total snowfall water equivalent (mm)
      hourly_wind_speeds: list of wind speeds [m/s]
      dt: time step in seconds
      
    Returns:
      A dictionary containing:
         Qupot (kg/m): Potential wind-driven transport.
         Qspot (kg/m): Snowfall-limited transport.
         Srwe (mm): Relocated water equivalent.
         Qinf (kg/m): The controlling transport value.
         Qt (kg/m): Mean annual snow transport.
         Control: Process controlling the transport (wind or snowfall).
    """
    Qupot = compute_Qupot(hourly_wind_speeds, dt)
    Qspot = 0.5 * T * Swe  # Snowfall-limited transport [kg/m]
    Srwe = theta * Swe    # Relocated water equivalent [mm]
    
    if Qupot > Qspot:
        Qinf = 0.5 * T * Srwe
        control = "Snowfall controlled"
    else:
        Qinf = Qupot
        control = "Wind controlled"
    
    Qt = Qinf * (1 - 0.14 ** (F / T))
    
    return {
        "Qupot (kg/m)": Qupot,
        "Qspot (kg/m)": Qspot,
        "Srwe (mm)": Srwe,
        "Qinf (kg/m)": Qinf,
        "Qt (kg/m)": Qt,
        "Control": control
    }

def compute_yearly_results(df, T, F, theta):
    """
    Compute the yearly (seasonal) snow transport parameters for every season in the data.
    The season is defined as July 1 of a given year to June 30 of the next year.
    
    Returns a DataFrame with one row per season.
    """
    seasons = sorted(df['season'].unique())
    results_list = []
    for s in seasons:
        season_start = pd.Timestamp(year=s, month=7, day=1)
        season_end = pd.Timestamp(year=s+1, month=6, day=30, hour=23, minute=59, second=59)
        df_season = df[(df['time'] >= season_start) & (df['time'] <= season_end)]
        if df_season.empty:
            continue
        # Calculate hourly Swe: precipitation counts when temperature < +1°C.
        df_season = df_season.copy()  # avoid SettingWithCopyWarning
        df_season['Swe_hourly'] = df_season.apply(
            lambda row: row['precipitation (mm)'] if row['temperature_2m (°C)'] < 1 else 0, axis=1)
        total_Swe = df_season['Swe_hourly'].sum()
        wind_speeds = df_season["wind_speed_10m (m/s)"].tolist()
        result = compute_snow_transport(T, F, theta, total_Swe, wind_speeds)
        result["season"] = f"{s}-{s+1}"
        results_list.append(result)
    return pd.DataFrame(results_list)

def compute_average_sector(df):
    """
    Compute the average directional breakdown (sectors) over all seasons.
    The function groups the data by season and computes the sector contributions
    for each season, then returns the mean across seasons.
    """
    sectors_list = []
    for s, group in df.groupby('season'):
        group = group.copy()
        group['Swe_hourly'] = group.apply(
            lambda row: row['precipitation (mm)'] if row['temperature_2m (°C)'] < 1 else 0, axis=1)
        ws = group["wind_speed_10m (m/s)"].tolist()
        wdir = group["wind_direction_10m (°)"].tolist()
        sectors = compute_sector_transport(ws, wdir)
        sectors_list.append(sectors)
    avg_sectors = np.mean(sectors_list, axis=0)
    return avg_sectors

def plot_rose(avg_sector_values, overall_avg):
    """
    Create a canvas with a polar (wind rose) plot showing the average directional breakdown.
    
    Parameters:
      avg_sector_values: list of 16 average transport values (kg/m) for the sectors.
      overall_avg: overall average yearly snow transport (Qt in kg/m) across all seasons.
                   This value will be converted to tonnes/m.
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    num_sectors = 16
    # Compute bin centers: each bin is 360/16 = 22.5° wide
    angles = np.deg2rad(np.arange(0, 360, 360/num_sectors))
    
    # Convert the sector values from kg/m to tonnes/m
    avg_sector_values_tonnes = np.array(avg_sector_values) / 1000.0

    ax.bar(angles, avg_sector_values_tonnes, width=np.deg2rad(360/num_sectors),
           align='center', edgecolor='black')
    
    # Ensure north is at the top and the direction is clockwise.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    # Set custom tick labels for each sector.
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    ax.set_xticks(angles)
    ax.set_xticklabels(directions)
    
    # Convert overall average from kg/m to tonnes/m and format with one decimal.
    overall_tonnes = overall_avg / 1000.0
    ax.set_title(
        f"Average Directional Distribution of Snow Transport\nOverall Average Qt: {overall_tonnes:,.1f} tonnes/m",
        va='bottom'
    )
    plt.tight_layout()
    plt.show()

def compute_fence_height(Qt, fence_type):
    """
    Calculate the necessary effective fence height (H) for storing a given snow drift.
    
    Parameters:
      Qt : float
           The calculated mean annual snow transport (drift) in kg/m.
      fence_type : str
           The fence type. Supported types are:
           "Wyoming", "Slat-and-wire", and "Solid".
    
    Returns:
      H : float
          The necessary effective fence height (in meters).
    
    Calculation:
      1. Convert Qt from kg/m to tonnes/m (divide by 1000).
      2. Use the storage capacity factor for the selected fence type:
             - Wyoming: 8.5
             - Slat-and-wire: 7.7
             - Solid: 2.9
      3. Calculate H = ( (Qt_tonnes) / (factor) )^(1/2.2)
    """
    Qt_tonnes = Qt / 1000.0
    if fence_type.lower() == "wyoming":
        factor = 8.5
    elif fence_type.lower() in ["slat-and-wire", "slat and wire"]:
        factor = 7.7
    elif fence_type.lower() == "solid":
        factor = 2.9
    else:
        raise ValueError("Unsupported fence type. Choose 'Wyoming', 'Slat-and-wire', or 'Solid'.")
    
    H = (Qt_tonnes / factor) ** (1 / 2.2)
    return H

def main():
    # Read the CSV file (skip metadata rows so that the header is read correctly).
    filename = "open-meteo-60.57N7.60E1212m.csv"
    df = pd.read_csv(filename, skiprows=3)
    
    # Convert the 'time' column to datetime.
    df['time'] = pd.to_datetime(df['time'])
    
    # Define season: if month >= 7, season = current year; otherwise, season = previous year.
    df['season'] = df['time'].apply(lambda dt: dt.year if dt.month >= 7 else dt.year - 1)
    
    # Parameters for the snow transport calculation.
    T = 3000      # Maximum transport distance in meters
    F = 30000     # Fetch distance in meters
    theta = 0.5   # Relocation coefficient
    
    # Compute seasonal results (yearly averages for each season).
    yearly_df = compute_yearly_results(df, T, F, theta)
    overall_avg = yearly_df['Qt (kg/m)'].mean()
    print("\nYearly average snow drift (Qt) per season:")
    print(f"Overall average Qt over all seasons: {overall_avg / 1000:.1f} tonnes/m")
    
    yearly_df_disp = yearly_df.copy()
    yearly_df_disp["Qt (tonnes/m)"] = yearly_df_disp["Qt (kg/m)"] / 1000
    print("\nYearly average snow drift (Qt) per season (in tonnes/m) and control type:")
    print(yearly_df_disp[['season', 'Qt (tonnes/m)', 'Control']].to_string(index=False, 
          formatters={'Qt (tonnes/m)': lambda x: f"{x:.1f}"}))
    
    overall_avg_tonnes = overall_avg / 1000
    print(f"\nOverall average Qt over all seasons: {overall_avg_tonnes:.1f} tonnes/m")
    
    # Compute the average directional breakdown (average over all seasons).
    avg_sectors = compute_average_sector(df)
    
    # Create the rose plot canvas with the average directional breakdown.
    plot_rose(avg_sectors, overall_avg)
    
    # Compute and print necessary fence heights for each season and for three fence types.
    fence_types = ["Wyoming", "Slat-and-wire", "Solid"]
    fence_results = []
    for idx, row in yearly_df.iterrows():
        season = row["season"]
        Qt_val = row["Qt (kg/m)"]
        res = {"season": season}
        for ft in fence_types:
            res[f"{ft} (m)"] = compute_fence_height(Qt_val, ft)
        fence_results.append(res)
    fence_df = pd.DataFrame(fence_results)
    print("\nNecessary fence heights per season (in meters):")
    print(fence_df.to_string(index=False, formatters={
        "Wyoming (m)": lambda x: f"{x:.1f}",
        "Slat-and-wire (m)": lambda x: f"{x:.1f}",
        "Solid (m)": lambda x: f"{x:.1f}"
    }))

if __name__ == "__main__":
    main()
