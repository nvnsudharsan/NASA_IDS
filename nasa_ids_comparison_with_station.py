import pandas as pd
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
from glob import glob
import pytz
from scipy.spatial import cKDTree
import matplotlib.dates as mdates

# Define NetCDF file path
nc_file_path = "/Users/geo-ns36752/Downloads/Control (1).nc"

# Open WRF dataset
ds = xr.open_dataset(nc_file_path)
t2 = ds.T2

# Convert WRF time to pandas datetime index and local time
wrf_time = pd.to_datetime(t2.time)
wrf_time = pd.DatetimeIndex([t.round('h') for t in wrf_time])
local_tz = pytz.timezone('America/Chicago')
wrf_time = wrf_time.tz_localize('UTC').tz_convert(local_tz)

# Extract latitude and longitude
lat = t2.lat
lon = t2.lon

# Gather station CSV files
Austin_csv = glob('/Users/geo-ns36752/Downloads/ISD/Austin/*.csv')
Houston_csv = glob('/Users/geo-ns36752/Downloads/ISD/Houston/*.csv')
Dallas_csv = glob('/Users/geo-ns36752/Downloads/ISD/Dallas/*.csv')
SanAntonio_csv = glob('/Users/geo-ns36752/Downloads/ISD/San_Antonio/*.csv')

all_csv = Austin_csv + Houston_csv + Dallas_csv + SanAntonio_csv

# Initialize lists for RMSE and MB calculations
RMSE_df = []
MB_df = []

# DataFrames to store WRF and station data
all_sta_t2 = pd.DataFrame()
all_wrf_t2 = pd.DataFrame()

# Flatten lat/lon arrays and create KDTree for nearest neighbor search
lat_vals = t2['lat'].values.ravel()
lon_vals = t2['lon'].values.ravel()
coords = np.vstack((lat_vals, lon_vals)).T  # Shape: (n_points, 2)
tree = cKDTree(coords)

for i, csv_file in enumerate(all_csv):
    try:
        # Read station data
        station_df = pd.read_csv(csv_file, usecols=range(10))
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        continue
    station_name = station_df['NAME'][0]
    station_df['DATE'] = pd.to_datetime(station_df['DATE'], errors='coerce')
    station_df = station_df.dropna(subset=['DATE'])
    station_df['DATE'] = station_df['DATE'].dt.tz_localize('UTC').dt.tz_convert('America/Chicago')
    station_lat, station_lon = station_df['LAT'][0], station_df['LON'][0]
    station_df = station_df[['DATE', 'TMP']]
    station_df.loc[station_df['TMP'] == 9999, 'TMP'] = np.nan
    station_df['TMP'] = station_df['TMP'] / 10.0
    station_df = station_df.set_index('DATE').resample('H').mean().reset_index()
    station_df = station_df.sort_values('DATE')
    wrf_time = wrf_time.sort_values()
    station_df = pd.merge_asof(wrf_time.to_frame(name="DATE"), station_df, on="DATE", direction="nearest")
    station_df['TMP'] = pd.to_numeric(station_df['TMP'], errors='coerce')
    dist, idx = tree.query([station_lat, station_lon])
    lat_idx, lon_idx = np.unravel_index(idx, t2['lat'].shape)
    if dist > 0.5:
        print(f"Warning: {csv_file} station is far from the closest WRF grid ({dist:.2f} degrees).")
    wrf_station_t2 = t2[:, lat_idx, lon_idx] - 273.15  
    if i == 0:
        all_sta_t2['DATE'] = station_df['DATE']
        all_wrf_t2['DATE'] = station_df['DATE']
    all_sta_t2[station_name] = station_df['TMP']
    all_wrf_t2[station_name] = wrf_station_t2
    if len(station_df['TMP']) == len(wrf_station_t2):
        RMSE = np.sqrt(((station_df['TMP'] - wrf_station_t2) ** 2).mean())
        MB = (wrf_station_t2 - station_df['TMP'].to_numpy()).mean()
        RMSE_df.append(RMSE)
        MB_df.append(MB)
    else:
        print(f"Skipping RMSE calculation for {station_name} due to shape mismatch.")

# Compute Mean and Standard Deviation
mean_station_temp = all_sta_t2.iloc[:, 1:].mean(axis=1)
std_station_temp = all_sta_t2.iloc[:, 1:].std(axis=1)
mean_wrf_temp = all_wrf_t2.iloc[:, 1:].mean(axis=1)
std_wrf_temp = all_wrf_t2.iloc[:, 1:].std(axis=1)
# Compute Mean RMSE and MAB
mean_rmse = np.mean(RMSE_df)
mean_mab = np.mean(np.abs(MB_df))


# Ensure wrf_time is in America/Chicago timezone
local_tz = pytz.timezone('America/Chicago')
wrf_time_local = wrf_time.tz_convert(local_tz)  # Ensure it's in CDT

# Convert timezone-aware datetime to naive (CDT values remain correct)
wrf_time_naive = wrf_time_local.tz_localize(None)

# Create a figure
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot mean temperature with standard deviation shading
ax1.plot(wrf_time_naive, mean_station_temp, color='black', label='Station')
ax1.fill_between(wrf_time_naive, mean_station_temp - std_station_temp, mean_station_temp + std_station_temp, color='black', alpha=0.2)
ax1.plot(wrf_time_naive, mean_wrf_temp, color='red', label='WRF')
ax1.fill_between(wrf_time_naive, mean_wrf_temp - std_wrf_temp, mean_wrf_temp + std_wrf_temp, color='red', alpha=0.2)

ax1.set_title('')
ax1.set_ylabel('Temperature (°C)')
ax1.set_xlim(wrf_time_naive[0], wrf_time_naive[-1])
ax1.legend(loc='upper left')

# Set major and minor ticks
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

# Force x-axis labels to show CDT by manually formatting them
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M CDT'))

plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Print Mean RMSE and MAB on the figure
ax1.text(wrf_time_naive[1], ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.05, 
         f'RMSE: {mean_rmse:.2f}°C\nMAB: {mean_mab:.2f}°C', fontsize=12, color='blue', 
         verticalalignment='bottom', horizontalalignment='left')

plt.tight_layout()
plt.savefig('/Users/geo-ns36752/Downloads/ISD/combined.pdf', dpi=300)
plt.show()
