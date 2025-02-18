import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import xarray as xr
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from wrf import to_np

# Load OpenStreetMap tiles for basemap
request = cimgt.OSM()

# Define Cities and their Latitude/Longitude Extents
cities = {
    "Austin": [-98.0, -97.0, 30.0, 31.0],
    "Dallas": [-97.7, -96.2, 32.3, 33.4],
    "Houston": [-96.0, -94.5, 29.0, 30.2],
}

# Load Processed WRF Dataset
ds = xr.open_dataset('/scratch/08457/h_kamath/Austin_heat/Simulations_processed/Control.nc')
frac = ds.TSK  # Extract TSK (skin temperature)
lats = ds.lat
lons = ds.lon

# Define Simulation Days (August 13-16, 2020)
days = range(13, 17)

# Define MODIS datasets and corresponding time labels
modis_datasets = {
    "MOD_Day": ("MOD11A1.061_1km.nc", "LST_Day_1km", "T17:00:00"),
    "MOD_Night": ("MOD11A1.061_1km.nc", "LST_Night_1km", "T04:00:00"),
    "MYD_Day": ("MYD11A1.061_1km.nc", "LST_Day_1km", "T20:00:00"),
    "MYD_Night": ("MYD11A1.061_1km.nc", "LST_Night_1km", "T08:00:00"),
}

# Process each dataset for each city
for city, extent in cities.items():
    print(f"\nProcessing for {city}...")

    for dataset_name, (modis_file, modis_var, modis_time_suffix) in modis_datasets.items():
        print(f"  - Processing {dataset_name} dataset...")

        wrfall = np.zeros((len(days), 582, 561))
        modisall = np.zeros((len(days), 582, 561))

        for d, day in enumerate(days):
            day_str = f"{day:02d}"  # Ensure two-digit format
            modis_time = np.datetime64(f"2020-08-{day_str}{modis_time_suffix}")

            # Load MODIS data
            ncfile = xr.open_dataset(f"/scratch/08457/h_kamath/Austin_heat/Simulations_processed/Observations/MODIS/MODIS/{modis_file}")
            modis = ncfile[modis_var].isel(time=day - 6).values
            ncfile.close()

            # Select WRF TSK data at the nearest time
            frachr = frac.sel(time=modis_time, method="nearest")

            # Load Land Cover Data (LU_INDEX)
            landcover_nc = xr.open_dataset("/scratch/08457/h_kamath/Austin_heat/New_with_trees_USE_THIS/Control/wrfout_d03_2020-08-15_18:00:00")
            landcover = landcover_nc['LU_INDEX'].isel(Time=0).values
            landcover_nc.close()

            # Mask invalid MODIS data
            modis[np.isnan(modis)] = 9999.0
            aa = ma.masked_where(modis >= 999.0, frachr.values)
            modis = ma.masked_where(modis >= 999.0, modis)

            # Apply land cover masking
            modis = ma.masked_where(landcover <= 12.0, modis)
            aa = ma.masked_where(landcover <= 12.0, aa)
            for la in range(14, 50):
                modis = ma.masked_where(landcover == la, modis)
                aa = ma.masked_where(landcover == la, aa)

            # Generate and Save Plots
            fig, axes = plt.subplots(1, 3, figsize=(12, 6), subplot_kw={'projection': request.crs})
            titles = ['(a) Model', '(b) MODIS', '(c) Model - MODIS']
            vmin, vmax = 300, 320

            for nn, ax in enumerate(axes):
                ax.set_extent(extent)

                if nn == 0:
                    data = aa
                    cmap = "jet"
                elif nn == 1:
                    data = modis
                    cmap = "jet"
                else:
                    data = aa - modis
                    cmap = "bwr"
                    vmin, vmax = -7.5, 7.5

                im = ax.pcolor(to_np(lons[10:-10, 10:-10]), to_np(lats[10:-10, 10:-10]), to_np(data[10:-10, 10:-10]),
                               transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, alpha=0.5, cmap=cmap)

                cbar = plt.colorbar(im, ax=ax, shrink=0.68)
                cbar.set_label('Temperature (K)' if nn < 2 else 'Difference (K)')
                ax.set_title(titles[nn], loc='left')

            # Save figure
            plot_filename = f"{city}_{dataset_name}_08{day_str}2020.pdf"
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)

            # Store data
            wrfall[d, :, :] = aa
            modisall[d, :, :] = modis

        # Save Data to NetCDF
        output_file = f"{city}_{dataset_name}.nc"
        ds_out = xr.Dataset(
            {"modis": (("time", "lat", "lon"), modisall),
             "mod": (("time", "lat", "lon"), wrfall)},
            coords={"time": list(days), "lat": lats, "lon": lons}
        )
        ds_out.to_netcdf(output_file)

print("\nAll processing complete")
