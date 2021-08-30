"""Convert SODA and GODA-data in a way 
that it can be processed by the implemented model"""
# %%
import os
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy as ctp

ds = xr.open_dataset("/Users/andreasbaumgartner/Documents/Enso/enso2/enso3/enso_cnn/enso/H19_dataset/GODAS/GODAS.input.36mn.1980_2015.nc")
label = xr.open_dataset("/Users/andreasbaumgartner/Documents/Enso/enso2/enso3/enso_cnn/enso/H19_dataset/GODAS/GODAS.label.12mn_2mv.1982_2017.nc")

# %%
# Sort input data
years = np.array(ds.time.data, dtype='datetime64[Y]')
for idx_year, y in enumerate(years):
    time_tmp = np.arange(f"{y}-01-01" , f"{y+1}-01-01", dtype='datetime64[M]')
    sst_tmp = ds['sst'].isel(time=idx_year, lev=slice(24,36)).data
    t300_tmp = ds['t300'].isel(time=idx_year, lev=slice(24,36)).data

    if idx_year == 0:
        sst = sst_tmp
        t300 = t300_tmp
        time = time_tmp
        control = sst_tmp
    else:
        if control[0, 0, 0] != ds['sst'].isel(time=idx_year, lev=12, lat=0, lon=0).data:
            raise ValueError(f"Mismatch in ordering at year {y}")

        time = np.append(time, time_tmp, axis=0)
        sst = np.append(sst, sst_tmp, axis=0)
        t300 = np.append(t300, t300_tmp, axis=0)
        control = sst_tmp
    
# %%

dims = ['time', 'lat', 'lon']
coords = {
    'time': np.array(time, dtype="datetime64[D]"),
    'lat': ds['lat'][:].data, 
    'lon': ds['lon'][:].data
}
ds_new = xr.Dataset(
    data_vars={
        'ssta': (dims, np.array(sst)),
        't300a': (dims, np.array(t300)),
    },
    coords=coords,
    attrs=dict(
        description="GODAS of sst anomalies and t300 anomalies",
        url="https://drive.google.com/file/d/1Ht6__G4bFWguZTJ3nKc3XuEY1KKLMIIN/view"
    )
)

ds_new.to_netcdf("/Users/andreasbaumgartner/Documents/Enso/enso2/enso3/enso_cnn/enso/H19_dataset/GODAS/converted.GODAS.input.nc", "w")


# %%
# Sort labels
years = np.array(label.time.data, dtype='datetime64[Y]')
for idx_year, y in enumerate(years):
    time_tmp = np.arange(f"{y}-01-01" , f"{y+1}-01-01", dtype='datetime64[M]')
    nino_tmp = label['pr'].isel(time=idx_year, lat=0, lon=0).data

    if idx_year == 0:
        nino = nino_tmp
        time = time_tmp
    else:
        time = np.append(time, time_tmp, axis=0)
        nino = np.append(nino, nino_tmp, axis=0)
    

# %%
dims = ['time']
coords = {
    'time': np.array(time, dtype="datetime64[D]"),
}
ds_label = xr.Dataset(
    data_vars={
        'nino34': (dims, np.array(nino))
    },
    coords=coords,
    attrs=dict(
        description="Nino3.4 index of SODA models",
        url="https://drive.google.com/file/d/1Ht6__G4bFWguZTJ3nKc3XuEY1KKLMIIN/view"
    )
)
ds_label.to_netcdf("/Users/andreasbaumgartner/Documents/Enso/enso2/enso3/enso_cnn/enso/H19_dataset/GODAS/converted.GODAS.label.12mn_2mv.1982_2017.nc")

# %%

