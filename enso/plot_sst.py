import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import xarray as xr
import pandas as pd
import cartopy as ctp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.mpl.geoaxes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np


sodas_path = "enso/H19_dataset/SODA/SODA.input.36mn.1871_1970.nc"
sodas_path_l = "enso/H19_dataset/SODA/SODA.label.nino34.12mn_2mv.1873_1972.nc"
dt_sodas = xr.open_dataset("enso/H19_dataset/SODA/SODA.input.36mn.1871_1970.nc")

def plot_anomalies(data_sst,lev):
    dmap = data_sst["sst"][0].sel(lev=lev)
    proj = ccrs.Mollweide(central_longitude=0)

    # create figure
    fig, ax = plt.subplots(figsize=(9,6))
    ax = plt.axes(projection=proj)

    # axes properties
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)    
    ax.add_feature(ctp.feature.RIVERS)
    ax.add_feature(ctp.feature.BORDERS, linestyle=':')

    # set colormap
    cmap = plt.get_cmap("RdBu_r")

    # plot map
    im = ax.pcolormesh(
    dmap.coords['lon'], dmap.coords['lat'], dmap.data,
    cmap=cmap, 
    transform=ctp.crs.PlateCarree(central_longitude=0)
    )

    plt.show()


plot_anomalies(dt_sodas,lev=1)
plot_anomalies(dt_sodas,lev=2)
plot_anomalies(dt_sodas,lev=3)
plot_anomalies(dt_sodas,lev=4)
plot_anomalies(dt_sodas,lev=5)
plot_anomalies(dt_sodas,lev=6)
plot_anomalies(dt_sodas,lev=7)
plot_anomalies(dt_sodas,lev=8)
