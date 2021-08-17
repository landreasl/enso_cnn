# %%
from sst_data import SSTAData
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import xarray as xr
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from xarray.core.common import C

# TODO nn.sequential, torch.save, torch.load,
# time labeling mit dictionary damit shufflebar, netzwerk kernelsize, cmip,
# output tatächliche index mit vorhergesagt
# %%
"""dt = xr.open_dataset("H19_dataset/CMIP5/lon_CMIP5_input_sort_1861_2001.nc")
label_cmip = xr.open_dataset("H19_dataset/CMIP5/CMIP5_label_nino34_sort_1861_2001.nc")

# Open Datasets
dt_sodas = xr.open_dataset("H19_dataset/SODA/SODA.input.36mn.1871_1970.nc")
label_sodas = xr.open_dataset("H19_dataset/SODA/SODA.label.nino34.12mn_2mv.1873_1972.nc")

dt_sodas
# %%
dt_sst = dt["ssta"].sel(
                    time=slice(dt.time[24],dt.time[-1]),
                    model=lev
                    )

dt_nino = label_cmip["nino34"].sel(
                        time=slice(label_cmip.time[0], label_cmip.time[-25]),
                        model=lev)



len(dt_nino)
"""
# %%
# TODO cmip 3 jahre shift, dataset abfragen (erben) 
class SstDataset(Dataset):
    def __init__(
        self, file, label_file, long_range=[-180, 180], lat_range=[-60, 60], 
        lead_time=3, hist_time=3):
        super().__init__()

        with xr.open_dataset(file) as dt:
            dt = dt.sel(
                lon=slice(long_range[0], long_range[1]),
                lat=slice(lat_range[0], lat_range[1])
            )
            #TODO: ssta für andere datensätze konsistent 
            self.lon = dt["lon"]
            self.lat = dt["lat"]

            self.time = dt["time"][2:]  # Data is shifted by 2 years with respect to nino labels
            self.sst = dt["sst"][2:].sel(
                lev=slice(24,37), # Choose time between january and december 
            )  
            self.t300 = dt["t300"][2:].sel(
                lev=slice(24,37), # Choose time between january and december 
            )
            self.n_samples = dt.time.size
            with xr.open_dataset(label_file) as dt:
                self.nino34 = dt["pr"][:-2]
        
        self.hist_time = hist_time
        self.lead_time = lead_time


    def __getitem__(self, index):
        # To avoid an index out of bounce error, these if statements were implemented. The first and last datapoints
        # are thereby considered multiple times
        if index < self.hist_time:
            index = self.hist_time
            label_buff = self.nino34[(index+self.lead_time)]
        
        if (index+self.lead_time) >= len(self.nino34):
            index = len(self.nino34) - self.lead_time - 1
            label_buff = self.nino34[len(self.nino34)-1]

        else:
            label_buff = self.nino34[(index+self.lead_time)]
        

        data_point = np.zeros((self.hist_time * 2, len(self.lat), len(self.lon)))
        data_point[:self.hist_time,:,:] = self.sst[index-self.hist_time:index].data
        data_point[self.hist_time:,:,:] = self.t300[index-self.hist_time:index].data
        data_point = torch.from_numpy(data_point)

        label = torch.from_numpy(label_buff.data)
        assert self.sst[index].time == self.nino34[index].time
        return data_point, label

    def cut_map_area(ds, lon_range, lat_range):
        """Cut an area in the map."""
        ds_area = ds.sel(
            lon=slice(lon_range[0], lon_range[1]),
            lat=slice(lat_range[0], lat_range[1])
        )
        return ds_area

    def __len__(self):
        return self.time.size

    def time2timestamp(self, time):
        """Convert np.datetime64 to int."""
        return (time - np.datetime64('1950-01-01', 'ns')) / np.timedelta64(1, 'D') 

    def timestamp2time(self, timestamp):
        """Convert timestamp to np.datetime64 object."""
        return (np.datetime64('1950-01-01', 'ns')
                + timestamp * np.timedelta64(1, 'D') )
    
# %%
