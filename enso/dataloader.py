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
import pandas as pd

# TODO nn.sequential, torch.save, torch.load,
# time labeling mit dictionary damit shufflebar, netzwerk kernelsize, cmip,
# output tatächliche index mit vorhergesagt

class SstDataset(Dataset):
    def __init__(
        self, file, label_file, long_range=[-180, 180], lat_range=[-60, 60], 
        lead_time=3, hist_time=3, lev=1, is_cmip=True):
        super().__init__()

        with xr.open_dataset(file) as dt:
            dt = dt.sel(
                lon=slice(long_range[0], long_range[1]),
                lat=slice(lat_range[0], lat_range[1])
            )
            #TODO: ssta für andere datensätze konsistent 
            self.lon = dt["lon"]
            self.lat = dt["lat"]
            self.time = dt["time"][24:]  # Index 24: because data in cmip is shifted by 2 years in respect to nino labels

            if is_cmip == True:
                self.sst = dt["ssta"].sel(
                    time=slice(dt.time[24],dt.time[-1]),
                    model=lev
                    ) # für ander daten lev = lev
                self.t300 = dt["t300a"].sel(
                    time=slice(dt.time[24],dt.time[-1]),
                    model=lev)
                self.n_samples = dt.time.size
                with xr.open_dataset(label_file) as dt_label:
                    self.nino34 = dt_label["nino34"].sel(
                        time=slice(dt_label.time[0], dt_label.time[-25]),
                        model=lev
                        )   
            
            else: 
                self.sst = dt["ssta"].sel(
                    time=slice(dt.time[24],dt.time[-1]),
                    ) 
                self.t300 = dt["t300a"].sel(
                    time=slice(dt.time[24],dt.time[-1]),
                    )
                self.n_samples = dt.time.size
                with xr.open_dataset(label_file) as dt_label:
                    self.nino34 = dt_label["nino34"].sel(
                        time=slice(dt_label.time[0], dt_label.time[-25]),
                        )   


        self.hist_time = hist_time
        self.lead_time = lead_time


    def __getitem__(self, index):
        # To avoid an index out of bounce error, these if statements were implemented. The first and last datapoints
        # are thereby considered multiple times
        lead_index = index + self.lead_time

        if index < self.hist_time:
            index = self.hist_time
            label_buff = self.nino34[lead_index]
        
        if lead_index >= len(self.nino34):
            index = len(self.nino34) - self.lead_time - 1
            label_buff = self.nino34[-1]

        else:
            label_buff = self.nino34[lead_index]
        

        data_point = np.zeros((self.hist_time * 2, len(self.lat), len(self.lon)))
        data_point[:self.hist_time,:,:] = self.sst[index-self.hist_time:index].data
        data_point[self.hist_time:,:,:] = self.t300[index-self.hist_time:index].data
        data_point = torch.from_numpy(data_point)

        label = {
            "nino3_4" : torch.from_numpy(label_buff.data),
            "time_index" : lead_index
            }
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

