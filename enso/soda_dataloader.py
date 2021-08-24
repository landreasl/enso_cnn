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
dt = xr.open_dataset("/Users/andreasbaumgartner/Documents/Enso/enso2/enso3/enso_cnn/enso/H19_dataset/SODA/converted.SODA.input.nc")
label = xr.open_dataset("/Users/andreasbaumgartner/Documents/Enso/enso2/enso3/enso_cnn/enso/H19_dataset/SODA/converted.SODA.label.nino34.12mn_2mv.1873_1972.nc")


dt
label

# %%
dt_sst = dt["ssta"].sel(
                    time=slice(dt.time[24],dt.time[-1]),
                    model=lev
                    )

dt_nino = label_cmip["nino34"].sel(
                        time=slice(label_cmip.time[0], label_cmip.time[-25]),
                        model=lev)



len(dt_nino)

# %%
# TODO cmip 3 jahre shift, dataset abfragen (erben) 
class SodaSstDataset(Dataset):
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
            self.lev = dt["lev"]

            self.time = dt["time"][2:]  # Data is shifted by 2 years with respect to nino labels
            self.sst = dt["sst"][2:].sel(
                lev=slice(24,37) # Choose time between january and december 
            )  
            self.t300 = dt["t300"][2:].sel(
                lev=slice(24,37)
            )
            self.n_samples = dt.time.size
            with xr.open_dataset(label_file) as dt:
                self.nino34 = dt["pr"][:-2] 
        
        self.hist_time = hist_time
        self.lead_time = lead_time


    def __getitem__(self, index):
        # To avoid an index out of bounce error, these if-statements were implemented. The first and last datapoints
        # are thereby considered multiple times. 
        # Different to cmip_dataloader because of data's structure
        lead_index = index + self.lead_time

        if index < self.hist_time:
            index = self.hist_time
            lead_index = index + self.lead_time
            label_buff = self.nino34.sel(
                time = self.time[int(lead_index/12)],
                lev = self.lev[(lead_index+1)%12]
            )
        
        if lead_index >= (len(self.nino34.time)*12):
            index = (len(self.nino34) * 12) - self.lead_time - 1
            lead_index = index + self.lead_time
            label_buff = self.nino34.sel(
                time = self.time[int(lead_index/12)],
                lev = self.lev[(lead_index+1)%12])

        else: 
            label_buff = self.nino34.sel(
                time = self.time[int(lead_index/12)],
                lev = self.lev[(lead_index+1)%12]
            )
        
        sst_buff = self.sst.sel(
            time = self.time[int(index/12)]
        ).data

        t300_buff = self.t300.sel(
            time = self.time[int(index/12)]
        ).data

        data_point = np.zeros((self.hist_time * 2, len(self.lat), len(self.lon)))

        for i in range(self.hist_time+1):
            data_point[i,:,:] = self.sst.sel(
                time = self.time[int(index-self.hist_time+1+i/12)],
                lev = (self.lev[((index-i)%12)+24])
            ).data
            data_point[self.hist_time+i,:,:] = self.t300.sel(
                time = self.time[int(index-i/12)],
                lev = (self.lev[((index-i)%12)+24])
            ).data
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
    
print(12%15)


# %%
