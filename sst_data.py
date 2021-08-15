"""
Loading and preprocessing of SST data.

@author: Jakob Schlör
"""
import sys, os
import datetime
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy as ctp

PATH = os.path.dirname(os.path.abspath(__file__))


class SSTAData(Dataset):
    """SST anomalies dataset.

    Args:
    ----------
    nc_file: (str)  
        filename  
    lon_range: (list)
        range of longitudes where the ssta are cut 
    lat_range: (list)
        range of latitudes where the ssta are cut
    normalization: (str)
        method of normalizing the ssta data, i.e. False (default), 'minmax', 'zscore'
    transform: (bool)
        torch.transform objects for transformation upon calling get_item from dataloader
    """

    def __init__(self, nc_file, 
                 lon_range=[-70, 120], lat_range=[-30, 30], 
                 normalization=None, transform=None):
        super().__init__()

        with xr.open_dataset(nc_file) as ds:
            # change coordinates
            self.dataset = self.set_antimeridian2zero(ds)
            da = self.dataset['analysed_sst']
            # cut pacific
            lon_range = gp.get_antimeridian_coord(lon_range)
            self.da_cutArea = gp.cut_map_area(da, 
                lon_range=lon_range, lat_range=lat_range)
        
        # Normalize data if necessary
        self.normalization = normalization
        if self.normalization is not None:
            self.dataarray = self.normalize_data(method=normalization)
        else:
            self.dataarray = self.da_cutArea.copy()

        self.dims = self.dataarray.shape
        self.dim_name = self.dataarray.dims
        self.time = self.dataarray[self.dim_name[0]].data
        self.x = self.dataarray[self.dim_name[1]].data
        self.y = self.dataarray[self.dim_name[2]].data

        # Store the position of NaNs which is needed for reconstructing the map later
        self.idx_nan = xr.ufuncs.isnan(self.dataarray)

        self.transform = transform
    

    def __len__(self):
        return len(self.dataarray)
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'data':   self.dataarray[idx].data,
                  'label':  self.dataarray[idx].time.dt.month.data, #[0, 0], # TODO: month dim for normalized data
                  'time':  self.time2timestamp(self.dataarray[idx].time.data)}

        if self.transform:
            sample = self.transform(sample)

        return sample['data'], {'label': sample['label'], 'time': sample['time']}
    

    def get_dataarray(self):
        return self.dataarray
    

    def time2timestamp(self, time):
        """Convert np.datetime64 to int."""
        return (time - np.datetime64('1950-01-01', 'ns')) / np.timedelta64(1, 'D') 
    

    def timestamp2time(self, timestamp):
        """Convert timestamp to np.datetime64 object."""
        return (np.datetime64('1950-01-01', 'ns') 
                + timestamp * np.timedelta64(1, 'D') )
    
    
    def normalize_data(self, method='minmax'):
        """Normalize self.da_cutArea with a given method.

        Args:
        -----
        method (str)
            method to normalize data, e.g. 'minmax', 'zscore'
        """
        flatten = self.da_cutArea.stack(z = self.da_cutArea.dims)
        if method == 'minmax':
            norm_data = (
                (flatten - flatten.min(skipna=True)) / 
                (flatten.max(skipna=True) - flatten.min(skipna=True))
            )
        elif method == 'zscore':
            norm_data = (
                (flatten - flatten.mean(skipna=True)) / flatten.std(skipna=True)
            )
        else:
            print(f'Your selected normalization method "{method}" does not exist.')

        return norm_data.unstack('z')


    def unnormalize_data(self, dmap, method='minmax'):
        """Back transformation from normalized units to temperature.
        Args:
            dmap: (xr.dataArray) input map in normalized units

        Return:
            (xr.dataArray) transformed dmap in temperature units
        """
        flatten = self.da_cutArea.stack(z = self.dataarray.dims)
        if method == 'minmax':
            max_arr = flatten.max(skipna=True)
            min_arr = flatten.min(skipna=True)
            rec_map = dmap * (max_arr - min_arr) + min_arr         
        elif method == 'zscore':
            rec_map = (dmap * flatten.std(skipna=True)) + flatten.mean(skipna=True)
        else:
            print(f'Your selected normalization method "{method}" does not exist.')
        return rec_map


    def flatten_map(self):
        """Return flattened data array in spatial dimension, i.e. (time, n_x*n_y).
        
        NaNs are removed in each dimension.
        """
        flat_arr = self.dataarray.data.reshape(
            self.dims[0], self.dims[1]*self.dims[2]
        )
        flat_nonans = []
        for a in flat_arr:
            idx_nan = np.isnan(a)
            flat_nonans.append(a[~idx_nan])

        return np.array(flat_nonans)


    def get_map(self, data, name=None, unnormalize=True):
        """Restore data map (xr.DataArray) from flattened array.

        This also includes adding NaNs which have been removed.

        Parameters:
        -----------
        data: (torch.tensor)
            Flatten datapoint with NaNs removed
        name: (str)
            Name of returned xr.DataArray
        unnormalize: boolean
            If data was normalized before it will be inversed. Default: True
        
        #TODO: time stamp in sample should be used to identify self.idx_nan 
        """
        if torch.is_tensor(data):
            data = data.to('cpu').detach().numpy()

        idx_nan_arr = self.idx_nan.data[0].flatten()
        # Number of non-NaNs should be equal to length of data
        assert np.count_nonzero(~idx_nan_arr) == len(data)
        # create array with NaNs
        data_map = np.empty(len(idx_nan_arr)) 
        data_map[:] = np.NaN
        # fill array with sample
        data_map[~idx_nan_arr] = data

        dmap = xr.DataArray(
            data=np.reshape(data_map, self.idx_nan.data[0].shape),
            coords=[self.dataarray.coords['lat'], self.dataarray.coords['lon']],
            name=name) 

        if self.normalization is not None and unnormalize is True:
            dmap = self.unnormalize_data(dmap, method=self.normalization)

        return dmap
    

    def get_map_index(self, idx_flat):
        """Get lat, lon and index of map from index of fatten array 
           withour Nans.
        
        Args:
        -----
        idx_flat: int, list
            index or list of indices of the flatten array with removed NaNs
        
        Return:
        idx_map: dict
            Corresponding indices of the map as well as lat and lon coordinates
        """

        mask_nan_arr = self.idx_nan.data[0].flatten()
        indices_flat = np.arange(0, np.count_nonzero(~mask_nan_arr), 1,
                                 dtype=int)
        # create array with NaNs
        idx_map_flat = np.empty(len(mask_nan_arr)) 
        idx_map_flat[:] = np.NaN
        # fill array with sample
        idx_map_flat[~mask_nan_arr] = indices_flat
        idx_map = xr.DataArray(
            data=np.reshape(idx_map_flat, self.idx_nan.data[0].shape),
            coords=[self.dataarray.coords['lat'], self.dataarray.coords['lon']],
            name='idx_map') 

        buff = idx_map.where(idx_map == idx_flat, drop=True)
        map_idx = {
            'lat': buff.lat.data,
            'lon': buff.lon.data,
            'idx': np.argwhere(idx_map.data == idx_flat)
        }
        return map_idx

    
    def get_map_convolution(self, data, name=None):
        """Restore xArray anomaly map from transformed torch.Tensor.

        Args:
        -----
        data: torch.Tensor (1, num_lat, num_lon)
            Datapoint 
        name: str

        Return:
        -------
        map in 
        """
        data = data[0,:,:]
        if torch.is_tensor(data):
            data = data.to('cpu').detach().numpy()

        # Dimensions should be equal
        assert data.shape == self.idx_nan.data[0].shape

        # create array
        data_map = np.copy(data) 
#        data_map[self.idx_nan.data[0]] = np.NaN

        dmap = xr.DataArray(
            data=data_map,
            coords=[self.dataarray.coords['lat'], self.dataarray.coords['lon']],
            name=name) 

        if self.normalization is not None:
            dmap = self.unnormalize_data(dmap, method=self.normalization)

        return dmap



    def get_nino_indices(self, time_average=0):
        """Returns the time series of the Nino 1+2, 3, 3.4, 4.

        Parameters:
        -----------
        time_average: int
            Number of timesteps the rolling average is applied
        
        Returns:
        --------
        dictionary with time series
        
        """

        da = self.dataset['analysed_sst']
        nino12, nino12_std = gp.get_mean_time_series(
            da, lon_range=gp.get_antimeridian_coord([-90, -80]),
            lat_range=[-10, 0], time_roll=time_average
        )
        nino3, nino3_std = gp.get_mean_time_series(
            da, lon_range=gp.get_antimeridian_coord([-150, -90]),
            lat_range=[-5, 5], time_roll=time_average
        )
        nino34, nino34_std = gp.get_mean_time_series(
            da, lon_range=gp.get_antimeridian_coord([-170, -120]),
            lat_range=[-5, 5], time_roll=time_average
        )
        nino4, nino4_std = gp.get_mean_time_series(
            da, lon_range=gp.get_antimeridian_coord([160, -150]),
            lat_range=[-5, 5], time_roll=time_average
        )

        da = xr.DataArray( 
            np.array([nino12.data, nino3.data, nino34.data, nino4.data]).T,
            dims=['time', 'nino_idx'],
            coords=dict(time=nino12.time,
                        nino_idx=['12', '3', '34', '4'])
        )
        
        return da
    

    def get_nino_flavors(self):
        """ Get nino flavors from differences between Nino3 and Nino4.
        
        """
        # Nino indices
        nino_indices = self.get_nino_indices(time_average=90)

        # Identify El Nino and La Nina types 
        nino_years = []
        times = nino_indices.time
        for y in np.arange(times.min().data, times.max().data, dtype="datetime64[Y]"):
            nino3 = nino_indices.sel(time=f'{y}-01-15')[0,1]
            nino4 = nino_indices.sel(time=f'{y}-01-15')[0,3]

            # El Nino years
            if nino3.data >= 0.5 or nino4.data >= 0.5:
                buff_dic = {'year': f'{y}-01-15', 'type': 'Nino', 'label': 0}

                # EP type if DJF nino3 > 0.5 and nino3 > nino4 
                if nino3.data > nino4.data:
                    buff_dic['type'] = 'Nino_EP'
                    buff_dic['label'] = 1
                # CP type if DJF nino4 > 0.5 and nino3 < nino4
                elif nino4.data > nino3.data:
                    buff_dic['type'] = 'Nino_CP'
                    buff_dic['label'] = 2
                nino_years.append(buff_dic)
            elif nino3.data <= -0.5 or nino4.data <= -0.5:
                buff_dic = {'year': f'{y}-01-15', 'type': 'Nina', 'label': 3}
                # EP type if DJF nino3 < -0.5 and nino3 < nino4 
                if nino3.data < nino4.data:
                    buff_dic['type'] = 'Nina_EP'
                    buff_dic['label'] = 4
                # CP type if DJF nino4 < -0.5 and nino3 > nino4
                elif nino4.data < nino3.data:
                    buff_dic['type'] = 'Nina_CP'
                    buff_dic['label'] = 5
                nino_years.append(buff_dic)
        
        return nino_years


    
    ##########################################
    # Plotting functions for sst dataset
    ##########################################
    def plot_map(self, dmap, central_longitude=0, vmin=None, vmax=None,
                 ax=None, fig=None, color='RdBu_r', bar=True, ctp_projection='PlateCarree',
                 label=None):
        """Simple map plotting using xArray.
        
        Parameters:
        -----------
        """
        if ax is None:
            # set projection
            if ctp_projection =='Mollweide':
                proj = ctp.crs.Mollweide(central_longitude=central_longitude)
            elif ctp_projection=='PlateCarree':
                proj = ctp.crs.PlateCarree(central_longitude=central_longitude)
            else:
                raise ValueError(f'This projection {ctp_projection} is not available yet!')

            # create figure
            fig, ax = plt.subplots(figsize=(9,6))
            ax = plt.axes(projection=proj)

        # axes properties
        ax.coastlines()
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)    
        ax.add_feature(ctp.feature.RIVERS)
        ax.add_feature(ctp.feature.BORDERS, linestyle=':')

        # set colormap
        cmap = plt.get_cmap(color)
        kwargs_pl = dict() # kwargs plot function
        kwargs_cb = dict() # kwargs colorbar 
        if bar=='discrete':
            normticks = np.arange(0, dmap.max(skipna=True)+2, 1, dtype=int)
            kwargs_pl['norm'] = mpl.colors.BoundaryNorm(normticks, cmap.N)
            kwargs_cb['ticks'] = normticks + 0.5 
        
         # choose symmetric vmin and vmax
        if vmin is None and vmax is None:
            vmin = dmap.min(skipna=True)
            vmax = dmap.max(skipna=True)
            vmax = vmax if vmax > (-1*vmin) else (-1*vmin) 
            vmin = -1*vmax

        # plot map
        im = ax.pcolormesh(
            dmap.coords['lon'], dmap.coords['lat'], dmap.data,
            cmap=cmap, vmin=vmin, vmax=vmax, 
            transform=ctp.crs.PlateCarree(central_longitude=central_longitude),
            **kwargs_pl
        )

        # set colorbar
        if bar:
            label = dmap.name if label is None else label
            cbar = plt.colorbar(im, extend='both', orientation='horizontal',
                                label=label, shrink=1.0, ax=ax, **kwargs_cb)

            if bar=='discrete':
                cbar.ax.set_xticklabels(normticks[:-1]+1)

        return {'ax': ax, "im": im, 'fig': fig}


############################################
# Helper functions
############################################
def get_antimeridian_coord(lons):
    """Change of coordinates from normal to antimeridian."""
    lons = np.array(lons)
    lons_new = np.where(lons < 0, (lons + 180),(lons - 180)) 
    return lons_new

def set_antimeridian2zero(ds):
    """Set the antimeridian to zero.

    Easier to work with the pacific then.
    """
    # Roll data such that it is centered around the antimeridian
    ds_rolled = ds.roll(lon=(ds.dims['lon'] // 2))
    # Change lon coordinates
    lons = ds_rolled.lon
    lons_new = get_antimeridian_coord(lons)
    ds_rolled = ds_rolled.assign_coords(
        lon=lons_new
    )
    print('Set the antimeridian to the new longitude zero.')
    return ds_rolled 

def cut_map_area(ds, lon_range, lat_range):
    """Cut an area in the map."""
    ds_area = ds.sel(
        lon=slice(lon_range[0], lon_range[1]),
        lat=slice(lat_range[0], lat_range[1])
    )
    return ds_area

def get_mean_time_series(da, lon_range, lat_range, time_roll=0):
    """Get mean time series of selected area.

    Parameters:
    -----------
    da: xr.DataArray
        Data
    lon_range: list
        [min, max] of longitudinal range
    lat_range: list
        [min, max] of latiduninal range
    """
    da_area = cut_map_area(da, lon_range, lat_range) 
    ts_mean = da_area.mean(dim=('lon', 'lat'), skipna=True)
    ts_std = da_area.std(dim=('lon', 'lat'), skipna=True)
    if time_roll > 0:
        ts_mean = ts_mean.rolling(time=time_roll, center=True).mean()
        ts_std = ts_std.rolling(time=time_roll, center=True).mean()

    return ts_mean, ts_std