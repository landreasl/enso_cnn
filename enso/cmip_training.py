# %%
from enso_dataloader import SstDataset
import nn_enso

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import xarray as xr
import numpy as np

sodas_path = "enso/H19_dataset/SODA/SODA.input.36mn.1871_1970.nc"
sodas_label_path= "enso/H19_dataset/SODA/SODA.label.nino34.12mn_2mv.1873_1972.nc"
godas_path = "enso/H19_dataset/GODAS/GODAS.input.36mn.1980_2015.nc"
godas_label_path = "enso/H19_dataset/GODAS/GODAS.label.12mn_2mv.1982_2017.nc"
cmip_path = "enso/H19_dataset/CMIP5/lon_CMIP5_input_sort_1861_2001.nc"
cmip_label_path = "enso/H19_dataset/CMIP5/CMIP5_label_nino34_sort_1861_2001.nc"

# Open Datasets
dt_sodas = xr.open_dataset("enso/H19_dataset/SODA/SODA.input.36mn.1871_1970.nc")
dt_godas = xr.open_dataset("enso/H19_dataset/GODAS/GODAS.input.36mn.1980_2015.nc")
dt_cmip = xr.open_dataset("enso/H19_dataset/CMIP5/lon_CMIP5_input_sort_1861_2001.nc")

# Open label files with nino 3.4 index
label_cmip = xr.open_dataset("enso/H19_dataset/CMIP5/CMIP5_label_nino34_sort_1861_2001.nc")
label_sodas = xr.open_dataset("enso/H19_dataset/SODA/SODA.label.nino34.12mn_2mv.1873_1972.nc")
label_godas = xr.open_dataset("enso/H19_dataset/GODAS/GODAS.label.12mn_2mv.1982_2017.nc")

dt_cmip

# %%
model = nn_enso.Net(10, 1, len(dt_cmip.lon), len(dt_cmip.lat), input_channels=6)
lr = 0.0001
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr) 
model.train()
n_epochs = 10
# %%

# Initialize loss per epoche
epochs_train_saver = np.zeros(n_epochs)
epochs_valid_saver = np.zeros(n_epochs)

for epochs in range(n_epochs):
    train_loss = 0.0

# Iteration over all models of cmip
    for cmip_model in dt_cmip.model:
        cmip_training_data = SstDataset(cmip_path, cmip_label_path, lev=cmip_model+1)
        train_loader = DataLoader(cmip_training_data, batch_size=64, shuffle=False)
        iterations = 0
        for data, target in train_loader:
            model.double() 
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.float(), target.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

        epochs_train_saver[epochs] = train_loss
        print(f'Epoch: {epochs+1}\t Cmip-Model: {cmip_model} \tTraining Loss: {train_loss}')



# %%
"""    with torch.no_grad():
        for data, target in valid_loader:
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)
        
        valid_loss = valid_loss/len(valid_loader.sampler)
        epochs_valid_saver[epochs] = valid_loss"""

# %%
