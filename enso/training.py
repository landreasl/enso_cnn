
from dataloader import SstDataset
import nn_enso

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


sodas_path = "enso/H19_dataset/SODA/converted.SODA.input.nc"
sodas_label_path = "enso/H19_dataset/SODA/converted.SODA.label.nino34.12mn_2mv.1873_1972.nc"
godas_path = "enso/H19_dataset/GODAS/converted.GODAS.input.nc"
godas_label_path = "enso/H19_dataset/GODAS/converted.GODAS.label.12mn_2mv.1982_2017.nc"
cmip_path = "enso/H19_dataset/CMIP5/lon_CMIP5_input_sort_1861_2001.nc"
cmip_label_path = "enso/H19_dataset/CMIP5/CMIP5_label_nino34_sort_1861_2001.nc"

dt_cmip = xr.open_dataset(
    "enso/H19_dataset/CMIP5/lon_CMIP5_input_sort_1861_2001.nc")

dt_soda = xr.open_dataset(sodas_path)


model = nn_enso.Net(10, 1, len(dt_cmip.lon),
                    len(dt_cmip.lat), input_channels=6)
lr = 0.005
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
model.train()
n_epochs = 200
lead_time = 3
# %%

# Initialize loss per epoche
epochs_train_saver = np.zeros(n_epochs)
epochs_valid_saver = np.zeros(n_epochs)
epochs_soda_train_saver = np.zeros(n_epochs)

for epochs in range(n_epochs):
    print(f'Epoche: {epochs+1}')

    # Start Training with cmip-data
    # Iterations over all models of cmip
    for cmip_model in range(len(dt_cmip.model)):
        train_loss = 0.0
        cmip_training_data = SstDataset(
            cmip_path, cmip_label_path, lead_time = lead_time, lev=cmip_model+1)
        train_loader = DataLoader(
            cmip_training_data, batch_size=32, shuffle=False)
        for data, target in train_loader:
            model.double()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.float().squeeze(), target.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

        epochs_train_saver[epochs] = train_loss
        print(f'Cmip-Model: {cmip_model+1} \tTraining Loss: {train_loss}')
    

for epochs in range(n_epochs):
    print(f'Epoche: {epochs+1}')
    # Training with SODAS-data
    train_loss = 0.0
    soda_training_data = SstDataset(
        sodas_path, sodas_label_path, is_cmip=False, lead_time = lead_time)
    train_loader = DataLoader(soda_training_data, batch_size=32, shuffle=False)
    for data, target in train_loader:
        model.double()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.float().squeeze(), target.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)

    
    epochs_soda_train_saver[epochs] = train_loss
    print(f'Soda \tTraining Loss: {train_loss}')

    # Validation with GODA-data
    goda_valid_data = SstDataset(godas_path, godas_label_path, is_cmip=False, lead_time = lead_time)
    valid_loader = DataLoader(goda_valid_data, 32, shuffle=False)

    valid_loss = 0.0

    with torch.no_grad():
        for data, target in valid_loader:
            output = model(data)
            loss = criterion(output.float().squeeze(), target.float())
            valid_loss += loss.item()*data.size(0)
        
        epochs_valid_saver[epochs] = valid_loss
    
    print(f'Goda \tValidation Loss: {valid_loss}')



plt.plot(epochs_train_saver, label='train-loss CMIP')
plt.plot(epochs_valid_saver, label='validation-loss GODA')
plt.plot(epochs_soda_train_saver, label='train-loss SODAS')
plt.legend()
plt.title(f"Number of Epochs = {n_epochs} Learning Rate = {lr}")
plt.savefig(f"/Users/andreasbaumgartner/Desktop/enso_training/{n_epochs}_epochs_{lr}_lr.png")
plt.show()


# %%
"""    with torch.no_grad():
        for data, target in valid_loader:
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)
        
        valid_loss = valid_loss/len(valid_loader.sampler)
        epochs_valid_saver[epochs] = valid_loss"""

# %%
