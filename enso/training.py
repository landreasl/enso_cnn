# %%
from dataloader import SstDataset
import nn_enso

import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


PATH = os.path.dirname(os.path.abspath(__file__))

sodas_path = PATH + "/H19_dataset/SODA/converted.SODA.input.nc"
sodas_label_path = PATH + "/H19_dataset/SODA/converted.SODA.label.nino34.12mn_2mv.1873_1972.nc"
godas_path = PATH + "/H19_dataset/GODAS/converted.GODAS.input.nc"
godas_label_path = PATH + "/H19_dataset/GODAS/converted.GODAS.label.12mn_2mv.1982_2017.nc"
cmip_path = PATH + "/H19_dataset/CMIP5/lon_CMIP5_input_sort_1861_2001.nc"
cmip_label_path = PATH + "/H19_dataset/CMIP5/CMIP5_label_nino34_sort_1861_2001.nc"

dt_cmip = xr.open_dataset(
    cmip_path)
# %%
model = nn_enso.Net(10, 1, len(dt_cmip.lon),
                    len(dt_cmip.lat), input_channels=6)



lr = 0.005
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
model.train()
n_epochs = 100
# %%

# Initialize loss per epoche
epochs_train_saver = np.zeros(n_epochs)
epochs_valid_saver = np.zeros(20)
epochs_soda_train_saver = np.zeros(20)


for lead_time in range(1,24):
    for epochs in range(n_epochs):
        print(f'Epoche: {epochs+1}')
        # Start Training with cmip-data
        # Iterations over all models of cmip
        for cmip_model in range(len(dt_cmip.model)):
            train_loss = 0.0
            cmip_training_data = SstDataset(
                cmip_path, cmip_label_path, lead_time = lead_time, lev=cmip_model+1)
            train_loader = DataLoader(
                cmip_training_data, batch_size=32, shuffle=True)
            for data, target in train_loader:
                target = target["nino3_4"]
                model.double()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output.float().squeeze(), target.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*data.size(0)

            epochs_train_saver[epochs] = train_loss
            print(f'Cmip-Model: {cmip_model+1} \tTraining Loss: {train_loss}')
        

    for epochs in range(20):
        print(f'Epoche: {epochs+1}')
        # Training with SODAS-data
        train_loss = 0.0
        soda_training_data = SstDataset(
            sodas_path, sodas_label_path, is_cmip=False, lead_time = lead_time)
        train_loader = DataLoader(soda_training_data, batch_size=32, shuffle=True)
        for data, target in train_loader:
            target = target["nino3_4"]
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
                target = target["nino3_4"]
                output = model(data)
                loss = criterion(output.float().squeeze(), target.float())
                valid_loss += loss.item()*data.size(0)
            
            epochs_valid_saver[epochs] = valid_loss
        
        print(f'Goda \tValidation Loss: {valid_loss}')
    # %%

    plt.plot(epochs_train_saver, label='train-loss CMIP')
    plt.title(f"Number of Epochs = {n_epochs} Learning Rate = {lr}")
    plt.savefig(f"/Users/andreasbaumgartner/Desktop/enso_training/{n_epochs}_epochs_{lr}_lr_cmip.png")
    plt.show()
    plt.plot(epochs_valid_saver, label='validation-loss GODA')
    plt.plot(epochs_soda_train_saver, label='train-loss SODAS')
    plt.legend()
    plt.title(f"Number of Epochs = {n_epochs} Learning Rate = {lr}")
    plt.savefig(f"/Users/andreasbaumgartner/Desktop/enso_training/{n_epochs}_epochs_{lr}_lr_soda.png")
    plt.show()
    # %%
    torch.save(model.state_dict(), PATH + f"/output/cnn_model_{n_epochs}_epochs.pt")


    # %%
    model = nn_enso.Net(10, 1, len(dt_cmip.lon),
                        len(dt_cmip.lat), input_channels=6)
    model.load_state_dict(torch.load(PATH + "/output/cnn_model_100_epochs.pt"))
    model.eval()
    # %%
    lead_time = 3
    goda_valid_data = SstDataset(godas_path, godas_label_path, is_cmip=False, lead_time = lead_time)
    valid_loader = DataLoader(goda_valid_data, 32, shuffle=False)
    time = []
    pred_nino = []
    actual_nino = []
    count = 0
    hist_time = 3
    with torch.no_grad():
        for data, label in valid_loader:
            if count < hist_time:
                print("out")

            elif count >= 12:
                print("in")
                break

            else:    
                model.double()
                pred_nino.append(model(data).numpy()) 
                actual_nino.append(label["nino3_4"].numpy())
                time.append(goda_valid_data.time[label["time_index"]])
                print("yes" + str(count))
            
            count += 1



    # %%
    plt.plot(np.array(time).flatten(), np.array(actual_nino).flatten(), label = "nino3.4")
    plt.plot(np.array(time).flatten(), np.array(pred_nino).flatten(), label = "predicted nino3.4")
    plt.legend()
    plt.show()
    # %%
    np.corrcoef(np.array(pred_nino).flatten(), np.array(actual_nino).flatten())
    # %%
