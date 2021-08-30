# %%
from dataloader import SstDataset
import nn_enso

from posixpath import dirname
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
# Number of Feature-Maps (30 oder 50)
M = 30
# Number of Neurons (30 oder 50)
N = 30
model = nn_enso.Net(M, N, len(dt_cmip.lon),
                    len(dt_cmip.lat), input_channels=6)

LR = 0.005
BATCH_SIZE = 400
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=LR)
model.train()

PRETRAINING_EPOCHS = 600
REANALYSIS_EPOCHS = 20
# %%

# Initialize loss per epoche
epochs_train_saver = np.zeros(PRETRAINING_EPOCHS)
epochs_valid_saver = np.zeros(REANALYSIS_EPOCHS)
epochs_soda_train_saver = np.zeros(REANALYSIS_EPOCHS)

for lead_time in range(1,24):
    dirname = PATH + f"/output/model_lead_{lead_time}"
    try: 
        os.mkdir(dirname)
    except:
        print("directory failed")
    for epochs in range(PRETRAINING_EPOCHS):
        print(f'Epoche: {epochs+1}')
        # Start Training with cmip-data
        # Iterations over all models of cmip
        for cmip_model in range(len(dt_cmip.model)):
            train_loss = 0.0
            cmip_training_data = SstDataset(
                cmip_path, cmip_label_path, lead_time = lead_time, lev=cmip_model+1)
            train_loader = DataLoader(
                cmip_training_data, batch_size=BATCH_SIZE, shuffle=True)
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

    for epochs in range(REANALYSIS_EPOCHS):
        print(f'Epoche: {epochs+1}')
        # Training with SODAS-data
        train_loss = 0.0
        soda_training_data = SstDataset(
            sodas_path, sodas_label_path, is_cmip=False, lead_time = lead_time)
        train_loader = DataLoader(soda_training_data, batch_size=BATCH_SIZE, shuffle=True)
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
        valid_loader = DataLoader(goda_valid_data, BATCH_SIZE, shuffle=False)

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
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot((epochs_train_saver), label='train-loss CMIP')
    ax1.set_title(f"Number of Epochs = {PRETRAINING_EPOCHS} Learning Rate = {LR}")
    ax2.plot((epochs_valid_saver), label='validation-loss GODA')
    ax2.plot((epochs_soda_train_saver), label='train-loss SODAS')
    ax2.legend()
    ax2.set_title(f"Number of Epochs = {REANALYSIS_EPOCHS} Learning Rate = {LR}")
    plt.savefig(dirname + f"/{PRETRAINING_EPOCHS}_epochs_{LR}_lr.png")
    # %%
    torch.save(model.state_dict(), dirname + f"/cnn_model_{PRETRAINING_EPOCHS}_epochs.pt")
    # %%
    lead_time = 3
    goda_valid_data = SstDataset(godas_path, godas_label_path, is_cmip=False, lead_time = lead_time)
    valid_loader = DataLoader(goda_valid_data, 1, shuffle=False)
    time = []
    pred_nino = []
    actual_nino = []
    count = 0
    hist_time = 3
    with torch.no_grad():
        for data, label in valid_loader:
            if count < hist_time:
                None

            elif count >= 405:
                break

            else:    
                model.double()
                pred_nino.append(model(data).numpy()) 
                actual_nino.append(label["nino3_4"].numpy())
                time.append(goda_valid_data.time[label["time_index"]])
            
            count += 1

    # Auskommentiert, da sich der Plot nicht speichern liess (value error)
    """plt.plot((np.array(time).flatten()), np.array(actual_nino).flatten(), label = "nino3.4", )
    plt.plot((np.array(time)).flatten(), np.array(pred_nino).flatten(), label = "predicted nino3.4")
    plt.legend()
    plt.show()
    plt.savefig(dirname + "/nino.png")"""

    coefficient = np.corrcoef(np.array(pred_nino).flatten(), np.array(actual_nino).flatten())
    coefficient = coefficient[0,1]
    print(coefficient)
    # %%
    with open("coefficient.txt", "a") as f:
        f.write(f"{lead_time}\t {coefficient}\n")
    # %%
