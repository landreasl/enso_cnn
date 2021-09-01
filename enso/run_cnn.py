# %%
from dataloader import SstDataset
import nn_enso
import train

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


# Set Parameters and paths
sodas_path = PATH + "/H19_dataset/SODA/converted.SODA.input.nc"
sodas_label_path = PATH + "/H19_dataset/SODA/converted.SODA.label.nino34.12mn_2mv.1873_1972.nc"
godas_path = PATH + "/H19_dataset/GODAS/converted.GODAS.input.nc"
godas_label_path = PATH + "/H19_dataset/GODAS/converted.GODAS.label.12mn_2mv.1982_2017.nc"
cmip_path = PATH + "/H19_dataset/CMIP5/lon_CMIP5_input_sort_1861_2001.nc"
cmip_label_path = PATH + "/H19_dataset/CMIP5/CMIP5_label_nino34_sort_1861_2001.nc"
dt_cmip = xr.open_dataset(cmip_path)
# Number of Feature-Maps (30 oder 50)
M = 30
# Number of Neurons (30 oder 50)
N = 30
# GPU or CPU
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
print('Device to run vae: ', device)

lead_time = 3

# %%
# Create CNN model
LR = 0.005
BATCH_SIZE = 400
model = nn_enso.Net(M, N, len(dt_cmip.lon),
                    len(dt_cmip.lat), input_channels=6).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=LR)
model.train()

# %%
PRETRAINING_EPOCHS = 1
REANALYSIS_EPOCHS = 1

# Train on CMIP
model, train_loss_cmip, val_loss_cmip = train.cmip_pretraining(
    cmip_path, cmip_label_path, godas_path, godas_label_path,
    lead_time, PRETRAINING_EPOCHS, model, optimizer, criterion,
    device=device, batch_size=BATCH_SIZE
)
# Train on SODAS
model, train_loss_sodas, val_loss_sodas = train.reanalysis_training(
    sodas_path, sodas_label_path, godas_path, godas_label_path,
    lead_time, REANALYSIS_EPOCHS, model, optimizer, criterion,
    device=device, batch_size=BATCH_SIZE
)

torch.save(model.state_dict(),
           PATH + f"/output/cnn_model_lead_{lead_time}_epochs_{PRETRAINING_EPOCHS}_{REANALYSIS_EPOCHS}.pt")


# %%
# Plot losses
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot((train_loss_cmip), label='train-loss CMIP')
ax1.plot((val_loss_cmip), label='validation-loss GODA')
ax1.set_title(f"Number of Epochs = {PRETRAINING_EPOCHS} Learning Rate = {LR}")
plt.savefig(PATH + f"/plots/loss_cmip_lead_{lead_time}_epochs_{REANALYSIS_EPOCHS}_LR_{LR}.png")

ax2.plot((train_loss_sodas), label='train-loss SODA')
ax2.plot((val_loss_sodas), label='val-loss GODAS')
ax2.legend()
ax2.set_title(f"Number of Epochs = {REANALYSIS_EPOCHS} Learning Rate = {LR}")
plt.savefig(PATH + f"/plots/loss_soda_lead_{lead_time}_epochs_{REANALYSIS_EPOCHS}_LR_{LR}.png")
# %%
# # Load model
# model = nn_enso.Net(M, N, len(dt_cmip.lon),
#                     len(dt_cmip.lat), input_channels=6).to(device)
# model.load_state_dict(torch.load(PATH + "/output/cnn_model_600_epochs.pt"))
# model.eval()
# %%
# Compare Nino3.4 and its prediction
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
        data = data.to(device)
        if count < hist_time:
            None
        elif count >= 405:
            break
        else:    
            model.double()
            pred_nino.append(model(data).cpu().numpy()) 
            actual_nino.append(label["nino3_4"].numpy())
            time.append(goda_valid_data.time[label["time_index"]])
        
        count += 1

plt.plot(np.array(time).flatten(), np.array(actual_nino).flatten(), label = "nino3.4")
plt.plot(np.array(time).flatten(), np.array(pred_nino).flatten(), label = "predicted nino3.4")
plt.legend()
plt.savefig(PATH + f"/plots/Nino34_prediction_{PRETRAINING_EPOCHS}_epochs_{LR}_lr.png")

# Correlation coefficient 
coefficient = np.corrcoef(np.array(pred_nino).flatten(), np.array(actual_nino).flatten())
coefficient = coefficient[0,1]
print(coefficient)
    # %%
with open(PATH + "/output/coefficient.txt", "a") as f:
     f.write(f"{lead_time}\t {coefficient}\n")
# %%
