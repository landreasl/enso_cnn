# %%
from numpy.lib.function_base import average
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
import pandas as pd
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


M = 30
N = 30
# %%
for lead_time in range(1):  
    model = nn_enso.Net(M, N, len(dt_cmip.lon),
                        len(dt_cmip.lat), input_channels=6)
    model.load_state_dict(torch.load(PATH + f"/output/model_lead_{lead_time}/cnn_model_1_epochs.pt"))
    model.eval()

    goda_valid_data = SstDataset(godas_path, godas_label_path, is_cmip=False, lead_time = lead_time)
    valid_loader = DataLoader(goda_valid_data, 1, shuffle=False)
    time = []
    pred_nino = []
    average_predicted_nino_time = []
    actual_nino = []
    coefficient_saver = np.zeros((12, 23))
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
    j = 0 # index for coefficient_saver
    for i in range(10, 22): 
        average_predicted_nino = []
        average_actual_nino = []
        a = 0
        average_predicted_nino_time.append(time[i])
        time_buff = []
        while (a+i+1)<len(time):
            average_predicted_nino.append(average(
                [pred_nino[i-1+a], pred_nino[i+a], pred_nino[i+1+a]])
                )
            average_actual_nino.append(average(
                [actual_nino[i-1+a], actual_nino[i+a], actual_nino[i+1+a]])
                )
            time_buff.append(time[a])
            a += 12
        plt.plot(time_buff, average_predicted_nino)
        plt.plot(time_buff, average_actual_nino)
        plt.show()

        coefficient = np.corrcoef(
            np.array(average_predicted_nino).flatten(), np.array(average_actual_nino).flatten()
            )
        coefficient = coefficient[0,1]
        coefficient_saver[11-j][lead_time] = coefficient
        print(coefficient)
        j += 1

# %%

seasons = ["DJF","NDJ", "OND", "SON", "ASO", "JAS", "JJA", "MJJ", "AMJ", "MAM", "FMA", "JFM"]
print(coefficient_saver.shape)
corellation = coefficient_saver
forcast = np.arange(1,24)
fig, ax = plt.subplots()
im = ax.imshow(corellation, cmap="Reds")
ax.set_yticks(np.arange(len(seasons)))
ax.set_xticks(np.arange(0,24))
ax.set_yticklabels(seasons)
ax.set_xticklabels(forcast)
ax.set_ylabel("Season")
ax.set_xlabel("Forecast Lead (months)")
plt.colorbar(im)
    # %%
# %%
