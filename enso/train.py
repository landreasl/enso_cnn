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



def fit_model(model, optimizer, criterion, loader, device='cpu'):
    """Fit model with backpropagation.

    Args:
        model ([type]): [description]
        optimizer ([type]): [description]
        criterion ([type]): [description]
        loader ([type]): [description]
        device (str, optional): [description]. Defaults to 'cpu'.

    Returns:
        model (torch.nn) 
        loss (list)
    """
    train_loss = 0.0
    for data, target in loader:
        data = data.to(device)
        target = target["nino3_4"].to(device)
        model.double()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.float().squeeze(), target.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    
    return model, train_loss


def validate_model(model, criterion, loader, device='cpu'):
    """Validation loss of model.

    Args:
        model ([type]): [description]
        criterion ([type]): [description]
        loader ([type]): [description]
        device (str, optional): [description]. Defaults to 'cpu'.

    Returns:
        model (torch.nn) 
        valid_loss (list)
    """
    valid_loss = 0.0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target["nino3_4"].to(device)
            output = model(data)
            loss = criterion(output.float().squeeze(), target.float())
            valid_loss += loss.item()*data.size(0)

    return model, valid_loss


def cmip_pretraining(cmip_path, cmip_label_path, 
                     godas_path, godas_label_path,
                     lead_time, epochs,
                     model, optimizer, criterion,
                     device='cpu', batch_size=400):
    """Pretraining the model with CMIP data.

    Args:
        cmip_path ([type]): [description]
        cmip_label_path ([type]): [description]
        godas_path ([type]): [description]
        godas_label_path ([type]): [description]
        lead_time ([type]): [description]
        epochs ([type]): [description]
        model ([type]): [description]
        optimizer ([type]): [description]
        criterion ([type]): [description]
        device (str, optional): [description]. Defaults to 'cpu'.
        batch_size (int, optional): [description]. Defaults to 400.

    Returns:
        [type]: [description]
    """
    train_loss = np.zeros(epochs)
    valid_loss = np.zeros(epochs)

    dt_cmip = xr.open_dataset(cmip_path)
    goda_valid_data = SstDataset(godas_path, godas_label_path, is_cmip=False, lead_time = lead_time)
    valid_loader = DataLoader(goda_valid_data, batch_size, shuffle=False)

    for epoch in range(epochs):
        print(f'Epoche: {epoch+1}')
        # Start Training with cmip-data
        # Iterations over all models of cmip
        for cmip_model in range(len(dt_cmip.model)):
            cmip_training_data = SstDataset(
                cmip_path, cmip_label_path, lead_time=lead_time, lev=cmip_model+1)
            train_loader = DataLoader(
                cmip_training_data, batch_size=batch_size, shuffle=True)
            
            model, loss = fit_model(model, optimizer, criterion, train_loader, device)
            train_loss[epoch] = loss
            print(f'Cmip-Model: {cmip_model+1} \tTraining Loss: {loss}')
        
        # validation
        model, loss = validate_model(model, criterion, valid_loader, device)
        valid_loss[epoch] = loss
        print(f'Validation loss: {loss}')

    return model, train_loss, valid_loss


def reanalysis_training(sodas_path, sodas_label_path,
                        godas_path, godas_label_path,
                        lead_time, epochs,
                        model, optimizer, criterion,
                        device='cpu', batch_size=400):
    # Load training data
    soda_training_data = SstDataset(
        sodas_path, sodas_label_path, is_cmip=False, lead_time = lead_time
    )
    train_loader = DataLoader(soda_training_data, batch_size=batch_size,
                              shuffle=True)
    # Load validation data
    goda_valid_data = SstDataset(godas_path, godas_label_path, is_cmip=False, lead_time = lead_time)
    valid_loader = DataLoader(goda_valid_data, batch_size, shuffle=False)

    train_loss = np.zeros(epochs)
    valid_loss = np.zeros(epochs)
    for epoch in range(epochs):
        print(f'Epoche: {epoch+1}')
        # Training with SODAS-data
        model, loss = fit_model(model, optimizer, criterion, train_loader,
                                device)
        train_loss[epoch] = loss
        print(f'Soda \tTraining Loss: {loss}')

        # Validating with GODA
        model, loss = validate_model(model, criterion, valid_loader,
                                     device)
        valid_loss[epoch] = loss
        print(f'Goda \tValidation Loss: {loss}')
    
    return model, train_loss, valid_loss