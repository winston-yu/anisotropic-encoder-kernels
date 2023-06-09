import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split
import torch.optim as optim

import torchvision
from torchvision import transforms
import tqdm

import matplotlib.pyplot as plt

import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

torch.use_deterministic_algorithms(True) 
torch.backends.cudnn.deterministic = True

### ANISOTROPIC KERNELS ###

def asymmetric_kernel(r_batch, x_batch, jacobian, sigma_squared):
    """
    r_batch: BATCH of reference points (batch_size, 1, side_length, side_length)
    x_batch: BATCH of data points (for dimensions, see above)
    returns a matrix [a(r, x)]_{r \in r_batch, x \in x_batch}
    """ # NOTE: super inefficient
    out = torch.zeros(len(r_batch), len(x_batch))
    for ind, x in enumerate(x_batch): # make a constant vector of the same image; then do len(x_batch) times
        constant = torch.stack(len(r_batch)*[x], dim=0) # (batch_size, 1, side_length, side_length)
        power = torch.einsum('bdbonn, bonn -> bd', jacobian, constant - r_batch) # (batch_size, encoding_dim)
        power = (power ** 2).sum(axis=1) # batch_size
        out[:, ind] = power
    return torch.exp(- out / (2 * sigma_squared))

def anisotropic_mmd(x_batch, y_batch, r_batch, jacobian, sigma_squared):
    Ax = asymmetric_kernel(r_batch, x_batch, jacobian, sigma_squared=sigma_squared) # form "Gram" matrix of asymmetric kernel for X
    Ay = asymmetric_kernel(r_batch, y_batch, jacobian, sigma_squared=sigma_squared) # form "Gram" matrix of asymmetric kernel for Y
    mu_Xs = (Ax.sum(axis=1) / Ax.shape[1]) # calculate mean embeddings for X sample
    mu_Ys = (Ay.sum(axis=1) / Ay.shape[1]) # calculate mean embeddings for Y sample
    return ((mu_Xs - mu_Ys) ** 2).mean() # calculate MMD

def permutation_test_anisotropic_mmd(x_batch, y_batch, r_batch, encoder, n_perms, sigma_squared=1):
    nx, ny = x_batch.shape[0], y_batch.shape[0]
    jacobian = torch.autograd.functional.jacobian(encoder, r_batch).detach() # shape: (encoding_dim, input_dim) * input_dim
    
    observed_mmd = anisotropic_mmd(
        x_batch.detach(), 
        y_batch.detach(), 
        r_batch.detach(), 
        jacobian, 
        sigma_squared=sigma_squared
    )
    simulated_mmds = []
    xy_batch = torch.cat([x_batch, y_batch], dim=0).detach()
    for _ in tqdm.trange(n_perms):
        shuffled_idx = torch.randperm(nx + ny)
        xy_batch = xy_batch[shuffled_idx]
        shuffled_x_batch, shuffled_y_batch = xy_batch[:nx], xy_batch[nx:]
        simulated_mmds.append(anisotropic_mmd(shuffled_x_batch, shuffled_y_batch, r_batch, jacobian, sigma_squared=sigma_squared))
    return observed_mmd, simulated_mmds

### ENCODER KERNEL ###

def encoder_kernel(r_batch, x_batch, encoder_r_batch, encoder, sigma_squared):
    """
    r_batch: batch of reference points
    x_batch: batch of data points
    return: [(k \circ f)(r,x)]_{r \in r_batch, x \in x_batch}
    """
    out = torch.zeros(len(r_batch), len(x_batch))
    for ind, x in enumerate(x_batch):
        to_be_stacked = encoder(x.unsqueeze(0))
        stacked = torch.stack(len(r_batch)*[to_be_stacked], dim=0).squeeze()
        power = ((stacked - encoder_r_batch) ** 2).sum(axis=1)
        out[:, ind] = power
    return torch.exp(- out / (2 * sigma_squared)) 

def encoder_mmd(x_batch, y_batch, r_batch, encoder_r_batch, encoder, sigma_squared):
    Kfx = encoder_kernel(r_batch, x_batch, encoder_r_batch, encoder, sigma_squared) # form "Gram" matrix of encoder kernel for X
    Kfy = encoder_kernel(r_batch, y_batch, encoder_r_batch, encoder, sigma_squared) # form "Gram" matrix of encoder kernel for Y
    mu_Xs = (Kfx.sum(axis=1) / Kfx.shape[1]) # calculate mean embeddings for X sample # replace with .mean(axis=1)
    mu_Ys = (Kfy.sum(axis=1) / Kfy.shape[1]) # calculate mean embeddings for Y sample
    return ((mu_Xs - mu_Ys) ** 2).mean() # calculate MMD

def permutation_test_encoder_mmd(x_batch, y_batch, r_batch, encoder, n_perms, sigma_squared=1):
    nx, ny = x_batch.shape[0], y_batch.shape[0]
    encoder_r_batch = encoder(r_batch) # shape: (batch_size, encoding_dim)
    
    observed_mmd = encoder_mmd(
        x_batch.detach(), 
        y_batch.detach(), 
        r_batch.detach(), 
        encoder_r_batch.detach(), 
        encoder, 
        sigma_squared=sigma_squared
    )
    simulated_mmds = []
    xy_batch = torch.cat([x_batch, y_batch], dim=0).detach()
    for _ in tqdm.trange(n_perms):
        shuffled_idx = torch.randperm(nx + ny)
        xy_batch = xy_batch[shuffled_idx]
        simulated_mmds.append(encoder_mmd(xy_batch[:nx], xy_batch[nx:], r_batch, encoder_r_batch, encoder, sigma_squared=sigma_squared))
    return observed_mmd, simulated_mmds

### ISOTROPIC KERNEL ###

def isotropic_kernel(r_batch, x_batch, sigma_squared):
    """
    r_batch: BATCH of reference points (batch_size, 1, side_length, side_length)
    x_batch: BATCH of data points (for dimensions, see above)
    returns a matrix [a(r, x)]_{r \in r_batch, x \in x_batch}
    """
    out = torch.zeros(len(r_batch), len(x_batch))
    for ind, x in enumerate(x_batch):
        constant = torch.stack(len(r_batch)*[x], dim=0).squeeze() # (batch_size, side_length, side_length)
        power = ((constant - r_batch.squeeze()) ** 2).sum(axis=(2,1)) # batch_size # TODO: fix dimensions
        out[:, ind] = power
    return torch.exp(- out / (2 * sigma_squared))

def isotropic_mmd(x_batch, y_batch, r_batch, sigma_squared):
    Sx = isotropic_kernel(r_batch, x_batch, sigma_squared=sigma_squared) # form "Gram" matrix of isotropic_kernel kernel for X
    Sy = isotropic_kernel(r_batch, y_batch, sigma_squared=sigma_squared) # form "Gram" matrix of isotropic_kernel kernel for Y
    mu_Xs = (Sx.sum(axis=1) / Sx.shape[1]) # calculate mean embeddings for X sample
    mu_Ys = (Sy.sum(axis=1) / Sy.shape[1]) # calculate mean embeddings for Y sample
    return ((mu_Xs - mu_Ys) ** 2).mean() # calculate MMD

def permutation_test_isotropic_mmd(x_batch, y_batch, r_batch, n_perms, sigma_squared=1):
    nx, ny = x_batch.shape[0], y_batch.shape[0]
    
    observed_mmd = isotropic_mmd(
        x_batch.detach(), 
        y_batch.detach(), 
        r_batch.detach(), 
        sigma_squared=sigma_squared
    )
    simulated_mmds = []
    xy_batch = torch.cat([x_batch, y_batch], dim=0).detach()
    for _ in tqdm.trange(n_perms):
        shuffled_idx = torch.randperm(nx + ny)
        xy_batch = xy_batch[shuffled_idx]
        shuffled_x_batch, shuffled_y_batch = xy_batch[:nx], xy_batch[nx:]
        simulated_mmds.append(isotropic_mmd(shuffled_x_batch, shuffled_y_batch, r_batch, sigma_squared=sigma_squared))
    return observed_mmd, simulated_mmds