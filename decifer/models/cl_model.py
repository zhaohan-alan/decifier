#!/usr/bin/env python3

import sys
sys.path.append("./")
import os
import torch
import torch.nn as nn
import numpy as np

def augment_xrd(
    batch_q,
    batch_iq,
    qmin = 0.0,
    qmax = 10.0,
    qstep = 0.01,
    fwhm_range = (0.001, 0.5),
    noise_range = (0.001, 0.025),
    intensity_scale_range = (0.95, 1.0),
    mask_prob = 0.1,
):
    # Define the continuous Q grid
    q_cont = np.arange(qmin, qmax, qstep)
    augmented_patterns = []
        
    for q, iq in zip(batch_q, batch_iq):
        # Initialize intensities_continuous
        iq_cont = np.zeros_like(q_cont)
        
        # Sample a random FWHM from fwhm_range and convert to standard deviation
        fwhm = np.random.uniform(*fwhm_range)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        
        # Apply Gaussian broadening to the peaks
        for q_peak, iq_peak in zip(q, iq):
            if q_peak != 0:
                gaussian_broadening = iq_peak * np.exp(-0.5 * ((q_cont - q_peak) / sigma) ** 2)
                iq_cont += gaussian_broadening
        
        # Normalize the continuous intensities
        iq_cont /= (np.max(iq_cont) + 1e-16)
        
        # Random scaling of intensities
        if intensity_scale_range is not None:
            intensity_scale = np.random.uniform(*intensity_scale_range)
            iq_cont = iq_cont * intensity_scale
        
        # Random noise addition
        if noise_range is not None:
            noise_scale = np.random.uniform(*noise_range)
            background = np.random.randn(len(iq_cont)) * noise_scale
            iq_cont = iq_cont + background
        
        # Random masking
        if mask_prob is not None:
            mask = np.random.rand(len(iq_cont)) > mask_prob
            iq_cont = iq_cont * mask
        
        # Clipping
        iq_cont = np.clip(iq_cont, a_min=0.0, a_max=None)
        
        # Append to list
        augmented_patterns.append(iq_cont.astype(np.float32))
        
    # Stack all augmented patterns into a numpy array
    xrd_augmented_batch = np.stack(augmented_patterns)
    
    # Convert to torch tensor
    xrd_augmented_batch = torch.from_numpy(xrd_augmented_batch)
    
    return xrd_augmented_batch

def cont_xrd_from_disc(
    batch_q,
    batch_iq,
    qmin=0.0,
    qmax=10.0,
    qstep=0.01,
    fwhm=0.01  # A default small FWHM value for consistent broadening
):
    # Define the continuous Q grid
    q_continuous = np.arange(qmin, qmax, qstep)
    num_q_points = len(q_continuous)
    
    xrd_patterns = []
    
    for q, iq in zip(batch_q, batch_iq):
        # Initialize intensities_continuous
        intensities_continuous = np.zeros_like(q_continuous)
        
        # Convert FWHM to standard deviation
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        
        # Apply Gaussian broadening to the peaks
        for q_peak, intensity in zip(q, iq):
            if q_peak != 0:
                gaussian_broadening = intensity * np.exp(-0.5 * ((q_continuous - q_peak) / sigma) ** 2)
                intensities_continuous += gaussian_broadening
        
        # Normalize the continuous intensities
        intensities_continuous /= (np.max(intensities_continuous) + 1e-16)
        
        # Append to list
        xrd_patterns.append(intensities_continuous.astype(np.float32))
    
    # Stack all patterns into a numpy array
    xrd_batch = np.stack(xrd_patterns)
    
    # Convert to torch tensor
    xrd_batch = torch.from_numpy(xrd_batch)
    
    return xrd_batch

# Model definitions
class CLEncoder(nn.Module):
    def __init__(
        self, 
        embedding_dim = 512, 
        proj_dim = 32, 
        qmin = 0.0,
        qmax = 10.0,
        qstep = 0.01,
        fwhm_range = (0.001, 0.5),
        noise_range = (0.001, 0.025),
        intensity_scale_range = (0.95, 1.0), 
        mask_prob = 0.1,
    ):
        super(CLEncoder, self).__init__()
        # XRD parameters
        self.qmin = qmin
        self.qmax = qmax
        self.qstep = qstep

        # Input dim
        self.qs = np.arange(qmin, qmax, qstep)
        input_dim = len(self.qs)

        # Encoder head
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, proj_dim)
        )
        # Augmentation parameters
        self.intensity_scale_range = intensity_scale_range
        self.mask_prob = mask_prob
        self.fwhm_range = fwhm_range
        self.noise_range = noise_range

        self.aug_kwargs = {
            'intensity_scale_range': intensity_scale_range,
            'mask_prob': mask_prob,
            'qmin': qmin,
            'qmax': qmax,
            'qstep': qstep,
            'fwhm_range': fwhm_range,
            'noise_range': noise_range,
        }

    def forward(self, data, train=True):
        if train:
            # Get two augmentations of the same batch
            augm1 = augment_xrd(*data, **self.aug_kwargs).to(next(self.enc.parameters()).device)
            augm2 = augment_xrd(*data, **self.aug_kwargs).to(next(self.enc.parameters()).device)
        
            # Extract encoder embeddings
            h_1 = self.enc(augm1)
            h_2 = self.enc(augm2)

            # Extract low-dimensional embeddings
            h_1_latent = self.proj(h_1)
            h_2_latent = self.proj(h_2)
            return h_1, h_2, h_1_latent, h_2_latent
        else:
            # Generate XRD patterns without augmentation
            xrd_batch = augment_xrd(
                *data,
                qmin=self.qmin,
                qmax=self.qmax,
                qstep=self.qstep,
                fwhm_range=(0.01,0.01),
                noise_range = None,
                intensity_scale_range = None,
                mask_prob = None,
            ).to(next(self.enc.parameters()).device)

            # Extract encoder embeddings
            h = self.enc(xrd_batch)
            h_latent = self.proj(h)
            return h, h_latent
