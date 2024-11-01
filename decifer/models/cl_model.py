#!/usr/bin/env python3

import sys
sys.path.append("./")
import os
import torch
import torch.nn as nn
import numpy as np

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
