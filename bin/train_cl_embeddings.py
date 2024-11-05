#!/usr/bin/env python3

import argparse
import os
import gzip
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pytorch_metric_learning.losses import NTXentLoss
from tqdm.auto import tqdm

from decifer.cl_model import CLEncoder
from decifer.cl_utility import AugmentatedDeciferDataset

def main():
    parser = argparse.ArgumentParser(description='Train a contrastive learning encoder and save embeddings.')

    parser.add_argument('--data_file', type=str, required=True, help='Path to the input h5 data file.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer.')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Dimension of the embedding vector.')
    parser.add_argument('--proj_dim', type=int, default=32, help='Dimension of the projection vector.')
    parser.add_argument('--subset_size', type=int, default=None, help='Size of the subset to train on. If not specified, use the entire dataset.')
    parser.add_argument('--output_folder', type=str, default='conditioning_embeddings', help='Folder to save the embeddings.')
    parser.add_argument('--save_every', type=int, default=10, help='Print loss every N epochs.')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for NTXentLoss.')
    parser.add_argument('--qmin', type=float, default=0.0, help='Q-min for XRD calculation')
    parser.add_argument('--qmax', type=float, default=10.0, help='Q-max for XRD calculation')
    parser.add_argument('--qstep', type=float, default=0.01, help='Q-step for XRD calculation')
    parser.add_argument('--fwhm_range', nargs=2, type=float, default=[0.001, 0.5], help='Range for FWHM of peaks in XRD calculation')
    parser.add_argument('--noise_range', nargs=2, type=float, default=[0.001, 0.025], help='Range for additive noise to XRD calculation')
    parser.add_argument('--intensity_scale_range', nargs=2, type=float, default=[0.95,1.0], help='Intensity scaling range for augmentation.')
    parser.add_argument('--mask_prob', type=float, default=0.05, help='Mask probability for augmentation.')
    parser.add_argument('--load_model', type=str, default=None, help='Model path. Default is None. If provided, embeddings will be calculated using this model.')
    parser.add_argument('--tsne_plot', action='store_true', help='If set, generate a t-SNE plot instead of saving embeddings.')
    parser.add_argument('--num_augmentations', type=int, default=5, help='Number of augmentations per CIF for t-SNE plot.')
    parser.add_argument('--tsne_perplexity', type=float, default=30.0, help='Perplexity parameter for t-SNE.')
    parser.add_argument('--num_samples_tsne', type=int, default=10, help='Number of CIF samples to use in t-SNE plot.')
    parser.add_argument('--show_augmentations', action='store_true', help='If set, plot original and augmented XRD patterns for selected CIFs.')
    parser.add_argument('--num_cifs_to_show', type=int, default=3, help='Number of CIFs to show in augmentation comparison.')
    parser.add_argument('--no_train', action='store_true', help='If set, skip training and use an untrained model to generate embeddings.')
    parser.add_argument('--save_embeddings', action='store_true', help='If set, generate and save embeddings from model (if set).')

    args = parser.parse_args()

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define augmentation kwargs and xrd kwargs
    xrd_kwargs = {
        'qmin': args.qmin,
        'qmax': args.qmax,
        'qstep': args.qstep,
    }

    augmentation_kwargs = {
        'intensity_scale_range': tuple(args.intensity_scale_range),
        'fwhm_range': tuple(args.fwhm_range),
        'noise_range': tuple(args.noise_range),
        'mask_prob': args.mask_prob,
    }

    # Input dim
    args.input_dim = len(np.arange(args.qmin, args.qmax, args.qstep))

    # Load dataset
    dataset = AugmentatedDeciferDataset(args.data_file, ["xrd_disc.q", "xrd_disc.iq", "cif_name"], xrd_kwargs, augmentation_kwargs)
        
    # Make directory
    os.makedirs(args.output_folder, exist_ok=True)

    if args.load_model:

        checkpoint = torch.load(args.load_model)
        model_args = checkpoint["model_args"]
        # Initialize the encoder model
        xrd_encoder = CLEncoder(
            **model_args,
        )
        xrd_encoder.to(device)

        xrd_encoder.load_state_dict(checkpoint["model"])
    elif not args.no_train:
        # Proceed with training
        xrd_encoder = CLEncoder(
            input_dim = args.input_dim,
            embedding_dim=args.embedding_dim,
            proj_dim=args.proj_dim,
        )
        xrd_encoder.to(device)

        # Optimizer and loss function
        optimizer = optim.Adam(xrd_encoder.parameters(), lr=args.learning_rate)
        loss_function = NTXentLoss(temperature=args.temperature)
        # GradScaler
        scaler = torch.cuda.amp.GradScaler()

        # Create a subset of the dataset if specified
        if args.subset_size is not None and args.subset_size < len(dataset):
            subset_indices = np.random.choice(len(dataset), args.subset_size, replace=False)
            subset_dataset = Subset(dataset, subset_indices)
        else:
            subset_dataset = dataset

        # DataLoader
        dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True)

        # Training loop
        num_epochs = args.epochs

        for epoch in range(num_epochs):
            xrd_encoder.train()
            total_loss = 0
            total_samples = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
    
                #with torch.autograd.profiler.profile(use_cuda=True) as prof:
                aug1 = batch['aug1']
                aug2 = batch['aug2']
                cif_names = batch['cif_name']

                optimizer.zero_grad()

                # Forward pass
                with torch.cuda.amp.autocast():
                    _, _, z_i, z_j = xrd_encoder(aug1, aug2)

                    # Normalize embeddings
                    z_i = nn.functional.normalize(z_i, dim=1)
                    z_j = nn.functional.normalize(z_j, dim=1)

                    # Prepare embeddings and labels
                    embeddings = torch.cat([z_i, z_j], dim=0)
                    batch_size = z_i.size(0)
                    labels = torch.arange(batch_size).repeat(2).to(device)

                    # Compute loss
                    loss = loss_function(embeddings, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * batch_size
                total_samples += batch_size

            avg_loss = total_loss / total_samples
            if (epoch+1) % args.save_every == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

                # Save model
                checkpoint = {
                    "model": xrd_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": {
                        "input_dim": args.input_dim, 
                        "embedding_dim": args.embedding_dim,
                        "proj_dim": args.proj_dim,
                    },
                    "xrd_kwargs": {
                        "qmin": args.qmin,
                        "qmax": args.qmax,
                        "qstep": args.qstep,
                    },
                    "aug_kwargs": {
                        "fwhm_range": tuple(args.fwhm_range),
                        "noise_range": tuple(args.noise_range),
                        "intensity_scale_range": tuple(args.intensity_scale_range),
                        "mask_prob": args.mask_prob,
                    },
                }
                print(f"Saving checkpoint to {args.output_folder}...", flush=True)
                torch.save(checkpoint, os.path.join(args.output_folder, "ckpt.pt"))
                #print(prof.key_averages().table(sort_by="cuda_time_total"))
                
    else:
        # Proceed with training
        xrd_encoder = CLEncoder(
            input_dim = len(np.arange(args.qmin, args.qmax, args.qstep)),
            embedding_dim=args.embedding_dim,
            proj_dim=args.proj_dim,
        )
        xrd_encoder.to(device)
        # If --no_train is set and no model is loaded, we'll use the untrained model
        print("Using untrained model to generate embeddings.")

    # Decide whether to generate embeddings, t-SNE plot, or show augmentations
    if args.save_embeddings:
        # Save embeddings
        embeddings_output_folder = os.path.join(args.output_folder, "embeddings")
        os.makedirs(embeddings_output_folder, exist_ok=True)

        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Saving embeddings'):
                aug1 = batch['aug1']
                aug2 = batch['aug2']
                cif_names = batch['cif_name']

                h, _ = xrd_encoder(aug1, aug2, train=False)  # Get encoder embeddings h
                h = h.cpu().numpy()

                for i in range(h.shape[0]):
                    cif_name = cif_names[i]
                    embedding = h[i]
                    output_file = os.path.join(embeddings_output_folder, f"{cif_name}.pkl.gz")
                    with gzip.open(output_file, 'wb') as f:
                        pickle.dump(embedding, f)

if __name__ == '__main__':
    main()
