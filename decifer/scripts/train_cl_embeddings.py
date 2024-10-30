#!/usr/bin/env python3

import sys
sys.path.append("./")

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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from decifer import (
    space_group_symbol_to_number,
    extract_space_group_symbol,
    DeciferDataset,
    CLEncoder,
    augment_xrd,
)
        
def collate_fn(batch):
    # Seperate fields from the batch
    xrd_q = [sample[0] for sample in batch]
    xrd_iq = [sample[1] for sample in batch]
    cif_names = [sample[2] for sample in batch]
    spacegroup_symbols = [sample[3] for sample in batch]

    # Determine max lengths 
    max_len_q = max(len(x) for x in xrd_q)
    max_len_iq = max(len(x) for x in xrd_iq)

    # Pad 
    padded_q = [np.pad(x, (0, max_len_q - len(x)), mode='constant') for x in xrd_q]
    padded_iq = [np.pad(x, (0, max_len_iq - len(x)), mode='constant') for x in xrd_iq]

    # Convert to numpy for batching
    batch_q = np.array(padded_q, dtype=np.float32)
    batch_iq = np.array(padded_iq, dtype=np.float32)

    # Return
    return batch_q, batch_iq, cif_names, spacegroup_symbols

def generate_tsne_plot(xrd_encoder, dataset, device, args):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    xrd_encoder.eval()

    # Choose a subset of the dataset for visualization
    num_samples = min(len(dataset), args.num_samples_tsne)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset_dataset = Subset(dataset, indices)
    data_loader = DataLoader(subset_dataset, batch_size=num_samples, shuffle=False, collate_fn=collate_fn)

    # Collect embeddings and labels
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Generating t-SNE plot'):
            *data, cif_names, spacegroup_symbols = batch
                
            for _ in range(args.num_augmentations):

                # Generate augmented embeddings
                h, _, _, _ = xrd_encoder(data, train=True)  # Get encoder embeddings h
                h = h.cpu().numpy()

                all_embeddings.extend(h)
                all_labels.extend(spacegroup_symbols)
                

    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Plotting
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(all_labels)
    num_classes = len(unique_labels)
    max_colors = 20  # Maximum colors in 'tab20' colormap
    if num_classes > max_colors:
        print(f"Number of classes ({num_classes}) exceeds maximum colors ({max_colors}). Some colors will be reused.")
    colors = plt.cm.get_cmap('tab20', max_colors)

    markers = ['o', '^', 'D', 's', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_', '.', ',', '1', '2', '3', '4']
    if num_classes > len(markers):
        print(f"Number of classes ({num_classes}) exceeds number of markers ({len(markers)}). Some markers will be reused.")

    for idx, label in enumerate(unique_labels):
        idxs = np.where(all_labels == label)
        plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], label=label, alpha=0.7,
                    color=colors(idx % max_colors), marker=markers[idx % len(markers)])

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.title('t-SNE plot of embeddings with multiple augmentations, colored by Spacegroup')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_augmentations(xrd_encoder, dataset, args):

    # Choose a few CIFs to show
    num_cifs_to_show = args.num_cifs_to_show
    indices = np.random.choice(len(dataset), num_cifs_to_show, replace=False)
    subset_dataset = Subset(dataset, indices)
    data_loader = DataLoader(subset_dataset, batch_size=num_cifs_to_show, shuffle=False, collate_fn=collate_fn)

    xrd_encoder.eval()

    with torch.no_grad():
        for batch in data_loader:
            *data, cif_names, spacegroup_symbols = batch

            # Generate original XRD patterns
            original_patterns = generate_xrd_patterns(
                data,
                qmin=xrd_encoder.qmin,
                qmax=xrd_encoder.qmax,
                qstep=xrd_encoder.qstep,
                fwhm=0.01  # Small FWHM for clear peaks
            ).cpu().numpy()

            # Generate augmented XRD patterns
            augmented_patterns = augment_xrd(
                data,
                **xrd_encoder.aug_kwargs
            ).cpu().numpy()

            q_values = xrd_encoder.qs

            # Plotting
            for i in range(num_cifs_to_show):
                plt.figure(figsize=(12, 6))
                plt.plot(q_values, original_patterns[i], label='Original', linewidth=2)
                plt.plot(q_values, augmented_patterns[i], label='Augmented', linewidth=1)
                plt.title(f'Original vs Augmented XRD Pattern for CIF: {cif_names[i]}')
                plt.xlabel('Q (1/Å)')
                plt.ylabel('Intensity (a.u.)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

    print("Displayed original and augmented XRD patterns for selected CIFs.")

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

    # Load dataset
    dataset = DeciferDataset(args.data_file, ["xrd_disc.q", "xrd_disc.iq", "cif_name", "spacegroup"])
        
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
            embedding_dim=args.embedding_dim,
            proj_dim=args.proj_dim,
            qmin=args.qmin,
            qmax=args.qmax,
            qstep=args.qstep,
            fwhm_range=tuple(args.fwhm_range),
            noise_range=tuple(args.noise_range),
            intensity_scale_range=tuple(args.intensity_scale_range),
            mask_prob=args.mask_prob
        )
        xrd_encoder.to(device)

        # Optimizer and loss function
        optimizer = optim.Adam(xrd_encoder.parameters(), lr=args.learning_rate)
        loss_function = NTXentLoss(temperature=args.temperature)

        # Create a subset of the dataset if specified
        if args.subset_size is not None and args.subset_size < len(dataset):
            subset_indices = np.random.choice(len(dataset), args.subset_size, replace=False)
            subset_dataset = Subset(dataset, subset_indices)
        else:
            subset_dataset = dataset

        # DataLoader
        dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        # Training loop
        num_epochs = args.epochs

        for epoch in range(num_epochs):
            xrd_encoder.train()
            total_loss = 0
            total_samples = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                *data, cif_names, spacegroup_symbols = batch

                optimizer.zero_grad()

                # Forward pass
                _, _, z_i, z_j = xrd_encoder(data)

                # Normalize embeddings
                z_i = nn.functional.normalize(z_i, dim=1)
                z_j = nn.functional.normalize(z_j, dim=1)

                # Prepare embeddings and labels
                embeddings = torch.cat([z_i, z_j], dim=0)
                batch_size = z_i.size(0)
                labels = torch.arange(batch_size).repeat(2).to(device)

                # Compute loss
                loss = loss_function(embeddings, labels)
                loss.backward()
                optimizer.step()

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
                        "embedding_dim": args.embedding_dim,
                        "proj_dim": args.proj_dim,
                        "qmin": args.qmin,
                        "qmax": args.qmax,
                        "qstep": args.qstep,
                        "fwhm_range": tuple(args.fwhm_range),
                        "noise_range": tuple(args.noise_range),
                        "intensity_scale_range": tuple(args.intensity_scale_range),
                        "mask_prob": args.mask_prob,
                    }
                }
                print(f"Saving checkpoint to {args.output_folder}...", flush=True)
                torch.save(checkpoint, os.path.join(args.output_folder, "ckpt.pt"))
                
    else:
        # Proceed with training
        xrd_encoder = CLEncoder(
            embedding_dim=args.embedding_dim,
            proj_dim=args.proj_dim,
            qmin=args.qmin,
            qmax=args.qmax,
            qstep=args.qstep,
            fwhm_range=tuple(args.fwhm_range),
            noise_range=tuple(args.noise_range),
            intensity_scale_range=tuple(args.intensity_scale_range),
            mask_prob=args.mask_prob
        )
        xrd_encoder.to(device)
        # If --no_train is set and no model is loaded, we'll use the untrained model
        print("Using untrained model to generate embeddings.")

    # Decide whether to generate embeddings, t-SNE plot, or show augmentations
    if args.show_augmentations:
        show_augmentations(xrd_encoder, dataset, args)
    elif args.tsne_plot:
        generate_tsne_plot(xrd_encoder, dataset, device, args)
    elif args.save_embeddings:
        # Save embeddings
        embeddings_output_folder = os.path.join(args.output_folder, "embeddings")
        os.makedirs(embeddings_output_folder, exist_ok=True)

        # Embed conditioning
        xrd_encoder.eval()
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Saving embeddings'):
                *data, cif_names, spacegroup_symbols = batch
                h, _ = xrd_encoder(data, train=False)  # Get encoder embeddings h
                h = h.cpu().numpy()

                for i in range(h.shape[0]):
                    cif_name = cif_names[i]
                    embedding = h[i]
                    output_file = os.path.join(embeddings_output_folder, f"{cif_name}.pkl.gz")
                    with gzip.open(output_file, 'wb') as f:
                        pickle.dump(embedding, f)
    else:
        # Do nothing
        pass

if __name__ == '__main__':
    main()
