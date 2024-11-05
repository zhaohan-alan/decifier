#!/usr/bin/env python3

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from tqdm.auto import tqdm

from decifer.decifer_dataset import DeciferDataset
from decifer.utility import disc_to_cont_xrd

class AugmentatedDeciferDataset(DeciferDataset):
    def __init__(self, h5_path, data_keys, xrd_kwargs, augmentation_kwargs=None):
        super().__init__(h5_path, data_keys)

        self.xrd_kwargs = xrd_kwargs
        self.aug_kwargs = augmentation_kwargs if augmentation_kwargs is not None else {}

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        # Perform augmentation here using self.aug_kwargs
        sample['aug1'] = disc_to_cont_xrd(
            sample['xrd_disc.q'].unsqueeze(0),
            sample['xrd_disc.iq'].unsqueeze(0),
            **self.aug_kwargs,
        )['iq'].squeeze(0).to('cuda')
        sample['aug2'] = disc_to_cont_xrd(
            sample['xrd_disc.q'].unsqueeze(0),
            sample['xrd_disc.iq'].unsqueeze(0),
            **self.aug_kwargs,
        )['iq'].squeeze(0).to('cuda')

        out = {
            'aug1': sample['aug1'],
            'aug2': sample['aug2'],
            'cif_name': sample['cif_name'],
        }

        return out

def collate_fn(batch):

    # Seperate fields from the batch
    xrd_q = [torch.tensor(sample['xrd_disc.iq'], dtype=torch.float32) for sample in batch]
    xrd_iq = [torch.tensor(sample['xrd_disc.q'], dtype=torch.float32) for sample in batch]
    cif_names = [sample['cif_name'] for sample in batch]
    spacegroups = [sample['spacegroup'] for sample in batch]

    # Pad
    batch_q = pad_sequence(xrd_q, batch_first=True)
    batch_iq = pad_sequence(xrd_iq, batch_first=True)

    # Return
    return batch_q, batch_iq, cif_names, spacegroups

def save_embeddings(filename, embeddings, labels=None):
    if labels is not None:
        np.savez(filename, embeddings=embeddings, labels=labels)
    else:
        np.savez(filename, embeddings=embeddings)

    print(f"Embeddings and labels saved to {filename}.")

def load_embeddings(filename):
    data = np.load(filename)
    embeddings = data['embeddings']
    labels = data['labels'] if 'labels' in data else None
    print(f"Embeddings{' and labels' if labels is not None else ''} loaded from {filename}.")
    return embeddings, labels

def normalize_embeddings_minmax(embeddings_2d, feature_range=(0,1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    normalize_embeddings = scaler.fit_transform(embeddings_2d)
    return normalize_embeddings

def generate_tsne_plot(xrd_encoder, dataset, args):

    # Set seed
    np.random.seed(42)
    torch.manual_seed(42)

    if args.gen_emb:

        xrd_encoder.eval()

        # Choose a subset of the dataset for visualization
        num_samples = min(len(dataset), args.num_samples_tsne)
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset_dataset = Subset(dataset, indices)
        data_loader = DataLoader(subset_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)

        # Collect embeddings and labels
        all_embeddings = []
        all_labels = []

        with torch.no_grad():

            pbar = tqdm(total=len(data_loader)*args.num_augmentations, desc='Generating t-SNE plot')

            for batch in data_loader:

                batch_q, batch_iq, cif_names, spacegroups = batch

                for _ in range(args.num_augmentations):

                    # Make manual augmentations
                    aug = disc_to_cont_xrd(
                        batch_q,
                        batch_iq,
                        qmin = args.qmin,
                        qmax = args.qmax,
                        qstep = args.qstep,
                        fwhm_range= tuple(args.fwhm_range),
                        noise_range = tuple(args.noise_range),
                        mask_prob = args.mask_prob,
                    )['iq'].to('cuda')

                    h = xrd_encoder.enc(aug).cpu().numpy()
                    all_embeddings.extend(h)
                    all_labels.extend(spacegroups)

                    pbar.update(1)

            pbar.close()

        all_embeddings = np.array(all_embeddings)
        all_labels = np.array(all_labels)
    
        save_embeddings(args.emb_path, all_embeddings, all_labels)
    else:
        all_embeddings, all_labels = load_embeddings(args.emb_path)

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    embeddings_2d = normalize_embeddings_minmax(embeddings_2d, feature_range=(-1, 1))

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
    plt.title('t-SNE plot of embeddings with multiple augmentations')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Perform PCA
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    embeddings_2d = normalize_embeddings_minmax(embeddings_2d, feature_range=(-1, 1))

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
    plt.title('t-SNE plot of embeddings with multiple augmentations')
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
    data_loader = DataLoader(subset_dataset, batch_size=num_cifs_to_show, shuffle=False)

    xrd_encoder.eval()

    with torch.no_grad():
        for batch in data_loader:
            batch_q, batch_iq, cif_names, _ = batch

            # Generate original XRD patterns
            original_patterns = disc_to_cont_xrd(
                batch_q,
                batch_iq,
                qmin=xrd_encoder.qmin,
                qmax=xrd_encoder.qmax,
                qstep=xrd_encoder.qstep,
                fwhm_range=(0.01, 0.01)  # Small FWHM for clear peaks
            )['iq'].cpu().numpy()

            # Generate augmented XRD patterns
            augmented_patterns = disc_to_cont_xrd(
                batch_q,
                batch_iq,
                **xrd_encoder.aug_kwargs
            )['iq'].cpu().numpy()

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
