#!/usr/bin/env python3

import os
import torch
import argparse
import numpy as np

from decifer.cl_utility import (
    AugmentatedDeciferDataset,
    generate_tsne_plot,
    show_augmentations,
)
from decifer.cl_model import CLEncoder
from decifer.decifer_dataset import DeciferDataset
    

def main():
    parser = argparse.ArgumentParser(description='Show/Save statistics on CL model.')
    
    parser.add_argument('--data_file', type=str, required=True, help='Path to the input h5 data file.')
    parser.add_argument('--model', type=str, default=None, help='Model path. Default is None. If provided, embeddings will be calculated using this model.')
    parser.add_argument('--output_folder', type=str, default='cl_embeddings', help='Folder to save the embeddings.')
    parser.add_argument('--embedding_dim', type=int, default=512, help='')
    parser.add_argument('--proj_dim', type=int, default=8, help='')

    parser.add_argument('--qmin', type=float, default=0.0, help='Q-min for XRD calculation')
    parser.add_argument('--qmax', type=float, default=10.0, help='Q-max for XRD calculation')
    parser.add_argument('--qstep', type=float, default=0.01, help='Q-step for XRD calculation')
    parser.add_argument('--fwhm_range', nargs=2, type=float, default=[0.001, 0.5], help='Range for FWHM of peaks in XRD calculation')
    parser.add_argument('--noise_range', nargs=2, type=float, default=[0.001, 0.025], help='Range for additive noise to XRD calculation')
    parser.add_argument('--intensity_scale_range', nargs=2, type=float, default=[0.95,1.0], help='Intensity scaling range for augmentation.')
    parser.add_argument('--mask_prob', type=float, default=0.05, help='Mask probability for augmentation.')

    parser.add_argument('--tsne_plot', action='store_true', help='If set, generate a t-SNE plot instead of saving embeddings.')
    parser.add_argument('--num_augmentations', type=int, default=5, help='Number of augmentations per CIF for t-SNE plot.')
    parser.add_argument('--tsne_perplexity', type=float, default=30.0, help='Perplexity parameter for t-SNE.')
    parser.add_argument('--num_samples_tsne', type=int, default=100, help='Number of CIF samples to use in t-SNE plot.')

    parser.add_argument('--show_augmentations', action='store_true', help='If set, plot original and augmented XRD patterns for selected CIFs.')
    parser.add_argument('--num_cifs_to_show', type=int, default=3, help='Number of CIFs to show in augmentation comparison.')

    parser.add_argument('--gen_emb', action='store_true')
    parser.add_argument('--emb_path', type=str, default='./embeddings.npz', help='Path to embeddings npz. Default None')
    
    args = parser.parse_args()

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if args.model is not None:
        checkpoint = torch.load(args.model)
        model_args = checkpoint["model_args"]
        # Initialize the encoder model
        xrd_encoder = CLEncoder(
            **model_args,
        )
        xrd_encoder.to(device)

        xrd_encoder.load_state_dict(checkpoint["model"])
    else:
        xrd_encoder = CLEncoder(
            input_dim = len(np.arange(args.qmin, args.qmax, args.qstep)),
            embedding_dim=args.embedding_dim,
            proj_dim=args.proj_dim,
        )
        xrd_encoder.to(device)
    
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
    
    # Load dataset
    # dataset = AugmentatedDeciferDataset(args.data_file, ["xrd_disc.q", "xrd_disc.iq", "cif_name"], xrd_kwargs, augmentation_kwargs)
    dataset = DeciferDataset(args.data_file, ["xrd_disc.q", "xrd_disc.iq", "cif_name", "spacegroup"])
    
    # Make directory
    os.makedirs(args.output_folder, exist_ok=True)

    if args.tsne_plot:
        generate_tsne_plot(xrd_encoder, dataset, args)
    if args.show_augmentations:
        show_augmentations(xrd_encoder, dataset, args)

if __name__ == "__main__":
    main()

