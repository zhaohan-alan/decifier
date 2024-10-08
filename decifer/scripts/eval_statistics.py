#!/usr/bin/env python
# coding: utf-8

"""
Script to process .eval files, perform dimensionality reduction, visualize data,
and save plots to a user-defined directory.

Usage:
    python script_name.py -i file1.eval file2.eval ... -o /path/to/output/directory
"""

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pymatgen.symmetry.groups import SpaceGroup
from tqdm.auto import tqdm
from pymatgen.core import Element
import warnings

# Suppress RuntimeWarnings from numpy
warnings.filterwarnings("ignore", category=RuntimeWarning)

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        args (Namespace): Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process .eval files and visualize data.')
    parser.add_argument('-i', '--input_files', nargs='+', required=True,
                        help='List of .eval files to process.')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Directory to save output plots.')
    parser.add_argument('--remove_outliers', action='store_true',
                        help='Remove outliers from the data.')
    parser.add_argument('--outlier_method', default='zscore', choices=['zscore', 'iqr'],
                        help='Method to use for outlier detection.')
    parser.add_argument('--outlier_thresh', type=float, default=2.0,
                        help='Threshold for outlier detection.')
    args = parser.parse_args()
    return args

def process_descriptor(descriptor_series):
    """
    Process variable-length descriptors into fixed-length representations.

    Parameters:
        descriptor_series (pd.Series): Series containing descriptors (arrays of variable length).

    Returns:
        features (np.ndarray): 2D array of shape (n_samples, fixed_length).
    """
    processed_features = []
    for descriptor in descriptor_series:
        descriptor = np.array(descriptor)
        # Compute summary statistics
        mean = np.mean(descriptor)
        std = np.std(descriptor)
        min_val = np.min(descriptor)
        max_val = np.max(descriptor)
        # Concatenate into a fixed-length vector
        feature_vector = np.array([mean, std, min_val, max_val])
        processed_features.append(feature_vector)
    features = np.vstack(processed_features)
    return features

def load_and_extract_features(file_list):
    """
    Load .eval files and extract relevant features.

    Parameters:
        file_list (list): List of file paths to .eval files.

    Returns:
        combined_df (pd.DataFrame): Combined DataFrame with relevant features.
    """
    combined_data = []
    columns_to_read = [
        'descriptors.soap_gen',
        'descriptors.acsf_gen',
        'descriptors.xrd_gen.iq',
        'dataset_name',
        'spacegroup',
        'seq_len',
        'species',
        'cell_params.a',
        'cell_params.b',
        'cell_params.c',
        'cell_params.alpha',
        'cell_params.beta',
        'cell_params.gamma',
        'cell_params.gen_vol',
        'cell_params.implied_vol',
        'validity.formula',
        'validity.spacegroup',
        'validity.bond_length',
        'validity.site_multiplicity',
    ]

    for file in tqdm(file_list, desc='Processing files'):
        df = pd.read_parquet(file, columns=columns_to_read)

        # Convert feature columns to numpy arrays
        for col in ['descriptors.soap_gen', 'descriptors.acsf_gen', 'descriptors.xrd_gen.iq']:
            df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

        combined_data.append(df)

    try:
        combined_df = pd.concat(combined_data, ignore_index=True)
    except ValueError as e:
        print(f"Error concatenating data frames: {e}")
        return pd.DataFrame()

    # Map 'spacegroup' symbols to integer numbers
    unique_spacegroups = combined_df['spacegroup'].unique()
    spacegroup_mapping = {}
    for sg in unique_spacegroups:
        spacegroup_mapping[sg] = get_spacegroup_int(sg)
    combined_df['spacegroup_int'] = combined_df['spacegroup'].map(spacegroup_mapping)

    return combined_df

def get_spacegroup_int(sg):
    """
    Get spacegroup integer number with exception handling.

    Parameters:
        sg (str): Spacegroup symbol.

    Returns:
        int_number (int): Corresponding integer number or NaN if not found.
    """
    try:
        return SpaceGroup(sg).int_number
    except:
        return np.nan

def get_feature_matrices(df):
    """
    Extract individual feature matrices for SOAP, ACSF, XRD, and combined features.

    Parameters:
        df (pd.DataFrame): DataFrame containing feature columns.

    Returns:
        feature_dict (dict): Dictionary containing individual and combined feature matrices.
        filtered_df (pd.DataFrame): DataFrame after filtering out NaN values.
    """
    # Filter out rows where any of the descriptor columns are None or NaN
    filtered_df = df.dropna(subset=['descriptors.soap_gen', 'descriptors.acsf_gen', 'descriptors.xrd_gen.iq'])

    if len(filtered_df) == 0:
        raise ValueError("No data left after filtering for NaN values.")

    # Process each descriptor column into fixed-length features
    soap_features_summary = process_descriptor(filtered_df['descriptors.soap_gen'])
    soap_features_full = np.vstack(filtered_df['descriptors.soap_gen'].values)

    acsf_features_summary = process_descriptor(filtered_df['descriptors.acsf_gen'])
    acsf_features_full = np.vstack(filtered_df['descriptors.acsf_gen'].values)

    xrd_features_summary = process_descriptor(filtered_df['descriptors.xrd_gen.iq'])
    xrd_features_full = np.vstack(filtered_df['descriptors.xrd_gen.iq'].values)

    # Concatenate summary features for combined features
    combined_features = np.hstack([soap_features_summary, acsf_features_summary, xrd_features_summary])

    feature_dict = {
        'SOAP_Summary': soap_features_summary,
        'SOAP_Full': soap_features_full,
        'ACSF_Summary': acsf_features_summary,
        'ACSF_Full': acsf_features_full,
        'XRD_Summary': xrd_features_summary,
        'XRD_Full': xrd_features_full,
        'Combined': combined_features
    }

    return feature_dict, filtered_df

def perform_dimensionality_reduction(features, n_components=2):
    """
    Perform dimensionality reduction on the feature matrix.

    Parameters:
        features (np.ndarray): The feature matrix.
        n_components (int): Number of components for PCA.

    Returns:
        reduced_features (np.ndarray): Reduced feature matrix.
        pca (PCA): Trained PCA model.
        scaler (StandardScaler): Fitted scaler object.
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features_scaled)

    return reduced_features, pca, scaler

def filter_outliers_on_reduced_features(reduced_features, method='zscore', thresh=3.0):
    """
    Filter outliers based on the reduced features.

    Parameters:
        reduced_features (np.ndarray): Reduced feature matrix (e.g., PCA components).
        method (str): Method to use for outlier detection ('zscore' or 'iqr').
        thresh (float): Threshold for outlier detection.

    Returns:
        mask (np.ndarray): Boolean mask indicating which samples to keep.
    """
    if method == 'zscore':
        from scipy.stats import zscore
        z_scores = np.abs(zscore(reduced_features))
        mask = (z_scores < thresh).all(axis=1)
    elif method == 'iqr':
        Q1 = np.percentile(reduced_features, 25, axis=0)
        Q3 = np.percentile(reduced_features, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - thresh * IQR
        upper_bound = Q3 + thresh * IQR
        mask = np.all((reduced_features >= lower_bound) & (reduced_features <= upper_bound), axis=1)
    else:
        raise ValueError("Invalid method for outlier detection. Choose 'zscore' or 'iqr'.")

    return mask

def visualize_samples(reduced_features_dict, df_filtered_dict, output_dir):
    """
    Visualize the samples after dimensionality reduction and save plots.

    Parameters:
        reduced_features_dict (dict): Dictionary of reduced feature matrices.
        df_filtered_dict (dict): Dictionary of filtered DataFrames for each descriptor.
        output_dir (str): Directory to save output plots.
    """
    sns.set(style='whitegrid')
    descriptors = ['SOAP_Summary', 'SOAP_Full', 'ACSF_Summary', 'ACSF_Full', 'XRD_Summary', 'XRD_Full', 'Combined']

    # Adjust font sizes
    plt.rcParams.update({'font.size': 10})

    for idx, descriptor in enumerate(descriptors):
        reduced_features = reduced_features_dict[descriptor]
        df_filtered = df_filtered_dict[descriptor]

        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'PC1': reduced_features[:, 0],
            'PC2': reduced_features[:, 1],
            'Dataset': df_filtered['dataset_name'].values,
            'Spacegroup': df_filtered['spacegroup_int'].values,
            'Sequence Length': df_filtered['seq_len'].values
        })

        # Apply markers for 'Dataset'
        unique_datasets = plot_df['Dataset'].unique()
        markers = ['o', '^', 'D', 'v', 'P', '*', 's', 'h', '+']  # Extend as needed

        plt.figure(figsize=(10, 8), dpi=300)
        ax = plt.gca()
        for i, dataset in enumerate(unique_datasets):
            idxs = plot_df['Dataset'] == dataset
            ax.scatter(
                plot_df.loc[idxs, 'PC1'],
                plot_df.loc[idxs, 'PC2'],
                marker=markers[i % len(markers)],
                edgecolor='k',
                alpha=0.7,
                label=dataset
            )

        ax.set_title(f"{descriptor} Descriptor", fontsize=12)
        ax.set_xlabel('Principal Component 1', fontsize=10)
        ax.set_ylabel('Principal Component 2', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(alpha=0.5)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title='Dataset', fontsize=8)

        # Save the plot
        filename = f"{descriptor}_PCA.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_histograms(stat_dict, output_dir):
    """
    Plot histograms for species, spacegroup, sequence length, and cell parameters.

    Parameters:
        stat_dict (dict): Dictionary containing statistical data.
        output_dir (str): Directory to save output plots.
    """
    # Collect all species, spacegroup, seq_len, and cell_params data
    all_species = []
    all_spacegroups = []
    all_seq_len = []
    cell_params = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'implied_vol', 'gen_vol']
    all_cell_params = {param: [] for param in cell_params}
    dataset_labels_species = []
    dataset_labels_spacegroups = []
    dataset_labels_seq_len = []
    dataset_labels_cell_params = {param: [] for param in cell_params}

    for dataset_name, fields in stat_dict.items():
        all_species.extend(fields['species'])
        all_spacegroups.extend(fields['spacegroup'])
        all_seq_len.extend(fields['seq_len'])
        dataset_labels_species.extend([dataset_name] * len(fields['species']))
        dataset_labels_spacegroups.extend([dataset_name] * len(fields['spacegroup']))
        dataset_labels_seq_len.extend([dataset_name] * len(fields['seq_len']))
        for param in cell_params:
            all_cell_params[param].extend(fields['cell_params'][param])
            dataset_labels_cell_params[param].extend([dataset_name] * len(fields['cell_params'][param]))

    # Define bins
    species_bins = np.linspace(min(all_species), max(all_species), 51)
    spacegroup_bins = np.linspace(min(all_spacegroups), max(all_spacegroups), 51)
    seq_len_bins = np.arange(min(all_seq_len), max(all_seq_len) + 1, 51)
    cell_params_bins = {param: np.linspace(min(all_cell_params[param]), max(all_cell_params[param]), 51) for param in cell_params}

    # Replace inf values with NaN and drop NaNs
    species_data = pd.Series(all_species).replace([np.inf, -np.inf], np.nan).dropna().tolist()
    spacegroup_data = pd.Series(all_spacegroups).replace([np.inf, -np.inf], np.nan).dropna().tolist()
    seq_len_data = pd.Series(all_seq_len).replace([np.inf, -np.inf], np.nan).dropna().tolist()
    for param in cell_params:
        all_cell_params[param] = pd.Series(all_cell_params[param]).replace([np.inf, -np.inf], np.nan).dropna().tolist()

    # Plot histograms
    fig, axes = plt.subplots(4, 3, figsize=(14, 18))
    axes = axes.ravel()

    sns.histplot(
        x=species_data,
        bins=species_bins,
        ax=axes[0],
        hue=dataset_labels_species,
        element='step',
        stat='density',
        common_norm=False
    )
    axes[0].set(title='Species', xlabel='Z')

    sns.histplot(
        x=spacegroup_data,
        bins=spacegroup_bins,
        ax=axes[1],
        hue=dataset_labels_spacegroups,
        element='step',
        stat='density',
        common_norm=False
    )
    axes[1].set(title='Spacegroup', xlabel='Spg number')

    sns.histplot(
        x=seq_len_data,
        bins=seq_len_bins,
        ax=axes[2],
        hue=dataset_labels_seq_len,
        element='step',
        stat='density',
        common_norm=False
    )
    axes[2].set(title='Sequence Length', xlabel='Seq Len')

    for i, param in enumerate(cell_params):
        print(len(all_cell_params[param]))
        sns.histplot(
            x=all_cell_params[param],
            bins=cell_params_bins[param],
            ax=axes[3+i],
            hue=dataset_labels_cell_params[param],
            element='step',
            stat='density',
            common_norm=False
        )
        axes[3+i].set(title=f'Cell Param: {param}', xlabel=param)

    for ax in axes.flatten():
        ax.set(ylabel='Density', xlim=(0, None), ylim=(None, None))
        ax.grid(alpha=0.2)

    fig.tight_layout()
    # Save the plot
    save_path = os.path.join(output_dir, 'Histograms.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_boxplots(stat_dict, output_dir):
    """
    Plot boxplots for different datasets.

    Parameters:
        stat_dict (dict): Dictionary containing statistical data.
        output_dir (str): Directory to save output plots.
    """
    from scipy import stats

    cell_params = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'implied_vol', 'gen_vol']
    params_data_dict = {}
    params_labels_dict = {}

    # Prepare data
    params_data_dict['Species Z'] = []
    params_labels_dict['Species Z'] = []
    params_data_dict['Spacegroup'] = []
    params_labels_dict['Spacegroup'] = []
    params_data_dict['Sequence Length'] = []
    params_labels_dict['Sequence Length'] = []
    for param in cell_params:
        params_data_dict[f'Cell Param: {param}'] = []
        params_labels_dict[f'Cell Param: {param}'] = []

    for dataset_name, fields in stat_dict.items():
        params_data_dict['Species Z'].extend(fields['species'])
        params_labels_dict['Species Z'].extend([dataset_name] * len(fields['species']))
        params_data_dict['Spacegroup'].extend(fields['spacegroup'])
        params_labels_dict['Spacegroup'].extend([dataset_name] * len(fields['spacegroup']))
        params_data_dict['Sequence Length'].extend(fields['seq_len'])
        params_labels_dict['Sequence Length'].extend([dataset_name] * len(fields['seq_len']))
        for param in cell_params:
            params_data_dict[f'Cell Param: {param}'].extend(fields['cell_params'][param])
            params_labels_dict[f'Cell Param: {param}'].extend([dataset_name] * len(fields['cell_params'][param]))

    # Plot boxplots
    n_rows = 4
    n_cols = 3
    figsize = (12, 28)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, param in enumerate(params_data_dict.keys()):
        param_data = params_data_dict[param]
        dataset_labels = params_labels_dict[param]

        df = pd.DataFrame({
            param: param_data,
            'Dataset': dataset_labels
        })

        sns.boxplot(x='Dataset', y=param, data=df, palette='pastel', linewidth=1.5, dodge=False, ax=axes[idx])
        axes[idx].set_title(f'{param} Across Datasets')
        axes[idx].set_ylabel(param, fontsize=10)
        axes[idx].set_xlabel('Dataset', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45, labelsize=6)
        axes[idx].tick_params(axis='y')
        axes[idx].grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    # Save the plot
    save_path = os.path.join(output_dir, 'Boxplots.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_validity_stats(df, output_dir):
    """
    Calculate validity statistics and create a bar plot.

    Parameters:
        df (pd.DataFrame): DataFrame containing validity columns.
        output_dir (str): Directory to save output plot.
    """
    validity_columns = ['validity.formula', 'validity.spacegroup', 'validity.bond_length', 'validity.site_multiplicity']
    df[validity_columns] = df[validity_columns].astype(bool)

    grouped = df.groupby('dataset_name')[validity_columns].mean() * 100
    grouped = grouped.reset_index().melt(id_vars='dataset_name', var_name='validity_metric', value_name='percentage_valid')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='dataset_name', y='percentage_valid', hue='validity_metric', data=grouped)
    plt.title("Validity Metrics per Dataset (as % of valid entries)")
    plt.ylabel("Percentage of Valid Entries (%)")
    plt.xlabel("Dataset")
    plt.legend(title="Validity Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(output_dir, 'ValidityMetrics.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and extract relevant features
    df = load_and_extract_features(args.input_files)

    # Extract statistical data
    unique_datasets = df['dataset_name'].unique()
    stat_dict = {}
    for dataset_name in unique_datasets:
        stat_dict[dataset_name] = {
            'species': [],
            'spacegroup': [],
            'seq_len': [],
            'cell_params': {},
            'validity': {}
        }
        idxs = df['dataset_name'] == dataset_name
        stat_dict[dataset_name]['spacegroup'] = df.loc[idxs, 'spacegroup_int'].tolist()
        stat_dict[dataset_name]['seq_len'] = df.loc[idxs, 'seq_len'].tolist()
        cell_params = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'implied_vol', 'gen_vol']
        for param in cell_params:
            stat_dict[dataset_name]['cell_params'][param] = df.loc[idxs, f'cell_params.{param}'].tolist()
        for specie_list in df.loc[idxs, 'species'].to_list():
            if specie_list is not None:
                for specie in specie_list:
                    stat_dict[dataset_name]['species'].append(int(Element(specie).Z))
        validity_params = ['formula', 'spacegroup', 'bond_length', 'site_multiplicity']
        for param in validity_params:
            validity_data = df.loc[idxs, f'validity.{param}'].replace({None: False})
            aligned_validity_data = validity_data.dropna().tolist()
            stat_dict[dataset_name]['validity'][param] = aligned_validity_data

    # Plot histograms
    plot_histograms(stat_dict, args.output_dir)

    # Plot boxplots
    plot_boxplots(stat_dict, args.output_dir)

    # Plot validity statistics
    plot_validity_stats(df, args.output_dir)

    # Extract individual and combined feature matrices
    feature_dict, df_filtered = get_feature_matrices(df)

    # Initialize dictionaries to store reduced features and filtered DataFrames
    reduced_features_dict = {}
    df_filtered_dict = {}

    for key in feature_dict:
        print(f"Performing PCA on {key} features...")
        reduced_features, pca, scaler = perform_dimensionality_reduction(feature_dict[key], n_components=2)

        # Outlier removal after PCA
        if args.remove_outliers:
            print(f"Removing outliers from {key} reduced features...")
            mask = filter_outliers_on_reduced_features(
                reduced_features, method=args.outlier_method, thresh=args.outlier_thresh
            )
            print(f"Samples before outlier removal: {len(reduced_features)}")
            print(f"Samples after outlier removal: {np.sum(mask)}")

            if np.sum(mask) == 0:
                raise ValueError(f"No data left after outlier removal on {key} features.")

            # Filter reduced features and DataFrame
            reduced_features = reduced_features[mask]
            df_filtered_key = df_filtered.iloc[mask].reset_index(drop=True)
            df_filtered_dict[key] = df_filtered_key
        else:
            df_filtered_dict[key] = df_filtered.copy()

        # Store the filtered reduced features
        reduced_features_dict[key] = reduced_features

    # Visualize the samples and save plots
    visualize_samples(
        reduced_features_dict,
        df_filtered_dict,
        args.output_dir
    )

if __name__ == "__main__":
    main()
