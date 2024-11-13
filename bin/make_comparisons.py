import argparse
import os
import yaml
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import gzip
import pickle

from scipy.integrate import simpson
from scipy.stats import ks_2samp
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wasserstein_distance

from sklearn.decomposition import PCA
from umap import UMAP

from omegaconf import OmegaConf
from typing import Dict, Optional

from decifer.utility import extract_space_group_symbol, space_group_symbol_to_number

from multiprocessing import Pool, cpu_count

def rwp(sample, gen):
    """
    Calculates the residual (un)weighted profile between a sample and a generated XRD pattern
    """
    return np.sqrt(np.sum(np.square(sample - gen), axis=-1) / np.sum(np.square(sample), axis=-1))

def r2(sample, gen):
    """
    Calculates the R^2 between a sample and a generated XRD pattern
    """
    return 1 - np.sum((sample - gen)**2) / np.sum((sample - np.mean(sample))**2)

def s12(y1, y2, l=2, s_range=np.arange(-500,501)):
    def triangular_weighting(s, l):
        return np.maximum(0, 1 - np.abs(s) / l)
    
    def cross_correlation(y1, y2, s_range):
        return [np.sum(y1 * np.roll(y2, shift)) for shift in s_range]

    c12 = cross_correlation(y1, y2, s_range)
    c11 = cross_correlation(y1, y1, s_range)
    c22 = cross_correlation(y2, y2, s_range)

    weights = triangular_weighting(s_range, l)
    weighted_c12 = weights * c12
    weighted_c11 = weights * c11
    weighted_c22 = weights * c22

    numerator = simpson(weighted_c12, x=s_range)
    denominator = np.sqrt(simpson(weighted_c11, x=s_range) * simpson(weighted_c22, x=s_range))
    return numerator / denominator if denominator != 0 else 0

def soap_distance(sample, gen):
    """
    Calculates the normalized soap distance between sample and gen soap descriptors
    """
    return np.dot(sample, gen) / np.sqrt(np.dot(sample, sample) * np.dot(gen, gen))

def cohen_d(sample1, sample2):
    mean_diff = np.mean(sample1) - np.mean(sample2)
    pooled_std = np.sqrt((np.var(sample1) + np.var(sample2)) / 2)
    return mean_diff / pooled_std

def ks_test(sample1, sample2):
    statistic, p_value = ks_2samp(sample1, sample2)
    return statistic, p_value

def percentile_improvement(new_values, reference_values, percentile=75):
    threshold = np.percentile(reference_values, percentile)
    return np.mean(new_values > threshold)

def process_file(file_path):
    """Processes a single .pkl.gz file."""
    try:
        with gzip.open(file_path, 'rb') as f:
            row = pickle.load(f)

        if row['status'] != 'success':
            return None

        # Extract Validity
        formula_validity = row['validity']['formula']
        bond_length_validity = row['validity']['bond_length']
        spacegroup_validity = row['validity']['spacegroup']
        site_multiplicity_validity = row['validity']['site_multiplicity']

        # Full validity check
        valid = all([formula_validity, bond_length_validity, spacegroup_validity, site_multiplicity_validity])
        if not valid:
            return None

        # Extract CIFs and descriptors
        cif_sample = row['cif_sample']
        cif_gen = row['cif_gen']

        xrd_dirty_from_sample = row['descriptors']['xrd_dirty_from_sample']['iq']
        xrd_clean_from_sample = row['descriptors']['xrd_clean_from_sample']['iq']
        xrd_clean_from_gen = row['descriptors']['xrd_clean_from_gen']['iq']

        xrd_disc_clean_from_sample = row['descriptors']['xrd_clean_from_sample']['iq_disc']
        xrd_disc_clean_from_gen = row['descriptors']['xrd_clean_from_gen']['iq_disc']
        q_disc_clean_from_sample = row['descriptors']['xrd_clean_from_sample']['q_disc']
        q_disc_clean_from_gen = row['descriptors']['xrd_clean_from_gen']['q_disc']

        disc_clean_from_sample = np.vstack((q_disc_clean_from_sample, xrd_disc_clean_from_sample)).T
        disc_clean_from_gen = np.vstack((q_disc_clean_from_gen, xrd_disc_clean_from_gen)).T
        hd_clean_value_1 = directed_hausdorff(disc_clean_from_sample, disc_clean_from_gen, seed=42)[0]
        hd_clean_value_2 = directed_hausdorff(disc_clean_from_gen, disc_clean_from_sample, seed=42)[0]

        ws_clean_value = wasserstein_distance(q_disc_clean_from_sample, q_disc_clean_from_gen, u_weights=xrd_disc_clean_from_sample, v_weights=xrd_disc_clean_from_gen)

        xrd_q = row['descriptors']['xrd_clean_from_gen']['q']
        soap_small_sample = row['descriptors']['soap_small_sample']
        soap_small_gen = row['descriptors']['soap_small_gen']
        soap_large_sample = row['descriptors']['soap_large_sample']
        soap_large_gen = row['descriptors']['soap_large_gen']

        # Calculate metrics
        rwp_dirty_value = rwp(xrd_dirty_from_sample, xrd_clean_from_gen)
        rwp_clean_value = rwp(xrd_clean_from_sample, xrd_clean_from_gen)
        s12_dirty_value = s12(xrd_dirty_from_sample, xrd_clean_from_gen)
        s12_clean_value = s12(xrd_clean_from_sample, xrd_clean_from_gen)
        hd_clean_value = max(hd_clean_value_1, hd_clean_value_2)
        r2_dirty_value = r2(xrd_dirty_from_sample, xrd_clean_from_gen)
        r2_clean_value = r2(xrd_clean_from_sample, xrd_clean_from_gen)
        distance_small = soap_distance(soap_small_sample, soap_small_gen)
        distance_large = soap_distance(soap_large_sample, soap_large_gen)

        # Extract space group
        spacegroup_sym = extract_space_group_symbol(cif_gen)
        spacegroup_num = space_group_symbol_to_number(spacegroup_sym)
        spacegroup_num = int(spacegroup_num) if spacegroup_num is not None else 0

        return {
            'rwp_dirty': rwp_dirty_value,
            'rwp_clean': rwp_clean_value,
            's12_dirty': s12_dirty_value,
            's12_clean': s12_clean_value,
            'hd_clean': hd_clean_value,
            'ws_clean': ws_clean_value,
            'r2_dirty': r2_dirty_value,
            'r2_clean': r2_clean_value,
            'soap_small_distance': distance_small,
            'soap_large_distance': distance_large,
            'xrd_q': xrd_q,
            'xrd_dirty_from_sample': xrd_dirty_from_sample,
            'xrd_clean_from_sample': xrd_clean_from_sample,
            'xrd_clean_from_gen': xrd_clean_from_gen,
            'spacegroup_sym': spacegroup_sym,
            'spacegroup_num': spacegroup_num,
            'cif_sample': cif_sample,
            'cif_gen': cif_gen,
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process(folder, debug_max=None):
    """Processes all files in the given folder using multiprocessing."""
    # Get list of files
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pkl.gz')]
    if debug_max is not None:
        files = files[:debug_max]

    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files), desc="Processing files..."))

    # Filter out None results and convert to DataFrame
    data_list = [res for res in results if res is not None]
    return pd.DataFrame(data_list)

def prepare_data_for_plotting(
    eval_folder_dict: Dict[str, str],
    debug_max: Optional[int] = None,
):
    df_data = {}
    labels = []
    pbar = tqdm(total = len(eval_folder_dict), desc="Processing datasets", leave=False)
    for label, eval_folder in eval_folder_dict.items():
        df_results = process(eval_folder, debug_max)
        df_data[label] = df_results
        labels.append(label)
        pbar.update(1)
    pbar.close()

    return df_data, labels

def plot_2d_scatter(
    data,
    label,
    ax,
    xlabel = None,
    ylabel = None,
    title = None,
    within_norm: Optional[float] = None,
    grid: bool = True,
):
    if within_norm:
        data = data[np.linalg.norm(data, axis=-1) <= within_norm]
    sns.scatterplot(x = data[:,0], y = data[:,1], label=label, ax=ax)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    if grid:
        ax.grid(alpha=.2)

def structural_diversity(
    df_data,
    labels,
    output_folder,
) -> None:

    # Use the small soap descriptors to make distribution scatter plots using PCA and UMAP
    pca = PCA(n_components=2, random_state=42)
    umap = UMAP(n_components=2, random_state=42)

    # SOAP PCA
    for label in labels:
        soap_sample = np.stack(df_data[label]['soap_small_sample'].values)
        soap_gen = np.stack(df_data[label]['soap_small_gen'].values)
        reduced_soap_sample = pca.fit_transform(soap_sample)
        reduced_soap_gen = pca.transform(soap_gen)

        fig, ax = plt.subplots(figsize=(10, 10))
        plot_2d_scatter(
            data = reduced_soap_sample,
            label = 'sample',
            xlabel = 'PCA 1',
            ylabel = 'PCA 2',
            title = f'PCA of small SOAP descriptors for {label}',
            ax = ax,
        )
        plot_2d_scatter(
            data = reduced_soap_gen,
            label = 'generated',
            ax = ax,
        )
        save_figure(fig, os.path.join(output_folder, f"PCA_SOAP_{label}.png"))
        plt.close(fig)

    # SOAP UMAP
    for label in labels:
        soap_sample = np.stack(df_data[label]['soap_small_sample'].values)
        soap_gen = np.stack(df_data[label]['soap_small_gen'].values)
        reduced_soap_sample = umap.fit_transform(soap_sample)
        reduced_soap_gen = umap.transform(soap_gen)

        fig, ax = plt.subplots(figsize=(10, 10))
        plot_2d_scatter(
            data = reduced_soap_sample,
            label = 'sample',
            xlabel = 'Component 1',
            ylabel = 'Component 2',
            title = f'UMAP of small SOAP descriptors for {label}',
            ax = ax,
        )
        plot_2d_scatter(
            data = reduced_soap_gen,
            label = 'generated',
            ax = ax,
        )
        save_figure(fig, os.path.join(output_folder, f"UMAP_SOAP_{label}.png"))
        plt.close(fig)

def plot_violin_box(
    data,
    labels,
    ylabel,
    title,
    ax,
    cut = 0,
    medians=None,
    ylim = None,
):
    sns.violinplot(data = data, cut = cut, ax=ax)
    sns.boxplot(data = data, whis = 1.5, fliersize = 2, linewidth = 1.5, boxprops = dict(alpha=0.2), ax=ax)
    if medians:
        for i, label in enumerate(labels):
            med_value = np.median(medians[label])
            text = ax.text(i, med_value + 0.01, f'{med_value:.2f}', ha='center', va='bottom', fontsize=10, color='black')
            text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    if ylim:
        ax.set_ylim(ylim)

def plot_histogram(
    data,
    labels,
    xlabel,
    ylabel,
    title,
    ax,
):
    for label in labels:
        values = data[label]
        ax.hist(values, bins=50, alpha=0.7, density=True, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

def save_plotting_data(data_to_save, output_folder):
    """Saves the plotting data to a pickle file."""
    data_file = os.path.join(output_folder, 'fingerprint_plotting_data.pkl')
    with open(data_file, 'wb') as f:
        pickle.dump(data_to_save, f)

def load_plotting_data(output_folder):
    """Loads the plotting data from a pickle file."""
    data_file = os.path.join(output_folder, 'fingerprint_plotting_data.pkl')
    with open(data_file, 'rb') as f:
        data_loaded = pickle.load(f)
    return data_loaded

def fingerprint_comparison(
    df_data,
    labels,
    output_folder,
    use_saved_data=False,
) -> None:
    if use_saved_data:
        # Load data from saved file
        data_loaded = load_plotting_data(output_folder)
        data_rwp_clean = data_loaded['data_rwp_clean']
        medians_rwp_clean = data_loaded['medians_rwp_clean']
        data_s12_clean = data_loaded['data_s12_clean']
        medians_s12_clean = data_loaded['medians_s12_clean']
        data_hd_clean = data_loaded['data_hd_clean']
        medians_hd_clean = data_loaded['medians_hd_clean']
        data_ws_clean = data_loaded['data_ws_clean']
        medians_ws_clean = data_loaded['medians_ws_clean']
        data_r2_clean = data_loaded['data_r2_clean']
        medians_r2_clean = data_loaded['medians_r2_clean']
        data_soap_large_distance = data_loaded['data_soap_large_distance']
        medians_soap_large_distance = data_loaded['medians_soap_large_distance']
        data_soap_small_distance = data_loaded['data_soap_small_distance']
        medians_soap_small_distance = data_loaded['medians_soap_small_distance']
        labels = data_loaded['labels']
    else:
        # Collect data for plotting
        data_rwp_clean = [df_data[label]['rwp_clean'].values for label in labels]
        medians_rwp_clean = {label: df_data[label]['rwp_clean'].values for label in labels}

        data_s12_clean = [df_data[label]['s12_clean'].values for label in labels]
        medians_s12_clean = {label: df_data[label]['s12_clean'].values for label in labels}

        data_hd_clean = [df_data[label]['hd_clean'].values for label in labels]
        medians_hd_clean = {label: df_data[label]['hd_clean'].values for label in labels}

        data_ws_clean = [df_data[label]['ws_clean'].values for label in labels]
        medians_ws_clean = {label: df_data[label]['ws_clean'].values for label in labels}

        data_r2_clean = [df_data[label]['r2_clean'].values for label in labels]
        medians_r2_clean = {label: df_data[label]['r2_clean'].values for label in labels}

        data_soap_large_distance = [df_data[label]['soap_large_distance'].values for label in labels]
        medians_soap_large_distance = {label: df_data[label]['soap_large_distance'].values for label in labels}

        data_soap_small_distance = [df_data[label]['soap_small_distance'].values for label in labels]
        medians_soap_small_distance = {label: df_data[label]['soap_small_distance'].values for label in labels}

        # Save the data for future use
        data_to_save = {
            'data_rwp_clean': data_rwp_clean,
            'medians_rwp_clean': medians_rwp_clean,
            'data_s12_clean': data_s12_clean,
            'medians_s12_clean': medians_s12_clean,
            'data_hd_clean': data_hd_clean,
            'medians_hd_clean': medians_hd_clean,
            'data_ws_clean': data_ws_clean,
            'medians_ws_clean': medians_ws_clean,
            'data_r2_clean': data_r2_clean,
            'medians_r2_clean': medians_r2_clean,
            'data_soap_large_distance': data_soap_large_distance,
            'medians_soap_large_distance': medians_soap_large_distance,
            'data_soap_small_distance': data_soap_small_distance,
            'medians_soap_small_distance': medians_soap_small_distance,
            'labels': labels,
        }
        save_plotting_data(data_to_save, output_folder)

    # Now plot using the data
    fig, axs = plt.subplots(7,1,figsize=(5, 10), sharex=True)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axs

    plot_violin_box(data_rwp_clean, labels, ylabel=r"$R_{wp}$", title="Fingerprints", ax=ax1,
                    medians=medians_rwp_clean)

    plot_violin_box(data_s12_clean, labels, ylabel=r"$S_{12}$", title="", ax=ax2,
                    medians=medians_s12_clean)

    plot_violin_box(data_hd_clean, labels, ylabel="HD", title="", ax=ax3,
                    medians=medians_hd_clean)

    plot_violin_box(data_ws_clean, labels, ylabel="WS", title="", ax=ax4,
                    medians=medians_ws_clean)

    plot_violin_box(data_r2_clean, labels, ylabel=r"$R^{2}$", title="", ax=ax5,
                    medians=medians_r2_clean)

    plot_violin_box(data_soap_large_distance, labels, ylabel="Structural similarity", title="", ax=ax6,
                    medians=medians_soap_large_distance)

    plot_violin_box(data_soap_small_distance, labels, ylabel="Structural similarity", title="", ax=ax7,
                    medians=medians_soap_small_distance)

    plt.tight_layout()
    plt.show()
    save_figure(fig, os.path.join(output_folder, "fingerprint_violin.png"))

def save_figure(
    fig,
    output_path,
):
    fig.savefig(output_path, dpi=200)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to comparison .yaml config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Parse yaml to namespace and merge (DictConfig)
    yaml_dictconfig = OmegaConf.create(yaml_config)
    
    # Create output folder
    os.makedirs(yaml_dictconfig.output_folder, exist_ok=True)

    if yaml_dictconfig.use_saved_data:
        # Skip processing and set df_data and labels to None
        df_data = None
        labels = None
    else:
        # Process data
        df_data, labels = prepare_data_for_plotting(yaml_dictconfig.eval_folder_dict, yaml_dictconfig.debug_max)

    if yaml_dictconfig.fingerprint_comparison:
        fingerprint_comparison(df_data, labels, yaml_dictconfig.output_folder, use_saved_data=yaml_dictconfig.use_saved_data)

    if yaml_dictconfig.structural_diversity:
        structural_diversity(df_data, labels, yaml_dictconfig.output_folder)
