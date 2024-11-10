import argparse
import os
import yaml
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from omegaconf import OmegaConf
from typing import Dict, Optional

from decifer.utility import extract_space_group_symbol, space_group_symbol_to_number

def rwp(sample, gen):
    """
    Calculates the residual (un)weighted profile between a sample and a generated XRD pattern
    """
    return np.sqrt(np.sum(np.square(sample - gen), axis=0) / np.sum(np.square(sample), axis=0))

def r2(sample, gen):
    """
    Calculates the R^2 between a sample and a generated XRD pattern
    """
    return 1 - np.sum((sample - gen)**2) / np.sum((sample - np.mean(sample))**2)

def soap_distance(sample, gen):
    """
    Calculates the normalized soap distance between sampe and gen soap descriptors
    """
    return np.dot(sample, gen) / np.sqrt(np.dot(sample, sample) * np.dot(gen, gen))

def process_eval(
        eval_path: str,
        debug_max: Optional[int] = None,
        disable_outer_pbar: bool = True,
        use_deprecated_keys: bool = False,
) -> pd.DataFrame:

    # Load eval file
    df_eval = pd.read_parquet(eval_path)

    # Make dataframe of processed data
    data_list = []

    total_rows = len(df_eval) if debug_max is None else min(debug_max, len(df_eval))
    pbar = tqdm(total = total_rows, desc=f"Processing {os.path.basename(eval_path)}", disable=disable_outer_pbar)

    for i, row in df_eval.iterrows():
        if row['status'] == 'fail':
            continue
        if debug_max is not None and i >= debug_max:
            break

        # Extract
        exclude_large_soap = False
        if not use_deprecated_keys:
            xrd_dirty_from_sample = row['descriptors.xrd_dirty_from_sample.iq']
            xrd_clean_from_sample = row['descriptors.xrd_clean_from_sample.iq']
            xrd_clean_from_gen    = row['descriptors.xrd_clean_from_gen.iq']
            xrd_q                 = row['descriptors.xrd_clean_from_gen.q']
            try:
                soap_large_sample = row['descriptors.soap_large_sample']
                soap_large_gen    = row['descriptors.soap_large_gen']
            except:
                exclude_large_soap = True
            soap_small_sample     = row['descriptors.soap_small_sample']
            soap_small_gen        = row['descriptors.soap_small_gen']
        else:
            xrd_dirty_from_sample = row['descriptors.xrd_sample.iq']
            xrd_clean_from_sample = row['descriptors.xrd_sample.iq']
            xrd_clean_from_gen    = row['descriptors.xrd_gen.iq']
            xrd_q                 = row['descriptors.xrd_gen.q']
            soap_large_sample     = row['descriptors.xrd_sample.iq']
            soap_large_gen        = row['descriptors.xrd_gen.iq']
            soap_small_sample     = row['descriptors.xrd_sample.iq']
            soap_small_gen        = row['descriptors.xrd_gen.iq']

        # Extract Validity
        formula_validity          = row['validity.formula']
        bond_length_validity      = row['validity.bond_length']
        spacegroup_validity       = row['validity.spacegroup']
        site_multiplicity_validity= row['validity.site_multiplicity']

        # Full validity
        valid = np.all([formula_validity, bond_length_validity, spacegroup_validity, site_multiplicity_validity])

        # Extract cifs
        if not use_deprecated_keys:
            cif_sample = row['cif_sample']
            cif_gen = row['cif_gen']
        else:        
            cif_sample = None
            cif_gen = row['cif_string']

        # Check validity conditions --> then calculate metrics
        if valid:

            # Calculate Rwp
            rwp_dirty_value = rwp(xrd_dirty_from_sample, xrd_clean_from_gen)
            rwp_clean_value = rwp(xrd_clean_from_sample, xrd_clean_from_gen)

            # Calculate R^2
            r2_dirty_value = r2(xrd_dirty_from_sample, xrd_clean_from_gen)
            r2_clean_value = r2(xrd_clean_from_sample, xrd_clean_from_gen)

            # Extract spacegroup
            spacegroup_sym = extract_space_group_symbol(cif_gen)
            spacegroup_num = space_group_symbol_to_number(spacegroup_sym)
            spacegroup_num = int(spacegroup_num) if spacegroup_num is not None else 0

            # Calculate soap kernel
            distance_small = soap_distance(soap_small_sample, soap_small_gen)
            
            # Define data dict
            data_dict = {
                'rwp_dirty'            : rwp_dirty_value,
                'rwp_clean'            : rwp_clean_value,
                'r2_dirty'             : r2_dirty_value,
                'r2_clean'             : r2_clean_value,
                'soap_small_distance'  : distance_small,
                'xrd_q'                : xrd_q,
                'xrd_dirty_from_sample': xrd_dirty_from_sample,
                'xrd_clean_from_sample': xrd_clean_from_sample,
                'xrd_clean_from_gen'   : xrd_clean_from_gen,
                'soap_small_sample'    : soap_small_sample,
                'soap_small_gen'       : soap_small_gen,
                'spacegroup_sym'       : spacegroup_sym,
                'spacegroup_num'       : spacegroup_num,
                'cif_sample'           : cif_sample,
                'cif_gen'              : cif_gen,
            }
            
            if not exclude_large_soap:
                distance_large = soap_distance(soap_large_sample, soap_large_gen)
                data_dict['soap_large_distance'] = distance_large
                data_dict['soap_large_sample'] = soap_large_sample
                data_dict['soap_large_gen'] = soap_large_gen

            # Append to data list
            data_list.append(data_dict)

        pbar.update(1)

    pbar.close()

    # Convert data list to dataframe
    df_results = pd.DataFrame(data_list)
    return df_results

def prepare_data_for_plotting(
    eval_path_dict: Dict[str, str],
    debug_max: Optional[int] = None,
    disable_outer_pbar: bool = True,
    use_deprecated_keys: bool = False,
):
    # Prepare data and labels
    df_data = {}
    labels = []
    pbar = tqdm(total = len(eval_path_dict), desc="Processing datasets", leave=False)
    for label, eval_path in eval_path_dict.items():
        df_results = process_eval(
            eval_path,
            debug_max = debug_max,
            disable_outer_pbar = disable_outer_pbar,
            use_deprecated_keys = use_deprecated_keys,
        )
        df_data[label] = df_results
        labels.append(label)
        pbar.update(1)
    pbar.close()

    rwp_dirty = [df_data[label]['rwp_dirty'].values for label in labels]
    rwp_clean = [df_data[label]['rwp_clean'].values for label in labels]
    r2_dirty = [df_data[label]['r2_dirty'].values for label in labels]
    r2_clean = [df_data[label]['r2_clean'].values for label in labels]
    try:
        soap_large_distance = [df_data[label]['soap_large_distance'].values for label in labels]
        soap_large_sample = [df_data[label]['soap_large_sample'].values for label in labels]
        soap_large_gen = [df_data[label]['soap_large_gen'].values for label in labels]
    except:
        soap_large_distance = None
        soap_large_sample = None
        soap_large_gen = None
    soap_small_distance = [df_data[label]['soap_small_distance'].values for label in labels]
    soap_small_sample = [df_data[label]['soap_small_sample'].values for label in labels]
    soap_small_gen = [df_data[label]['soap_small_gen'].values for label in labels]

    data_dict = dict(
        rwp_dirty = rwp_dirty,
        rwp_clean = rwp_clean,
        r2_dirty = r2_dirty,
        r2_clean = r2_clean,
        soap_large_distance = soap_large_distance,
        soap_large_sample = soap_large_sample,
        soap_large_gen = soap_large_gen,
        soap_small_distance = soap_small_distance,
        soap_small_sample = soap_small_sample,
        soap_small_gen = soap_small_gen,
    )

    return df_data, data_dict, labels

#def # TODO Make function for comparing soap descriptor distributions, PCA/T-SNE, internal comparisons of similarity and external to generated set etc.
# Also use FID to compare and access the distance between the distributions of features in real and generated structure sets.

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
    eval_path_dict: Dict[str, str],
    output_folder: str,
    debug_max: Optional[int] = None,
    disable_outer_pbar: bool = True,
    use_deprecated_keys: bool = False,
) -> None:
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    df_data, flat_data, labels = prepare_data_for_plotting(eval_path_dict, debug_max, disable_outer_pbar, use_deprecated_keys)

    # Use the small soap descriptors to make distribution scatter plots using PCA and T-SNE
    # We are plotting all generated structures compared to their samples in the same plot
        
    pca = PCA(n_components=2, random_state=42)
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
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
        plt.show()
    
    # TSNE
    for label in labels:
        soap_sample = np.stack(df_data[label]['soap_small_sample'].values)
        soap_gen = np.stack(df_data[label]['soap_small_gen'].values)
        reduced_soap_sample = tsne.fit_transform(soap_sample)
        reduced_soap_gen = tsne.fit_transform(soap_gen)

        fig, ax = plt.subplots(figsize=(10, 10))
        plot_2d_scatter(
            data = reduced_soap_sample,
            label = 'sample',
            xlabel = 'Component 1',
            ylabel = 'Component 2',
            title = f'TSNE of small SOAP descriptors for {label}',
            ax = ax,
        )
        plot_2d_scatter(
            data = reduced_soap_gen,
            label = 'generated',
            ax = ax,
        )
        save_figure(fig, os.path.join(output_folder, f"TSNE_SOAP_{label}.png"))
        plt.show()
    
    # UMAP
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
        plt.show()

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
    ax.axvline(x=0.5, lw=1, ls='--', c='k')
    ax.axvline(x=3.5, lw=1, ls='--', c='k')
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

def save_figure(
    fig, 
    output_path,
):
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

def fingerprint_comparison(
    eval_path_dict: Dict[str, str],
    output_folder: str,
    debug_max: Optional[int] = None,
    disable_outer_pbar: bool = True,
    use_deprecated_keys: bool = False,
) -> None:

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    df_data, flat_data, labels = prepare_data_for_plotting(eval_path_dict, debug_max, disable_outer_pbar, use_deprecated_keys)

    # Dirty
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10, 10), sharex=True)

    plot_violin_box(flat_data['rwp_dirty'], labels, ylabel=r"$R_{wp}$", title="Fingerprint comparisons (dirty)", ax=ax1,
                    medians = {label: df_data[label]['rwp_dirty'].values for label in labels})
    plot_violin_box(flat_data['r2_dirty'], labels, ylabel=r"$R^{2}$", title="", ax=ax2,
                    medians = {label: df_data[label]['r2_dirty'].values for label in labels})

    save_figure(fig, os.path.join(output_folder, "fingerprint_dirty_comparison_violin.png"))
    
    # Clean
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10, 10), sharex=True)

    plot_violin_box(flat_data['rwp_clean'], labels, ylabel=r"$R_{wp}$", title="Fingerprint comparisons (clean)", ax=ax1,
                    medians = {label: df_data[label]['rwp_clean'].values for label in labels})
    plot_violin_box(flat_data['r2_clean'], labels, ylabel=r"$R^{2}$", title="", ax=ax2,
                    medians = {label: df_data[label]['r2_clean'].values for label in labels})

    save_figure(fig, os.path.join(output_folder, "fingerprint_clean_comparison_violin.png"))

    # Dirty
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10, 6))

    plot_histogram({label: df_data[label]['rwp_dirty'].values for label in labels}, labels, xlabel=r"$R_{wp}$",
                   ylabel="Density", title="Histogram of Rwp Values (dirty)", ax=ax1)
    plot_histogram({label: df_data[label]['r2_dirty'].values for label in labels}, labels, xlabel=r"$R^2$",
                   ylabel="Density", title="Histogram of $R^2$ Values", ax=ax2)

    save_figure(fig, os.path.join(output_folder, "fingerprint_dirty_comparison_1d_histogram.png"))

    # Clean
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10, 6))

    plot_histogram({label: df_data[label]['rwp_clean'].values for label in labels}, labels, xlabel=r"$R_{wp}$",
                   ylabel="Density", title="Histogram of Rwp Values (clean)", ax=ax1)
    plot_histogram({label: df_data[label]['r2_clean'].values for label in labels}, labels, xlabel=r"$R^2$",
                   ylabel="Density", title="Histogram of $R^2$ Values", ax=ax2)

    save_figure(fig, os.path.join(output_folder, "fingerprint_clean_comparison_1d_histogram.png"))
    
    # SOAP (large descriptor) distance plot
    if flat_data['soap_large_distance'] is not None:
        fig, ax = plt.subplots(figsize=(10, 5))

        plot_violin_box(flat_data['soap_large_distance'], labels, ylabel="Structural similarity", title="SOAP (large) Distance distribution", ax=ax,
                        medians = {label: df_data[label]['soap_large_distance'].values for label in labels}, ylim=(0,1))

        save_figure(fig, os.path.join(output_folder, "fingerprint_soap_large_distance.png"))

    # SOAP (small descriptor) distance plot
    fig, ax = plt.subplots(figsize=(10, 5))

    plot_violin_box(flat_data['soap_small_distance'], labels, ylabel="Structural similarity", title="SOAP (small) Distance distribution", ax=ax,
                    medians = {label: df_data[label]['soap_small_distance'].values for label in labels}, ylim=(0,1))

    save_figure(fig, os.path.join(output_folder, "fingerprint_soap_small_distance.png"))

    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to comparison .yaml config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Parse yaml to namespace and merge (DictConfig)
    yaml_dictconfig = OmegaConf.create(yaml_config)

    if yaml_dictconfig.fingerprint_comparison:
        fingerprint_comparison(
            eval_path_dict = yaml_dictconfig.eval_path_dict, 
            output_folder = yaml_dictconfig.output_folder, 
            debug_max = yaml_dictconfig.debug_max,
            use_deprecated_keys = yaml_dictconfig.use_deprecated_keys
        )

    if yaml_dictconfig.structural_diversity:
        structural_diversity(
            eval_path_dict = yaml_dictconfig.eval_path_dict, 
            output_folder = yaml_dictconfig.output_folder, 
            debug_max = yaml_dictconfig.debug_max,
            use_deprecated_keys = yaml_dictconfig.use_deprecated_keys
        )
