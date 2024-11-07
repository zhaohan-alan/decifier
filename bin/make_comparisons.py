import argparse
import os
import yaml
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns

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

        # Extract XRD
        if not use_deprecated_keys:
            xrd_dirty_from_sample = row['descriptors.xrd_dirty_from_sample.iq']
            xrd_clean_from_sample = row['descriptors.xrd_clean_from_sample.iq']
            xrd_clean_from_gen    = row['descriptors.xrd_clean_from_gen.iq']
            xrd_q                 = row['descriptors.xrd_clean_from_gen.q']
            soap_from_sample      = row['descriptors.soap_from_sample']
            soap_from_gen         = row['descriptors.soap_from_gen']
        else:
            xrd_dirty_from_sample = row['descriptors.xrd_sample.iq']
            xrd_clean_from_sample = row['descriptors.xrd_sample.iq']
            xrd_clean_from_gen    = row['descriptors.xrd_gen.iq']
            xrd_q                 = row['descriptors.xrd_gen.q']
            soap_from_sample      = row['descriptors.xrd_sample.iq']
            soap_from_gen         = row['descriptors.xrd_gen.iq']

        # Extract Validity
        formula_validity = row['validity.formula']
        bond_length_validity = row['validity.bond_length']
        spacegroup_validity = row['validity.spacegroup']
        site_multiplicity_validity = row['validity.site_multiplicity']

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
            distance = soap_distance(soap_from_sample, soap_from_gen)

            # Append to data list
            data_list.append({
                'rwp_dirty': rwp_dirty_value,
                'rwp_clean': rwp_clean_value,
                'r2_dirty': r2_dirty_value,
                'r2_clean': r2_clean_value,
                'soap_distance': distance,
                'xrd_q': xrd_q,
                'xrd_dirty_from_sample': xrd_dirty_from_sample,
                'xrd_clean_from_sample': xrd_clean_from_sample,
                'xrd_clean_from_gen': xrd_clean_from_gen,
                'spacegroup_sym': spacegroup_sym,
                'spacegroup_num': spacegroup_num,
                'cif_sample': cif_sample,
                'cif_gen': cif_gen,
            })

        pbar.update(1)

    pbar.close()

    # Convert data list to dataframe
    df_results = pd.DataFrame(data_list)
    return df_results

#def # TODO Make function for comparing soap descriptor distributions, PCA/T-SNE, internal comparisons of similarity and external to generated set etc.

def plot_violin_box(
    data,
    labels,
    ylabel,
    title,
    ax,
    cut = 0,
    medians=None,
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

    data_rwp_dirty = [df_data[label]['rwp_dirty'].values for label in labels]
    data_rwp_clean = [df_data[label]['rwp_clean'].values for label in labels]
    data_r2_dirty = [df_data[label]['r2_dirty'].values for label in labels]
    data_r2_clean = [df_data[label]['r2_clean'].values for label in labels]
    data_soap_distance = [df_data[label]['soap_distance'].values for label in labels]

    # Dirty
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10, 10), sharex=True)

    plot_violin_box(data_rwp_dirty, labels, ylabel=r"$R_{wp}$", title="Fingerprint comparisons (dirty)", ax=ax1,
                    medians = {label: df_data[label]['rwp_dirty'].values for label in labels})
    plot_violin_box(data_r2_dirty, labels, ylabel=r"$R^{2}$", title="", ax=ax2,
                    medians = {label: df_data[label]['r2_dirty'].values for label in labels})

    save_figure(fig, os.path.join(output_folder, "fingerprint_dirty_comparison_violin.png"))
    
    # Clean
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10, 10), sharex=True)

    plot_violin_box(data_rwp_clean, labels, ylabel=r"$R_{wp}$", title="Fingerprint comparisons (clean)", ax=ax1,
                    medians = {label: df_data[label]['rwp_clean'].values for label in labels})
    plot_violin_box(data_r2_clean, labels, ylabel=r"$R^{2}$", title="", ax=ax2,
                    medians = {label: df_data[label]['r2_clean'].values for label in labels})

    save_figure(fig, os.path.join(output_folder, "fingerprint_clean_comparison_violin.png"))

    # Clean
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
    
    # SOAP distance plot
    fig, ax = plt.subplots(figsize=(10, 5))

    plot_violin_box(data_soap_distance, labels, ylabel="Structural similarity", title="SOAP Distance distribution", ax=ax,
                    medians = {label: df_data[label]['soap_distance'].values for label in labels})

    save_figure(fig, os.path.join(output_folder, "fingerprint_soap_distance.png"))

    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to comparison .yaml config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Parse yaml to namespace and merge (DictConfig)
    yaml_dictconfig = OmegaConf.create(yaml_config)

    fingerprint_comparison(
        eval_path_dict = yaml_dictconfig.eval_path_dict, 
        output_folder = yaml_dictconfig.output_folder, 
        debug_max = yaml_dictconfig.debug_max,
        use_deprecated_keys = yaml_dictconfig.use_deprecated_keys
    )
