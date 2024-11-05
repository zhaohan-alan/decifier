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
        else:
            xrd_dirty_from_sample = row['descriptors.xrd_sample.iq']
            xrd_clean_from_sample = row['descriptors.xrd_sample.iq']
            xrd_clean_from_gen    = row['descriptors.xrd_gen.iq']
            xrd_q                 = row['descriptors.xrd_gen.q']

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

            # Append to data list
            data_list.append({
                'rwp_dirty': rwp_dirty_value,
                'rwp_clean': rwp_clean_value,
                'r2_dirty': r2_dirty_value,
                'r2_clean': r2_clean_value,
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

def fingerprint_comparison(
    eval_path_dict: Dict[str, str],
    output_folder: str,
    debug_max: bool = None,
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

    # Make figure
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10, 10), sharex=True)

    ax1.set_title(r"Fingerprint comparison")

    # Plot combined violin and boxplot for Rwp (Top-left)
    sns.violinplot(data=data_rwp_clean, cut=0, ax=ax1)
    sns.boxplot(data=data_rwp_clean, whis=1.5, fliersize=2, linewidth=1.5, boxprops=dict(alpha=0.2), ax=ax1)

    # Add medians to the Rwp plot
    for i, label in enumerate(labels):
        med_value = np.median(df_data[label]['rwp_clean'].values)
        text = ax1.text(i, med_value + 0.01, f'{med_value:.2f}', ha='center', va='bottom', fontsize=10, color='black')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

    ax1.set_ylabel(r"$R_{wp}$")
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, rotation=30, ha='right')
    ax1.set_ylim(0,)

    ax1.axvline(x=0.5, lw=1, ls='--', c='k')
    ax1.axvline(x=3.5, lw=1, ls='--', c='k')

    # Plot combined violin and boxplot for R^2 (Top-right)
    sns.violinplot(data=data_r2_clean, cut=0, ax=ax2)
    sns.boxplot(data=data_r2_clean, whis=1.5, fliersize=2, linewidth=1.5, boxprops=dict(alpha=0.2), ax=ax2)

    # Add medians to the R^2 plot
    for i, label in enumerate(labels):
        med_value = np.median(df_data[label]['r2_clean'].values)
        text = ax2.text(i, med_value + 0.01, f'{med_value:.2f}', ha='center', va='bottom', fontsize=10, color='black')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
        
    ax2.set_ylabel(r"$R^2$")
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels, rotation=30, ha='right')

    ax2.axvline(x=0.5, lw=1, ls='--', c='k')
    ax2.axvline(x=3.5, lw=1, ls='--', c='k')

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "fingerprint_comparison_violin.png"), dpi=200)
    plt.show()
    plt.close()

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10, 6))

    # Plot normalized histograms for Rwp (Bottom-left)
    for label in labels:
        rwp_values = df_data[label]['rwp_clean'].values
        ax1.hist(rwp_values, bins=50, alpha=0.7, density=True, label=label)

    ax1.set_title("Histogram of Rwp Values")
    ax1.set_xlabel("$R_{wp}$")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2)

    # Plot normalized histograms for R^2 (Bottom-right)
    for label in labels:
        r2_values = df_data[label]['r2_clean'].values
        ax2.hist(r2_values, bins=50, alpha=0.7, density=True, label=label)

    ax2.set_title("Histogram of $R^2$ Values")
    ax2.set_ylabel("Density")
    ax2.set_xlabel(r"$R^2$")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "fingerprint_comparison_1d_histogram.png"), dpi=200)
    plt.show()
    plt.close()

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
