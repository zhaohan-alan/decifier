#!/usr/bin/env python3

import os
import re
import gzip
import yaml
import pickle
import argparse
import subprocess
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns

from scipy.stats import wasserstein_distance
from omegaconf import OmegaConf
from multiprocessing import Pool, cpu_count

from decifer_refactored.utility import extract_space_group_symbol, space_group_symbol_to_number

def rwp(sample, gen):
    """
    Calculates the residual (un)weighted profile between a sample and a generated XRD pattern
    """
    return np.sqrt(np.sum(np.square(sample - gen), axis=-1) / np.sum(np.square(sample), axis=-1))

def process_file(file_path):
    """Processes a single .pkl.gz file."""
    try:
        with gzip.open(file_path, 'rb') as f:
            row = pickle.load(f)

       # If successful generation, count 
        if 'success' not in row['status']:
            return None

        # Extract Validity
        formula_validity = row['validity']['formula']
        bond_length_validity = row['validity']['bond_length']
        spacegroup_validity = row['validity']['spacegroup']
        site_multiplicity_validity = row['validity']['site_multiplicity']
        valid = all([formula_validity, bond_length_validity, spacegroup_validity, site_multiplicity_validity])

        # Extract CIFs and XRD (Sample)
        cif_sample = row['cif_string_sample']
        xrd_q_continuous_sample = row['xrd_clean_sample']['q']
        xrd_iq_continuous_sample = row['xrd_clean_sample']['iq']
        xrd_q_discrete_sample = row['xrd_clean_sample']['q_disc']
        xrd_iq_discrete_sample = row['xrd_clean_sample']['iq_disc']

        # Extract CIFs and XRD (Generated)
        cif_gen = row['cif_string_gen']
        xrd_q_continuous_gen = row['xrd_clean_gen']['q']
        xrd_iq_continuous_gen = row['xrd_clean_gen']['iq']
        xrd_q_discrete_gen = row['xrd_clean_gen']['q_disc']
        xrd_iq_discrete_gen = row['xrd_clean_gen']['iq_disc']
        
        # Normalize for wasserstein
        # Wasserstein Distance
        xrd_iq_discrete_sample_normed = xrd_iq_discrete_sample / np.sum(xrd_iq_discrete_sample)
        xrd_iq_discrete_gen_normed = xrd_iq_discrete_gen / np.sum(xrd_iq_discrete_gen)
        wd_value = wasserstein_distance(xrd_q_discrete_sample, xrd_q_discrete_gen, u_weights=xrd_iq_discrete_sample_normed, v_weights=xrd_iq_discrete_gen_normed)

        # Rwp
        rwp_value = rwp(xrd_iq_continuous_sample, xrd_iq_continuous_gen)

        # Sequence lengths
        seq_len_sample = row['seq_len_sample']
        seq_len_gen = row['seq_len_gen']

        # Extract space group
        spacegroup_sym_sample = extract_space_group_symbol(cif_sample)
        spacegroup_num_sample = space_group_symbol_to_number(spacegroup_sym_sample)
        spacegroup_num_sample = int(spacegroup_num_sample) if spacegroup_num_sample is not None else np.nan

        spacegroup_sym_gen = extract_space_group_symbol(cif_gen)
        spacegroup_num_gen = space_group_symbol_to_number(spacegroup_sym_gen)
        spacegroup_num_gen = int(spacegroup_num_gen) if spacegroup_num_gen is not None else np.nan

        out_dict = {
            'rwp': rwp_value,
            'wd': wd_value,
            'cif_sample': cif_sample,
            'xrd_q_discrete_sample': xrd_q_discrete_sample,
            'xrd_iq_discrete_sample': xrd_iq_discrete_sample,
            'xrd_q_continuous_sample': xrd_q_continuous_sample,
            'xrd_iq_continuous_sample': xrd_iq_continuous_sample,
            'spacegroup_sym_sample': spacegroup_sym_sample,
            'spacegroup_num_sample': spacegroup_num_sample,
            'seq_len_sample': seq_len_sample,
            'cif_gen': cif_gen,
            'xrd_q_discrete_gen': xrd_q_discrete_gen,
            'xrd_iq_discrete_gen': xrd_iq_discrete_gen,
            'xrd_q_continuous_gen': xrd_q_continuous_gen,
            'xrd_iq_continuous_gen': xrd_iq_continuous_gen,
            'seq_len_gen': seq_len_gen,
            'spacegroup_sym_gen': spacegroup_sym_gen,
            'spacegroup_num_gen': spacegroup_num_gen,
            'formula_validity': formula_validity,
            'spacegroup_validity': spacegroup_validity,
            'bond_length_validity': bond_length_validity,
            'site_multiplicity_validity': site_multiplicity_validity,
            'validity': valid,
        }
        return out_dict
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process(folder, debug_max=None) -> pd.DataFrame:
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

def fingerprint_comparison(
    data_dict,
    dataset_labels,
    output_directory,
    vertical_lines=None,
    metrics_to_plot=None,
    color_pair_size=1,
):
    if metrics_to_plot is None:
        metrics_to_plot = [
            ('rwp', r"$R_{wp}$"),
            ('wd', "WD"),
        ]

    # Prepare the data in long format for efficient plotting
    plot_data = []
    for metric_key, _ in metrics_to_plot:
        for dataset_label in dataset_labels:
            metric_values = data_dict[dataset_label][metric_key].dropna()
            plot_data.extend([
                {'Dataset': dataset_label, 'Metric': metric_key, 'Value': val}
                for val in metric_values
            ])
    plot_data = pd.DataFrame(plot_data)

    # Initialize figure and axes
    fig, axes = plt.subplots(
        1, len(metrics_to_plot), figsize=(5, 5), sharey=True
    )
    if isinstance(axes, plt.Axes):
        axes = [axes]

    # Generate pairwise colors
    colors = sns.color_palette("tab10")
    dataset_colors = [colors[i // color_pair_size % len(colors)] for i in range(len(dataset_labels))]
    palette = {label: dataset_colors[i] for i, label in enumerate(dataset_labels)}

    # Plot each metric
    for idx, (metric_key, ylabel) in enumerate(metrics_to_plot):
        ax = axes[idx]
        metric_data = plot_data[plot_data['Metric'] == metric_key]

        # Plot violin and box plots
        sns.violinplot(
            data=metric_data,
            x="Value",
            y="Dataset",
            hue="Dataset",
            palette=palette,
            ax=ax,
            cut=0,
            linewidth=1.0,
            inner=None
        )
        sns.boxplot(
            data=metric_data,
            x="Value",
            y="Dataset",
            palette=palette,
            ax=ax,
            boxprops=dict(alpha=0.2),
            fliersize=2,
            whis=1.5,
            linewidth=0.5,
        )

        # Annotate medians
        for i, dataset_label in enumerate(dataset_labels):
            median_value = metric_data[metric_data['Dataset'] == dataset_label]['Value'].median()
            ax.text(
                median_value,
                i,
                f'{median_value:.2f}',
                ha='center',
                va='center',
                fontsize=8,
                color='black',
                path_effects=[
                    path_effects.Stroke(linewidth=2, foreground='white'),
                    path_effects.Normal()
                ]
            )

        # Add vertical reference lines
        if vertical_lines:
            for vline in vertical_lines:
                ax.axhline(y=vline + 0.5, color='gray', linestyle='--', linewidth=1.0)

        # Customize axes
        ax.set_xlabel(ylabel)
    
    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
    
    axes[0].set_yticks(np.arange(len(dataset_labels)))
    dataset_labels = [l.replace("\\n", "\n") for l in dataset_labels]
    axes[0].set_yticklabels(dataset_labels)
    axes[0].set_ylabel("")

    # Adjust layout and save the plot
    plt.tight_layout()
    output_path = os.path.join(output_directory, "fingerprint_comparison.pdf")
    plt.savefig(output_path)
    plt.show()

def save_figure(
    fig,
    output_path,
):
    fig.savefig(output_path, dpi=200, bbox_inches='tight')

def compare_crystal_system_metrics(
    dataset_dict,
    dataset_labels,
    output_directory,
    legend_ncol = 3,
    x_anchor = 0.6,
    y_anchor = 1.1,
) -> None:
    # Mapping spacegroup numbers to crystal systems
    spacegroup_to_crystal_system = {
        'Triclinic': range(1, 3),
        'Monoclinic': range(3, 16),
        'Orthorhombic': range(16, 75),
        'Tetragonal': range(75, 143),
        'Trigonal': range(143, 168),
        'Hexagonal': range(168, 195),
        'Cubic': range(195, 231)
    }

    # Function to map spacegroup number to its corresponding crystal system
    def get_crystal_system(spacegroup_number):
        try:
            sg_number = int(spacegroup_number)
            for system, sg_range in spacegroup_to_crystal_system.items():
                if sg_number in sg_range:
                    return system
            return 'Unknown'
        except (ValueError, TypeError):
            return 'Unknown'

    metrics_to_plot = [
        ('rwp', r"$R_{wp}$"),
        ('wd', "WD"),
    ]

    crystal_systems = list(spacegroup_to_crystal_system.keys())

    # Create a subplot for each metric and an additional one for sample distribution
    fig, axs = plt.subplots(1, len(metrics_to_plot) + 1, figsize=(5, 5), sharey=True, gridspec_kw={'width_ratios': [1] * len(metrics_to_plot) + [0.5]})
    if isinstance(axs, Axes):
        axs = [axs]

    for metric_idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        mean_dataframes = []
        for dataset_label in dataset_labels:
            dataset = dataset_dict[dataset_label].copy()
            if isinstance(dataset, pd.DataFrame):
                df = dataset.copy()
            elif isinstance(dataset, dict):
                df = pd.DataFrame.from_dict(dataset)
            else:
                raise ValueError(f"{dataset_label} is not a DataFrame or dict")

            df = df.dropna(subset=['spacegroup_num_sample'])
            df['crystal_system'] = df['spacegroup_num_sample'].apply(get_crystal_system)
            df = df[df['crystal_system'] != 'Unknown']

            grouped = df.groupby('crystal_system')
            mean_values = grouped[metric_key].mean().reset_index()
            std_values = grouped[metric_key].std().reset_index()
            mean_values['std'] = std_values[metric_key]
            mean_values['Dataset'] = dataset_label
            mean_dataframes.append(mean_values)

        combined_data = pd.concat(mean_dataframes)
        mean_pivot = combined_data.pivot(index='crystal_system', columns='Dataset', values=metric_key).reindex(crystal_systems)
        std_pivot = combined_data.pivot(index='crystal_system', columns='Dataset', values='std').reindex(crystal_systems)

        bar_positions = range(len(crystal_systems))
        bar_width = 0.2
        offsets = [i * bar_width - (len(dataset_labels) * bar_width / 2 - bar_width / 2) for i in range(len(dataset_labels))]
        offsets = list(reversed(offsets))

        for label_idx, dataset_label in enumerate(dataset_labels):
            means = mean_pivot[dataset_label].values
            stds = std_pivot[dataset_label].values
            lower_errors = np.clip(means - stds, a_min=0, a_max=None)
            upper_errors = stds

            axs[metric_idx].barh(
                [pos + offsets[label_idx] for pos in bar_positions],
                means,
                xerr=[means - lower_errors, upper_errors],
                height=bar_width,
                label=dataset_label,
                capsize=0,
                edgecolor='k',
                linewidth=1.0,
                error_kw = {"elinewidth": 1.0},
            )

        axs[metric_idx].set_xlabel(metric_label)

    # Plot the distribution of samples per crystal system
    sample_counts = []
    for dataset_label in dataset_labels:
        dataset = dataset_dict[dataset_label].copy()
        dataset['crystal_system'] = dataset['spacegroup_num_sample'].apply(get_crystal_system)
        counts = dataset['crystal_system'].value_counts().reindex(crystal_systems, fill_value=0)
        sample_counts.append(counts)

    total_counts = sum(sample_counts)
    axs[-1].barh(
        bar_positions,
        total_counts.values,
        height=0.5,
        color='grey',
        edgecolor='black',
        linewidth=1.0,
    )
    axs[-1].set_xlabel("Sample\nDistribution")
    axs[-1].set_xticks([])
    axs[-1].spines['right'].set_visible(False)
    axs[-1].spines['top'].set_visible(False)
    axs[-1].spines['bottom'].set_visible(False)

    axs[0].set_yticks(bar_positions)
    axs[0].set_yticklabels(crystal_systems)

    for ax in axs[:-1]:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))

    # Add a shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    labels = (l.replace("\\n", "\n") for l in labels)
    
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(x_anchor, y_anchor),
        ncol=legend_ncol,
        fontsize=9,
    )

    plt.tight_layout()  # Adjust layout
    output_path = os.path.join(output_directory, "crystal_system_metric_comparison.pdf")
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

def plot_metrics_vs_cif_length_histogram(
    df_data,
    labels,
    output_folder,
    num_bins=250,
) -> None:

    # Define the metrics you want to plot
    metrics = [
        ('rwp', r"$R_{wp}$"),
        ('wd', "WS"),
    ]

    # Combine dataframes and add a 'Dataset' column
    combined_df_list = []
    for label in labels:
        df = df_data[label].copy()
        df['Dataset'] = label
        # Compute length of 'cif_gen' string
        df['cif_gen_length'] = df['cif_gen'].apply(lambda x: len(str(x)) if x is not None else 0)
        combined_df_list.append(df)
    combined_df = pd.concat(combined_df_list, ignore_index=True)
    
    # Ensure 'cif_gen_length' and metrics are numeric
    combined_df['cif_gen_length'] = pd.to_numeric(combined_df['cif_gen_length'], errors='coerce')

    # Drop rows with missing values
    combined_df.dropna(subset=['cif_gen_length'], inplace=True)
    
    # Plotting mean values per crystal system across datasets using matplotlib
    fig, axs = plt.subplots(len(metrics),1,figsize=(14, 10), sharex=True)
    for i, (metric_key, metric_label) in enumerate(metrics):

        # Drop rows with missing metric values
        metric_df = combined_df.dropna(subset=[metric_key])

        # Bin the 'cif_gen_length' into 'num_bins' bins and calculate the mean of the metric
        metric_df['length_bin'] = pd.cut(metric_df['cif_gen_length'], bins=num_bins)
        grouped = metric_df.groupby('length_bin').agg(
            mean_metric=(metric_key, 'mean'),
            bin_center=('cif_gen_length', lambda x: (x.min() + x.max()) / 2)
        ).reset_index()

        # Plot using seaborn.barplot
        sns.barplot(
            data=grouped,
            x='bin_center',
            y='mean_metric',
            ax=axs[i],
            edgecolor='black',
            color='skyblue'
        )

        axs[i].set_ylabel(f'Mean {metric_label}')
    
    axs[0].set_title(f'Mean Metrics vs CIF Generated Length')
    plt.xticks(rotation=45, ha='right')
    axs[-1].set_xlabel('CIF Generated Length Bins')
    fig.tight_layout()
    plt.show()
    filename = os.path.join(output_folder, f'{metric_key}_mean_vs_cif_gen_length_histogram.png')
    plt.savefig(filename, dpi=200, bbox_inches='tight')

def extract_validity_stats(df):
    validity_columns = [
        'formula_validity',
        'spacegroup_validity',
        'bond_length_validity',
        'site_multiplicity_validity',
        'validity',
    ]
    
    # Ensure the columns are treated as boolean
    df[validity_columns] = df[validity_columns].astype(bool)

    # Calculate the percentage of valid entries for each metric (mean and std)
    validity_stats_mean = df[validity_columns].mean() * 100
    validity_stats_std = df[validity_columns].std() * 100
    
    # Combine mean and standard deviation into a single DataFrame
    validity_stats = pd.DataFrame({
        'mean (%)': validity_stats_mean,
        'std (%)': validity_stats_std
    })

    return validity_stats

def escape_underscores(text):
    """Escape underscores in text to avoid LaTeX interpretation as subscript."""
    return text.replace("_", r"\_")

def replace_underscores(text, replace_with=' '):
    """Escape underscores in text to avoid LaTeX interpretation as subscript."""
    return text.replace("_", replace_with)

def latex_to_png(latex_code, output_folder, filename='output', dpi=300):
    # Step 1: Write LaTeX code to a .tex file
    tex_filename = os.path.join(output_folder, f'{filename}.tex')
    with open(tex_filename, 'w') as f:
        f.write(r"""
        \documentclass{standalone}
        \usepackage{amsmath, booktabs}
        \begin{document}
        """ + latex_code + r"""
        \end{document}
        """)

    try:
        # Step 2: Compile the .tex file to a DVI file using `latex`
        subprocess.run(['latex', tex_filename], check=True)

        # Step 3: Convert the .dvi file to PNG using `dvipng`
        dvi_filename = filename + '.dvi'
        subprocess.run(['dvipng', '-D', str(dpi), '-T', 'tight', '-o', os.path.join(output_folder, filename) + '.png', dvi_filename], check=True)

        print(f"PNG image successfully created: {filename}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    finally:
        # Optional: Clean up the intermediate files (DVI, AUX, LOG)
        for ext in ['aux', 'log', 'dvi']:
             if os.path.exists(f'{filename}.{ext}'):
                 os.remove(f'{filename}.{ext}')
        pass

def validity_comparison(
    df_data,
    labels,
    output_folder,
    names=None,
):
    results = []
    for label in labels:
        # Extract stats and add the dataset label
        validity_stats = extract_validity_stats(df_data[label])
        validity_stats['Dataset'] = escape_underscores(label)
        validity_stats.reset_index(inplace=True)  # Convert the index into a column
        results.append(validity_stats)

    # Combine all results into a DataFrame
    results_df = pd.concat(results).reset_index(drop=True)

    # Pivot the DataFrame to get columns for each validity metric
    results_df = results_df.pivot(index='Dataset', columns='index', values=['mean (%)', 'std (%)'])

    # Debug: Print columns before renaming
    print("Columns before renaming:", results_df.columns.tolist())

    # Adjust column names to remove extra parentheses
    results_df.columns = [f"{col[1]} ({col[0].replace(' (%)', ' %')})" for col in results_df.columns]

    # Debug: Print columns after renaming
    print("Columns after renaming:", results_df.columns.tolist())

    # Calculate max values for each mean column
    mean_columns = [col for col in results_df.columns if '(mean %)' in col]
    max_values = results_df[mean_columns].max()

    # Create LaTeX table string
    table_str = r"""
\begin{tabular}{lccccc}
\midrule
\text{Dataset} & \text{Formula Validity (\%)}$\;\uparrow$ & \text{Spacegroup Validity (\%)}$\;\uparrow$ & \text{Bond Length Validity (\%)}$\;\uparrow$ & \text{Site Multiplicity Validity (\%)}$\;\uparrow$ & \text{Overall Validity (\%)}$\;\uparrow$ \\
\midrule
"""

    # Add rows from DataFrame to the LaTeX string
    for idx, row in results_df.reset_index().iterrows():
        if names is not None and len(names) > idx:
            dataset_name = replace_underscores(names[idx])
        else:
            dataset_name = row['Dataset']

        table_str += f"\\text{{{dataset_name}}} & "

        # Formula Validity
        col_mean = 'formula_validity (mean %)'
        col_std = 'formula_validity (std %)'
        if col_mean in row and col_std in row:
            if row[col_mean] == max_values[col_mean]:
                table_str += f"\\textbf{{{row[col_mean]:.2f}}} ± {row[col_std]:.2f} & "
            else:
                table_str += f"{row[col_mean]:.2f} ± {row[col_std]:.2f} & "
        else:
            table_str += "N/A & "

        # Spacegroup Validity
        col_mean = 'spacegroup_validity (mean %)'
        col_std = 'spacegroup_validity (std %)'
        if col_mean in row and col_std in row:
            if row[col_mean] == max_values[col_mean]:
                table_str += f"\\textbf{{{row[col_mean]:.2f}}} ± {row[col_std]:.2f} & "
            else:
                table_str += f"{row[col_mean]:.2f} ± {row[col_std]:.2f} & "
        else:
            table_str += "N/A & "

        # Bond Length Validity
        col_mean = 'bond_length_validity (mean %)'
        col_std = 'bond_length_validity (std %)'
        if col_mean in row and col_std in row:
            if row[col_mean] == max_values[col_mean]:
                table_str += f"\\textbf{{{row[col_mean]:.2f}}} ± {row[col_std]:.2f} & "
            else:
                table_str += f"{row[col_mean]:.2f} ± {row[col_std]:.2f} & "
        else:
            table_str += "N/A & "

        # Site Multiplicity Validity
        col_mean = 'site_multiplicity_validity (mean %)'
        col_std = 'site_multiplicity_validity (std %)'
        if col_mean in row and col_std in row:
            if row[col_mean] == max_values[col_mean]:
                table_str += f"\\textbf{{{row[col_mean]:.2f}}} ± {row[col_std]:.2f} & "
            else:
                table_str += f"{row[col_mean]:.2f} ± {row[col_std]:.2f} & "
        else:
            table_str += "N/A & "

        # Overall Validity
        col_mean = 'validity (mean %)'
        col_std = 'validity (std %)'
        if col_mean in row and col_std in row:
            if row[col_mean] == max_values[col_mean]:
                table_str += f"\\textbf{{{row[col_mean]:.2f}}} ± {row[col_std]:.2f} \\\\\n"
            else:
                table_str += f"{row[col_mean]:.2f} ± {row[col_std]:.2f} \\\\\n"
        else:
            table_str += "N/A \\\\\n"

    # Close the table
    table_str += r"\bottomrule" + "\n"
    table_str += r"\end{tabular}"

    # Generate LaTeX table as an image
    latex_to_png(table_str, output_folder, filename="validity")

def extract_metrics_stats(df):
    metrics_columns = [
        'rwp',
        'wd',
    ]

    # Ensure the columns are numeric
    df_metrics = df[metrics_columns].apply(pd.to_numeric, errors='coerce')
    
    # Calculate the mean and standard deviation for each metric
    metrics_mean = df_metrics.mean()
    metrics_std = df_metrics.std()
    
    # Combine mean and standard deviation into a single DataFrame
    metrics_stats = pd.DataFrame({
        'mean': metrics_mean,
        'std': metrics_std
    })
    metrics_stats['Metric'] = metrics_stats.index  # Add the metric names as a column
    
    return metrics_stats.reset_index(drop=True)

def metrics_comparison(
    df_data,
    labels,
    output_folder,
    names=None,
):
    results = []
    for idx, label in enumerate(labels):
        # Extract stats and add the dataset label
        metrics_stats = extract_metrics_stats(df_data[label])
        dataset_name = escape_underscores(label)
        if names is not None and len(names) > idx:
            dataset_name = escape_underscores(names[idx])
        metrics_stats['Dataset'] = dataset_name
        results.append(metrics_stats)
    
    # Combine all results into a single DataFrame
    results_df = pd.concat(results)
    
    # Pivot the DataFrame to get datasets as rows and metrics as columns
    pivot_df = results_df.pivot(index='Dataset', columns='Metric', values=['mean', 'std'])
    
    # Flatten MultiIndex columns
    pivot_df.columns = [f"{metric} ({stat})" for stat, metric in pivot_df.columns]
    
    # Define metrics and their display names
    metrics_info = {
        'rwp': {'display_name': r"$R_{wp}$", 'better': 'lower'},
        'wd': {'display_name': "WD", 'better': 'lower'},
    }
    
    # Prepare the LaTeX table header
    table_str = r"""
\begin{tabular}{l""" + "c" * len(metrics_info) + r"""}
\toprule
\textbf{Dataset}"""
    for metric_key in metrics_info:
        table_str += f" & \\textbf{{{metrics_info[metric_key]['display_name']}}}"
    table_str += r""" \\
\midrule
"""

    # Determine the best values for each metric
    best_values = {}
    for metric_key, info in metrics_info.items():
        col_mean = f"{metric_key} (mean)"
        if info['better'] == 'higher':
            best_values[metric_key] = pivot_df[col_mean].max()
        else:
            best_values[metric_key] = pivot_df[col_mean].min()
    
    # Add rows to the LaTeX table
    for dataset_name, row in pivot_df.iterrows():
        table_str += f"\\text{{{dataset_name}}}"
        for metric_key, info in metrics_info.items():
            col_mean = f"{metric_key} (mean)"
            col_std = f"{metric_key} (std)"
            mean_value = row.get(col_mean, np.nan)
            std_value = row.get(col_std, np.nan)
            if pd.notna(mean_value) and pd.notna(std_value):
                is_best = False
                if info['better'] == 'higher' and mean_value == best_values[metric_key]:
                    is_best = True
                elif info['better'] == 'lower' and mean_value == best_values[metric_key]:
                    is_best = True
                value_str = f"{mean_value:.2f} $\\pm$ {std_value:.2f}"
                if is_best:
                    value_str = f"\\textbf{{{value_str}}}"
                table_str += f" & {value_str}"
            else:
                table_str += " & N/A"
        table_str += r" \\" + "\n"

    # Close the table
    table_str += r"""\bottomrule
\end{tabular}
"""

    # Generate LaTeX table as an image
    latex_to_png(table_str, output_folder, filename="metrics_comparison")
    print(f"LaTeX table saved as image: 'metrics_comparison.png'")

def plot_validity_vs_cif_length(
    df_data,
    labels,
    output_folder,
    num_bins=75,
    figsize=(5, 5),
):
    fig, axes = plt.subplots(len(labels), 1, figsize=figsize, sharex=True)

    # Combine all data for the background distribution
    all_data = pd.concat(df_data.values(), ignore_index=True)
    all_data = all_data.dropna(subset=['seq_len_gen'])
    all_data['seq_len_gen'] = pd.to_numeric(all_data['seq_len_gen'], errors='coerce')
    all_data = all_data.dropna(subset=['seq_len_gen'])

    # Bin 'seq_len_gen' for the background distribution
    bins = np.linspace(all_data['seq_len_gen'].min(), all_data['seq_len_gen'].max(), num_bins + 1)

    all_handles_labels = []

    # Overlay frequency counts for each dataset as histograms
    colors = sns.color_palette("tab10")[2:]
    for ax, label, c in zip(axes, labels, colors):
        ax.hist(
            all_data['seq_len_gen'],
            bins=bins,
            histtype='step',
            linestyle=':',
            linewidth=1.0,
            label='Generated Distribution'
        )

        # Add sample distribution as an outlined histogram
        ax.hist(
            all_data['seq_len_sample'],
            bins=bins,
            histtype='step',
            linestyle='--',
            linewidth=1.0,
            label='Sample Distribution'
        )
        df = df_data[label]
        # Ensure 'seq_len_gen' is numeric
        df = df.dropna(subset=['seq_len_gen'])
        df['seq_len_gen'] = pd.to_numeric(df['seq_len_gen'], errors='coerce')
        df = df.dropna(subset=['seq_len_gen'])

        # Plot dataset-specific histogram
        ax.hist(
            df['seq_len_gen'],
            histtype='stepfilled',
            bins=bins,
            alpha=0.7,
            color=c,
            label=f"{label}",
            edgecolor='k',
        )

        # Customize histogram axes
        ax.set_yscale('log')
        ax.grid(alpha=0.2)
        
        # Collect handles and labels for each subplot
        handles, subplot_labels = ax.get_legend_handles_labels()
        all_handles_labels.append((handles, subplot_labels))
        
    axes[-1].set_xlabel('CIF Token Length')

    # Combine all handles and labels for the global legend
    handles, labels = zip(*all_handles_labels)
    handles = [h for sublist in handles for h in sublist]
    labels = [l for sublist in labels for l in sublist]
    unique_labels = dict(zip(labels, handles))

    # Add a single legend at the top of the figure
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper center', ncol=3, fontsize='small')

    # Add a single y-label for the entire figure
    fig.supylabel("Frequency (log scale)", fontsize=10)

    # Adjust layout and spacing
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    # Save the plot
    output_path = os.path.join(output_folder, 'validity_vs_cif_length.pdf')
    fig.savefig(output_path)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to comparison .yaml config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Parse yaml to namespace and merge (DictConfig)
    yaml_dictconfig = OmegaConf.create(yaml_config)
    
    # Create output folder
    assert os.path.exists(yaml_dictconfig.experiment_folder) and os.path.isdir(yaml_dictconfig.experiment_folder)
    comparison_folder = os.path.join(yaml_dictconfig.experiment_folder, "comparisons")
    os.makedirs(comparison_folder, exist_ok=True)

    # Create data if not already present
    df_data = {}
    for label, eval_folder_path in yaml_dictconfig.eval_dict.items():
        # Try to find pre-computed file
        safe_label = label.replace(' ', '_')
        safe_label = re.sub(r'[^\w\-_\.]', '', safe_label)
        comparison_pickle = os.path.join(comparison_folder, safe_label + '.pkl.gz')
        if os.path.exists(comparison_pickle):
            df_data[label] = pd.read_pickle(comparison_pickle)
        else:
            df_data[label] = process(eval_folder_path, yaml_dictconfig.debug_max)
            pd.DataFrame(df_data[label]).to_pickle(comparison_pickle)
    
    labels = yaml_dictconfig.eval_dict.keys()

    if yaml_dictconfig.validity_comparison:
        validity_comparison(
            df_data,
            labels, 
            comparison_folder,
        )

    if yaml_dictconfig.metrics_comparison:
        metrics_comparison(
            df_data, 
            labels, 
            comparison_folder,
        )
    
    if yaml_dictconfig.fingerprint_comparison:
        fingerprint_comparison(
            df_data, 
            labels,
            comparison_folder,
            vertical_lines = yaml_dictconfig.vlines, 
            color_pair_size=yaml_dictconfig.color_pair_size
        )

    if yaml_dictconfig.metrics_vs_seq_len:
        plot_metrics_vs_cif_length_histogram(
            df_data, 
            labels, 
            comparison_folder,
        )
    
    if yaml_dictconfig.crystal_system_metric_comparison:
        compare_crystal_system_metrics(
            df_data, 
            labels, 
            comparison_folder,
            legend_ncol=yaml_dictconfig.legend_ncol,
            x_anchor=yaml_dictconfig.x_anchor, 
            y_anchor=yaml_dictconfig.y_anchor
        )

    if yaml_dictconfig.validity_vs_seq_len:
        plot_validity_vs_cif_length(
            df_data, 
            labels, 
            comparison_folder,
        )
