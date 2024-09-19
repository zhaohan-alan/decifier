import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
import argparse

# Suppress the specific warning related to use_inf_as_na
warnings.filterwarnings("ignore", category=FutureWarning, message="use_inf_as_na option is deprecated")


def process_and_save_plots_and_stats(parquet_file_path, output_folder):
    # Set Seaborn style
    sns.set(style='whitegrid')

    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(parquet_file_path)

    # Handle infinite values and convert them to NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Rename cell parameter columns if necessary
    cell_param_cols = [col for col in df.columns if col.startswith('cell_params.')]

    if cell_param_cols:
        df.rename(columns=lambda x: x.replace('cell_params.', '') if x.startswith('cell_params.') else x, inplace=True)
    else:
        print("No 'cell_params' columns found.")

    # Numeric columns to check
    numeric_cols = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'implied_vol', 'gen_vol']

    # Convert numeric columns to appropriate data types
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # **Create an overall 'syntax_validity' column**
    syntax_validity_checks = [
        'syntax_validity.formula_consistency',
        'syntax_validity.atom_site_multiplicity',
        'syntax_validity.space_group_consistency'
    ]

    # Ensure these columns exist
    missing_checks = [check for check in syntax_validity_checks if check not in df.columns]
    if missing_checks:
        print(f"Missing syntax validity checks: {missing_checks}")
        for check in missing_checks:
            df[check] = False  # or handle as appropriate

    # Ensure that these columns are boolean
    df[syntax_validity_checks] = df[syntax_validity_checks].astype(bool)

    # **Combine individual checks into an overall syntax validity**
    df['syntax_validity'] = df[syntax_validity_checks].all(axis=1)

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save statistics table as a CSV
    generate_statistics_table(df, output_folder)

    # Generate and save the plots
    generate_plots(df, output_folder)


def generate_statistics_table(df, output_folder):
    # Columns required for evaluation
    required_columns = [
        'Dataset', 'Model', 'cif',
        'syntax_validity.space_group_consistency',
        'syntax_validity.atom_site_multiplicity',
        'syntax_validity.bond_length_acceptability',
    ]

    # Ensure validity columns are boolean
    df['syntax_validity.space_group_consistency'] = df['syntax_validity.space_group_consistency'].astype(bool)
    df['syntax_validity.atom_site_multiplicity'] = df['syntax_validity.atom_site_multiplicity'].astype(bool)
    df['syntax_validity.bond_length_acceptability'] = df['syntax_validity.bond_length_acceptability'].astype(bool)

    # Calculate overall validity
    df['valid'] = df[['syntax_validity.space_group_consistency', 'syntax_validity.atom_site_multiplicity', 'syntax_validity.bond_length_acceptability']].all(axis=1)

    # Group by Dataset and Model to compute statistics
    grouped = df.groupby(['Dataset', 'Model'])

    # Initialize a list to store the results
    results = []

    # Loop over each group to calculate statistics
    for (dataset, model), group in grouped:
        total_samples = len(group)

        # SG [%]
        sg_pass = group['syntax_validity.space_group_consistency'].sum()
        sg_percentage = (sg_pass / total_samples) * 100

        # ASM [%]
        asm_pass = group['syntax_validity.atom_site_multiplicity'].sum()
        asm_percentage = (asm_pass / total_samples) * 100
        
        # BLR [%]
        blr_pass = group['syntax_validity.bond_length_acceptability'].sum()
        blr_percentage = (blr_pass / total_samples) * 100

        # Valid [%]
        valid_pass = group['valid'].sum()
        valid_percentage = (valid_pass / total_samples) * 100

        # Average valid sequence length
        valid_seq_lengths = group.loc[group['valid'], 'seq_len']
        if not valid_seq_lengths.empty:
            avg_seq_length = valid_seq_lengths.mean()
            std_seq_length = valid_seq_lengths.std()
            seq_length_str = f"{avg_seq_length:.1f} ± {std_seq_length:.1f}"
        else:
            seq_length_str = "N/A"

        # Store the results
        results.append({
            'Dataset': dataset,
            'Model': model,
            'SG [%]': f"{sg_percentage:.1f}",
            'ASM [%]': f"{asm_percentage:.1f}",
            'BLR [%]': f"{blr_percentage:.1f}",
            'Valid [%]': f"{valid_percentage:.1f}",
            'Avg. Valid Seq. Len.': seq_length_str
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the statistics table as CSV
    results_df.to_csv(os.path.join(output_folder, 'statistics_table.csv'), index=False)

    print("Statistics table saved as 'statistics_table.csv'.")


def generate_plots(df, output_folder):
    # Plot histograms for cell lengths (a, b, c) and save them
    plt.figure(figsize=(8, 3))
    for i, param in enumerate(['a', 'b', 'c']):
        plt.subplot(1, 3, i+1)
        sns.histplot(df[param].dropna(), kde=True)
        plt.title(f'Distribution of {param}')
        plt.xlabel(f'Cell Length {param} (Å)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cell_lengths_histogram.png'))
    plt.close()

    # Plot histograms for cell angles (alpha, beta, gamma) and save them
    plt.figure(figsize=(8, 3))
    for i, param in enumerate(['alpha', 'beta', 'gamma']):
        plt.subplot(1, 3, i+1)
        sns.histplot(df[param].dropna(), kde=True)
        plt.title(f'Distribution of {param}')
        plt.xlabel(f'Cell Angle {param} (degrees)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cell_angles_histogram.png'))
    plt.close()

    # Pairplot of cell lengths to examine relationships and save it
    sns.pairplot(df[['a', 'b', 'c']].dropna())
    plt.suptitle('Pairplot of Cell Lengths', y=1.02)
    plt.savefig(os.path.join(output_folder, 'cell_lengths_pairplot.png'))
    plt.close()

    # Bar plot of the top 10 most common space groups and save it
    top_spacegroups = df['spacegroup'].value_counts().head(10)
    sns.barplot(y=top_spacegroups.index, x=top_spacegroups.values, orient='h')
    plt.title('Top 10 Space Groups')
    plt.xlabel('Count')
    plt.ylabel('Space Group')
    plt.savefig(os.path.join(output_folder, 'top_space_groups_barplot.png'))
    plt.close()

    # Scatter plot of implied volume vs. generated volume and save it
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='implied_vol', y='gen_vol', data=df)
    plt.title('Implied Volume vs. Generated Volume')
    plt.xlabel('Implied Volume (Å³)')
    plt.ylabel('Generated Volume (Å³)')
    plt.savefig(os.path.join(output_folder, 'implied_vs_generated_volume_scatterplot.png'))
    plt.close()
    
    print("Statistic plots saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a Parquet file, generate plots and statistics, and save them to a specified folder.')

    # Add arguments for parquet file path and output folder
    parser.add_argument('--eval_file_path', required=True, type=str, help='Path to the eval file.')
    parser.add_argument('--output_folder', required=True, type=str, help='Directory where the plots and statistics will be saved.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function to process data, generate plots, and save statistics
    process_and_save_plots_and_stats(args.eval_file_path, args.output_folder)
