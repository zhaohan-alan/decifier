import numpy as np
import pandas as pd
import os
import argparse
import subprocess

def extract_validity_stats(df):
    """
    Extract validity statistics from the dataset.

    Parameters:
        df (pd.DataFrame): DataFrame containing validity columns.

    Returns:
        pd.DataFrame: DataFrame with validity statistics.
    """
    validity_columns = ['validity.formula', 'validity.spacegroup', 'validity.bond_length', 'validity.site_multiplicity']
    
    # Ensure the columns are treated as boolean
    df[validity_columns] = df[validity_columns].astype(bool)
    
    # Calculate the percentage of valid entries for each metric
    validity_stats = df[validity_columns].mean() * 100
    
    return validity_stats

def dataset_validity_table(eval_paths):
    """
    Extract validity statistics from multiple datasets and prepare a comparison table.

    Parameters:
        eval_paths (list): List of paths to evaluation files.

    Returns:
        pd.DataFrame: DataFrame containing the validity statistics comparison table.
    """
    results = []

    # Process each dataset
    for path in eval_paths:
        dataset_name = os.path.basename(path).split('.')[0]
        
        # Load dataset and calculate validity stats
        df = pd.read_parquet(path)
        validity_stats = extract_validity_stats(df)
        
        # Add the dataset name to the stats
        validity_stats['Dataset'] = escape_underscores(dataset_name)
        results.append(validity_stats)

    # Combine all results into a DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df[['Dataset', 'validity.formula', 'validity.spacegroup', 'validity.bond_length', 'validity.site_multiplicity']]
    
    return results_df

def escape_underscores(text):
    """Escape underscores in text to avoid LaTeX interpretation as subscript."""
    return text.replace("_", r"\_")

def replace_underscores(text, replace_with=' '):
    """Escape underscores in text to avoid LaTeX interpretation as subscript."""
    return text.replace("_", replace_with)

def latex_to_png(latex_code, output_filename='output.png', dpi=300):
    # Step 1: Write LaTeX code to a .tex file
    tex_filename = 'latex_input.tex'
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
        dvi_filename = tex_filename.replace('.tex', '.dvi')
        subprocess.run(['dvipng', '-D', str(dpi), '-T', 'tight', '-o', output_filename, dvi_filename], check=True)

        print(f"PNG image successfully created: {output_filename}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    finally:
        # Optional: Clean up the intermediate files (DVI, AUX, LOG)
        for ext in ['aux', 'log', 'dvi', 'tex']:
            if os.path.exists(f'latex_input.{ext}'):
                os.remove(f'latex_input.{ext}')

def main(eval_paths, output_path, names=None):
    # Generate the validity comparison table
    results_df = dataset_validity_table(eval_paths)

    # Find the maximum values for each column (except 'Dataset')
    max_values = results_df[['validity.formula', 'validity.spacegroup', 'validity.bond_length', 'validity.site_multiplicity']].max()

    # Create LaTeX-like string to display
    table_str = r"""
    \begin{tabular}{lcccc}
    \midrule
    \text{Dataset} & \text{Formula Validity (\%)}$\;\uparrow$ & \text{Spacegroup Validity (\%)}$\;\uparrow$ & \text{Bond Length Validity (\%)}$\;\uparrow$ & \text{Site Multiplicity Validity (\%)}$\;\uparrow$ \\
    \midrule
    """

    # Add rows from DataFrame to the LaTeX string, bold the maximum values
    for idx, row in results_df.iterrows():
        if names is not None and len(names) > idx:
            row['Dataset'] = replace_underscores(names[idx])

        table_str += f"\\text{{{row['Dataset']}}} & "

        # Formula Validity
        if row['validity.formula'] == max_values['validity.formula']:
            table_str += f"\\textbf{{{row['validity.formula']:.2f}}} & "
        else:
            table_str += f"{row['validity.formula']:.2f} & "
        
        # Spacegroup Validity
        if row['validity.spacegroup'] == max_values['validity.spacegroup']:
            table_str += f"\\textbf{{{row['validity.spacegroup']:.2f}}} & "
        else:
            table_str += f"{row['validity.spacegroup']:.2f} & "

        # Bond Length Validity
        if row['validity.bond_length'] == max_values['validity.bond_length']:
            table_str += f"\\textbf{{{row['validity.bond_length']:.2f}}} & "
        else:
            table_str += f"{row['validity.bond_length']:.2f} & "

        # Site Multiplicity Validity
        if row['validity.site_multiplicity'] == max_values['validity.site_multiplicity']:
            table_str += f"\\textbf{{{row['validity.site_multiplicity']:.2f}}} \\\\\n"
        else:
            table_str += f"{row['validity.site_multiplicity']:.2f} \\\\\n"

    # Close the table
    table_str += r"\bottomrule" + "\n"
    table_str += r"\end{tabular}"

    latex_to_png(table_str, output_filename=output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and compare validity metrics for multiple datasets.")
    parser.add_argument('eval_paths', nargs='+', help='Paths to evaluation files (.eval).')
    parser.add_argument('--names', nargs='+', help="Names for the datasets in order of appearance.")
    parser.add_argument('--output-path', type=str, help='Output path of the .png file.', default='similarity_table.png')
    
    args = parser.parse_args()
    main(args.eval_paths, args.output_path, args.names)
