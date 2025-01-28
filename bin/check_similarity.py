import numpy as np
import pandas as pd
import os
import ot
import argparse
import subprocess
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def rbf_kernel(X, Y=None, gamma=1.0):
    """Compute the RBF (Gaussian) kernel between X and Y."""
    if Y is None:
        Y = X
    dist = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1)[np.newaxis, :] - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * dist)

def compute_mmd(X, Y, gamma=1.0):
    """Compute Maximum Mean Discrepancy (MMD) between two sets of samples."""
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd

def compute_wasserstein_nd(X, Y):
    """Compute the Wasserstein distance between two multi-dimensional distributions."""
    M = ot.dist(X, Y, metric='euclidean')
    n = X.shape[0]
    m = Y.shape[0]
    p = np.ones(n) / n  # Uniform distribution for X
    q = np.ones(m) / m  # Uniform distribution for Y
    dist = ot.emd2(p, q, M)
    return dist

def compute_proxy_a_distance(X, Y):
    """Compute Proxy-A-Distance (PAD) between two distributions using a Random Forest classifier."""
    # Combine datasets
    X_combined = np.vstack([X, Y])
    y_combined = np.hstack([np.zeros(X.shape[0]), np.ones(Y.shape[0])])  # 0 for X, 1 for Y
    
    # Train a Random Forest classifier and compute cross-validated accuracy
    clf = RandomForestClassifier(n_estimators=1000, random_state=42)
    error_rates = 1 - cross_val_score(clf, X_combined, y_combined, cv=5, scoring='accuracy')
    avg_error_rate = np.mean(error_rates)
    
    # Compute PAD based on classification error rate
    pad_value = 2 * (1 - 2 * avg_error_rate)
    return pad_value

def dataset_feature_vectors(eval_path, columns_to_concat):
    df = pd.read_parquet(eval_path)
    df = df.dropna(subset=columns_to_concat)
    return np.vstack(df[columns_to_concat].apply(lambda row: np.array(row.values.tolist()), axis=1))

def escape_underscores(text):
    """Escape underscores in text to avoid LaTeX interpretation as subscript."""
    return text.replace("_", r"\_")

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

def main(eval_paths, output_path):
    # Configuration
    columns_to_concat = [
        'cell_params.a', 
        'cell_params.b',
        'cell_params.c',
        'cell_params.alpha', 
        'cell_params.beta', 
        'cell_params.gamma', 
        'cell_params.implied_vol',
        'cell_params.gen_vol',
        'seq_len',
    ]
    
    eval_names = [os.path.basename(path).split('.')[0] for path in eval_paths]

    # Extract feature vectors
    feature_vectors = []
    for path in eval_paths:
        vec = dataset_feature_vectors(path, columns_to_concat)
        feature_vectors.append(vec)

    # Initialize results table
    results = []

    # Compute pairwise MMD, Wasserstein, and PAD
    for i in range(len(eval_names)):
        for j in range(i + 1, len(eval_names)):
            mmd_value = compute_mmd(feature_vectors[i], feature_vectors[j], gamma=1.0)
            dist_nd = compute_wasserstein_nd(feature_vectors[i], feature_vectors[j])
            pad_value = compute_proxy_a_distance(feature_vectors[i], feature_vectors[j])

            # Store results in a list
            results.append({
                "Dataset 1": escape_underscores(eval_names[i]),
                "Dataset 2": escape_underscores(eval_names[j]),
                "MMD": mmd_value,
                "Wasserstein Distance": dist_nd,
                "Proxy-A-Distance": pad_value
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Create LaTeX-like string to display
    table_str = r"""
    \begin{tabular}{llccc}
    \midrule
    \text{Dataset 1} & \text{Dataset 2} & \text{MMD}$\;\downarrow$ & \text{WD}$\;\downarrow$ & \text{PAD-RF}$\;\uparrow$ \\
    \midrule
    """

    # Add rows from DataFrame to the LaTeX string
    for _, row in results_df.iterrows():
        table_str += f"\\text{{{row['Dataset 1']}}} & \\text{{{row['Dataset 2']}}} & {row['MMD']:.5f} & {row['Wasserstein Distance']:.3f} & {row['Proxy-A-Distance']:.3f} \\\\\n"
        
    # Close the table
    table_str += r"\bottomrule" + "\n"
    table_str += r"\end{tabular}"

    latex_to_png(table_str, output_filename=output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MMD, Wasserstein, and Proxy-A-Distance for combinations of eval files.")
    parser.add_argument('eval_paths', nargs='+', help='Paths to evaluation files (.eval).')
    parser.add_argument('--output-path', type=str, help='Output path of the .png file.', default='similarity_table.png')
    
    args = parser.parse_args()
    main(args.eval_paths, args.output_path)
