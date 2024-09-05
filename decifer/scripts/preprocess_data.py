import sys
sys.path.append("./")
import os
import io
import pickle
import argparse
from pymatgen.io.cif import CifWriter, Structure
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool, cpu_count

import warnings
warnings.filterwarnings("ignore")

def create_stratification_key(pmg_structure, group_size):
    """
    Create a stratification key from the spacegroup.

    Args:
        pmg_structure: pymatgen Structure object.
        group_size (int): Size of the spacegroup bin.

    Returns:
        str: Stratification key combining spacegroup in defined groups.
    """
    
    spacegroup_number = pmg_structure.get_space_group_info()[1]
    group_start = ((spacegroup_number - 1) // group_size) * group_size + 1
    group_end = group_start + group_size - 1

    return f"{group_start}-{group_end}"

def process_single_cif(args):
    """
    Process a single CIF file to extract its content and spacegroup-based stratification key.

    Args:
        args (tuple): Contains (path, spacegroup_group_size, debug).
            path (str): Path to the CIF file.
            spacegroup_group_size (int): Spacegroup bin size for stratification.
            debug (bool): If True, print errors for failed CIF processing.

    Returns:
        tuple: (strat_key, cif_content) if processing is successful, else None.
    """
    path, spacegroup_group_size, debug = args
    try:
        struct = Structure.from_file(path)
        strat_key = create_stratification_key(struct, spacegroup_group_size)
        cif_content = CifWriter(struct=struct, symprec=0.1).__str__()
        return (strat_key, cif_content)
    except Exception as e:
        if debug:
            print(f"Error processing {path}: {e}")
        return None

def preprocess(data_dir, seed, spacegroup_group_size, debug_max=None, debug=False):
    """
    Preprocess CIF files by extracting their structure and creating a train/val/test split.

    Args:
        data_dir (str): Directory containing the raw CIF files.
        seed (int): Random seed for reproducibility.
        spacegroup_group_size (int): Group size for spacegroup stratification.
        debug_max (int, optional): Maximum number of CIFs to process (for debugging).
        debug (bool, optional): If True, print debugging information.

    Returns:
        None
    """

    # Find raw CIF files
    raw_dir = os.path.join(data_dir, "raw")
    cif_paths = sorted(glob(os.path.join(raw_dir, "*.cif")))[:debug_max]
    assert len(cif_paths) > 0, f"Cannot find any .cif files in {raw_dir}"

    # Output directory
    output_dir = os.path.join(data_dir, "preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare arguments for parallel processing
    tasks = [(path, spacegroup_group_size, debug) for path in cif_paths]

    # Parallel processing of CIF files using multiprocessing
    num_workers = min(cpu_count(), len(cif_paths))  # Use available CPU cores, limited to number of files
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(
            process_single_cif, tasks
        ), total=len(cif_paths), desc="preprocessing CIFs..."))

    # Filter out None results from failed processing
    valid_results = [result for result in results if result is not None]
    strat_keys, cif_contents = zip(*valid_results)

    print(f"No. CIFs before preprocessing: {len(cif_paths)} --> after: {len(cif_contents)}\n")

    # Create data splits
    train_size = int(0.8 * len(cif_contents))
    val_size = int(0.1 * len(cif_contents))
    test_size = len(cif_contents) - train_size - val_size
    print("Train size:", train_size)
    print("Val size:", val_size)
    print("Test size:", test_size, "\n")

    # Split data using stratification
    train_data, test_data = train_test_split(
        cif_contents, test_size=test_size, stratify=strat_keys, random_state=seed
    )
    train_data, val_data = train_test_split(
        train_data, test_size=val_size, stratify=[strat_keys[cif_contents.index(f)] for f in train_data], random_state=seed
    )

    # Save the train/val/test data splits
    for data, prefix in zip([train_data, val_data, test_data], ["train", "val", "test"]):
        print(f"Saving {prefix} dataset...", end="")
        with open(os.path.join(output_dir, f"{prefix}_dataset.pkl.gz"), "wb") as pkl:
            pickle.dump(data, pkl)
        print("DONE.")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare custom CIF files and save to a tar.gz file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, help="Path to the outer data directory", required=True)
    parser.add_argument("--group_size", type=int, help="Spacegroup group size for stratification", default=10)

    parser.add_argument("--preprocess", help="Preprocess files", action="store_true")
    parser.add_argument("--diffraction", help="Calculate XRD", action="store_true")  # Placeholder for future implementation
    parser.add_argument("--tokenize", help="Tokenize CIFs", action="store_true")  # Placeholder for future implementation

    parser.add_argument("--debug_max", help="Debug-feature: max number of files to process", type=int, default=0)
    parser.add_argument("--debug", help="Debug-feature: whether to print debug messages", action="store_true")

    args = parser.parse_args()

    # Adjust debug_max if no limit is specified
    if args.debug_max == 0:
        args.debug_max = None

    if args.preprocess:
        preprocess(args.data_dir, args.seed, args.group_size, args.debug_max, args.debug)
        print(f"Prepared CIF files have been saved to {os.path.join(args.data_dir, 'preprocessed')}")
