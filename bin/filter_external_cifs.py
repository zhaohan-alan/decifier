#!/usr/bin/env python

import argparse
import h5py
import pickle
import difflib
from multiprocessing import Pool
from tqdm import tqdm

def is_unique(mp20_cif, crystal_cifs, threshold=0.9):
    """
    Returns True if mp20_cif is not too similar to any cif in crystal_cifs.
    The inner loop is wrapped with tqdm to show progress for each comparison.
    """
    for train_cif in tqdm(crystal_cifs, desc="Comparing with training data", leave=False, disable=True):
        if difflib.SequenceMatcher(None, mp20_cif, train_cif).ratio() > threshold:
            return False
    return True

def worker(args):
    # Top-level worker function to allow pickling
    return is_unique(*args)

def main():
    parser = argparse.ArgumentParser(
        description="Filter mp-20 CIFs by comparing to crystallm dataset and keeping unique ones."
    )
    parser.add_argument("--crystallm_path", type=str, required=True,
                        help="Path to the crystallm HDF5 file (dataset 'cif_string').")
    parser.add_argument("--mp20_path", type=str, required=True,
                        help="Path to the mp-20 pickle file (with key 'cif').")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Similarity threshold (default: 0.9).")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional output file path to save filtered mp-20 CIFs (pickle format).")
    parser.add_argument("--processes", type=int, default=None,
                        help="Number of processes to use (default: uses all available CPUs).")
    args = parser.parse_args()

    # Load crystallm dataset
    print("Loading crystallm dataset from:", args.crystallm_path)
    with h5py.File(args.crystallm_path, "r") as h5_file:
        crystal_cifs = h5_file["cif_string"][:]

    # Load mp-20 dataset
    print("Loading mp-20 dataset from:", args.mp20_path)
    with open(args.mp20_path, "rb") as f:
        mp20_data = pickle.load(f)
    mp20_cifs = list(mp20_data["cif"][:])

    # Prepare arguments for multiprocessing
    pool_args = [(mp20_cif, crystal_cifs, args.threshold) for mp20_cif in mp20_cifs]

    print("Starting filtering using multiprocessing...")
    with Pool(processes=args.processes) as pool:
        # Outer progress bar for the overall mp-20 CIFs filtering.
        results = list(tqdm(pool.imap(worker, pool_args), total=len(pool_args)))
    
    # Select only the unique mp-20 CIFs
    filtered_mp20 = [cif for cif, unique in zip(mp20_cifs, results) if unique]
    
    print(f"Total mp-20 CIFs: {len(mp20_cifs)}")
    print(f"Filtered (unique) mp-20 CIFs: {len(filtered_mp20)}")

    # Optionally save the filtered results to a pickle file
    if args.output:
        with open(args.output, "wb") as f:
            pickle.dump(filtered_mp20, f)
        print(f"Filtered mp-20 CIFs saved to {args.output}")

if __name__ == '__main__':
    main()
