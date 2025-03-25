#!/usr/bin/env python

import argparse
import h5py
import pickle
import difflib
import os
from multiprocessing import Pool
from tqdm import tqdm

def is_unique(mp20_cif, crystal_cifs, threshold=0.9):
    for train_cif in tqdm(crystal_cifs, desc="Comparing with training data", leave=False):
        if difflib.SequenceMatcher(None, mp20_cif, train_cif).ratio() > threshold:
            return False
    return True

def worker(args):
    idx, mp20_cif, crystal_cifs, threshold = args
    result = is_unique(mp20_cif, crystal_cifs, threshold)
    return (idx, result)

def save_checkpoint(filtered_dict, path):
    with open(path, "wb") as f:
        pickle.dump(filtered_dict, f)

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

def main():
    parser = argparse.ArgumentParser(description="Filter mp-20 CIFs based on similarity to crystallm set with checkpointing.")
    parser.add_argument("--crystallm_path", type=str, required=True)
    parser.add_argument("--mp20_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Final output file (pickle) with filtered mp-20 CIFs.")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pkl", help="Path to checkpoint file.")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()

    # Load training CIFs
    print("Loading crystallm training CIFs...")
    with h5py.File(args.crystallm_path, "r") as h5_file:
        crystal_cifs = h5_file["cif_string"][:]

    # Load mp-20 CIFs
    print("Loading mp-20 CIFs...")
    with open(args.mp20_path, "rb") as f:
        mp20_data = pickle.load(f)
    mp20_cifs = list(mp20_data["cif"][:])

    # Load checkpoint if it exists
    print("Loading checkpoint...")
    checkpoint = load_checkpoint(args.checkpoint)
    already_done = set(checkpoint.keys())
    print(f"{len(already_done)} entries already processed.")

    # Prepare remaining work
    remaining_args = [
        (i, cif, crystal_cifs, args.threshold)
        for i, cif in enumerate(mp20_cifs)
        if i not in already_done
    ]

    print(f"Remaining entries to process: {len(remaining_args)}")

    batch_results = {}
    with Pool(processes=args.processes) as pool:
        for i, (idx, is_unique_result) in enumerate(tqdm(pool.imap(worker, remaining_args), total=len(remaining_args))):
            batch_results[idx] = is_unique_result

            # Save checkpoint every N results
            if (i + 1) % args.checkpoint_every == 0:
                checkpoint.update(batch_results)
                save_checkpoint(checkpoint, args.checkpoint)
                batch_results.clear()

    # Final checkpoint save
    if batch_results:
        checkpoint.update(batch_results)
        save_checkpoint(checkpoint, args.checkpoint)

    # Filter mp20 cif strings using final checkpoint map
    filtered_cifs = [mp20_cifs[i] for i, keep in sorted(checkpoint.items()) if keep]

    print(f"Final number of filtered mp-20 CIFs: {len(filtered_cifs)}")

    with open(args.output, "wb") as f:
        pickle.dump(filtered_cifs, f)
    print(f"Saved filtered mp-20 CIFs to: {args.output}")

if __name__ == '__main__':
    main()
