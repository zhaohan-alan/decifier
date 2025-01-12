#!/usr/bin/env python3
import os
import json
import base64
import argparse
import h5py
import numpy as np

###############################################################################
# Define a custom JSON encoder that can handle:
# - NumPy arrays   -> Convert to Python lists
# - bytes objects  -> Convert to base64 strings
###############################################################################
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Convert NumPy array to a nested list
            return obj.tolist()
        elif isinstance(obj, bytes):
            # Convert raw bytes to a Base64-encoded ASCII string
            return base64.b64encode(obj).decode('ascii')
        return super().default(obj)

def combine_h5_files(h5_files, output_h5):
    """
    Combine multiple HDF5 files into a single HDF5 file by streaming chunks 
    instead of loading everything in memory at once. Datasets with object dtype 
    (dtype('O')) are stored as variable-length strings. For each element that 
    is a list, np.ndarray, or bytes, we JSON-serialize it using NumpyEncoder.
    """
    if not h5_files:
        print(f"No files were found to combine for {output_h5}. Skipping.")
        return

    combined_data = {}
    total_files = len(h5_files)

    # Step 1: Read each HDF5 file in memory
    for file_idx, h5_file in enumerate(h5_files):
        print(f"\nProcessing file {file_idx + 1} of {total_files}: {h5_file}")
        print("  Reading data...")
        with h5py.File(h5_file, 'r') as f_in:
            # If first file, initialize combined_data keys
            if file_idx == 0:
                for key in f_in.keys():
                    combined_data[key] = []
            # Append arrays
            for key in f_in.keys():
                arr = f_in[key][()]  # loads entire dataset
                combined_data[key].append(arr)
        print(f"  Finished reading {h5_file}.")

    # Step 2: Create/overwrite output HDF5
    print(f"\nCombining data into {output_h5} ...")
    with h5py.File(output_h5, 'w') as f_out:
        for key, arrays in combined_data.items():
            print(f"  Combining dataset '{key}'...")

            # (A) Compute final shape
            total_length = sum(arr.shape[0] for arr in arrays)
            rest_shape = arrays[0].shape[1:]
            is_object_dtype = any(arr.dtype == np.dtype('O') for arr in arrays)

            # (B) Create output dataset
            if is_object_dtype:
                print(f"    Detected object dtype for '{key}'. Will store as variable-length strings.")
                string_dtype = h5py.vlen_dtype(str)
                out_dset = f_out.create_dataset(
                    key,
                    shape=(total_length,) + rest_shape,
                    maxshape=(None,) + rest_shape,
                    dtype=string_dtype,
                    chunks=True
                )
            else:
                out_dtype = arrays[0].dtype
                out_dset = f_out.create_dataset(
                    key,
                    shape=(total_length,) + rest_shape,
                    maxshape=(None,) + rest_shape,
                    dtype=out_dtype,
                    chunks=True
                )

            # (C) Write data in chunks
            current_index = 0
            for arr in arrays:
                arr_len = arr.shape[0]
                chunk_size = 10_000  # adjust as needed for memory usage
                for start_idx in range(0, arr_len, chunk_size):
                    end_idx = min(start_idx + chunk_size, arr_len)
                    chunk_data = arr[start_idx:end_idx]

                    # If object dtype, JSON-serialize each element
                    if is_object_dtype:
                        converted_list = []
                        for item in chunk_data:
                            # Use the custom NumpyEncoder to handle arrays + bytes
                            converted_list.append(json.dumps(item, cls=NumpyEncoder))
                        # Convert to a NumPy array of object (string)
                        chunk_data = np.array(converted_list, dtype=object)

                    # Write chunk to output
                    out_dset[current_index:current_index + (end_idx - start_idx)] = chunk_data
                    current_index += (end_idx - start_idx)

    print(f"Finished creating {output_h5}.\n")

def main():
    parser = argparse.ArgumentParser(description="Combine train/val/test HDF5 files across multiple datasets (chunk-based).")
    parser.add_argument("--input-dirs", nargs='+', required=True,
                        help="List of directories that each contain train.h5, val.h5, and test.h5.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to store the combined train.h5, val.h5, and test.h5.")
    parser.add_argument("--splits", nargs='+', required=True)
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # For each split, gather the existing HDF5 files from each input directory
    for split in args.splits:
        h5_files = []
        print(f"\nLooking for {split}.h5 in provided directories...")
        for in_dir in args.input_dirs:
            candidate = os.path.join(in_dir, f"{split}.h5")
            if os.path.isfile(candidate):
                print(f"  Found {candidate}")
                h5_files.append(candidate)
            else:
                print(f"  Warning: Did not find {candidate}")

        output_h5 = os.path.join(args.output_dir, f"{split}.h5")
        combine_h5_files(h5_files, output_h5)

    print("Done combining HDF5 files across the specified directories.")

if __name__ == "__main__":
    main()
