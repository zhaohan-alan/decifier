import sys
sys.path.append("./")
import os
import io
import pickle
import argparse
from pymatgen.io.cif import CifWriter, CifParser, Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import numpy as np

import h5py
import gzip
import json

import logging
from logging import handlers

from decifer import (
    Tokenizer,
    replace_symmetry_loop,
    remove_cif_header,
    remove_oxidation_loop,
    format_occupancies,
    extract_formula_units,
    replace_data_formula_with_nonreduced_formula,
    round_numbers,
)

import warnings
warnings.filterwarnings("ignore")

def init_worker(log_queue):
    """
    Initializes each worker process with a queue-based logger.
    This will be called when each worker starts.
    """
    queue_handler = handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.addHandler(queue_handler)
    logger.setLevel(logging.ERROR)

def log_listener(queue, log_dir):
    """
    Function excecuted by the log listener process.
    Receives messages from the queue and writes them to the log file.
    Rotates the log file based on size.
    """
    # Setup file handler (rotating)
    log_file = os.path.join("./" + log_dir, "error.log")
    handler = handlers.RotatingFileHandler(
        log_file, maxBytes=1024*1024, backupCount=5,
    )
    handler.setLevel(logging.ERROR)

    # Add a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Set up the log listener
    listener = handlers.QueueListener(queue, handler)
    listener.start()

    # return listener
    return listener

def save_metadata(metadata, data_dir):
    """
    Save metadata information (sizes, shapes, argument parameters, vocab size, etc.) into a centralized JSON file.

    Args:
        metadata (dict): Dictionary containing metadata information.
        data_dir (str): Directory where the metadata file should be saved.
    """
    metadata_file = os.path.join(data_dir, "metadata.json")
    
    # If the metadata file already exists, load the existing metadata and update it
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            existing_metadata = json.load(f)
        # Update existing metadata with new data
        existing_metadata.update(metadata)
        metadata = existing_metadata
    
    # Save updated metadata to the JSON file
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
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
            decimal_places (int): Number of decimal places to be enforced in floats in CIF file.
            remove_occ_less_than_one (bool): If True, structures with occupancy less than 1.0 will be removed.
            debug (bool): If True, print errors for failed CIF processing.

    Returns:
        tuple: (strat_key, cif_content) if processing is successful, else None.
    """
    path, spacegroup_group_size, decimal_places, remove_occ_less_than_one, debug = args
    logger = logging.getLogger()
    name = os.path.basename(path)
    try:
        # Make structure
        struct = Structure.from_file(path)
        
        # Option for removing structures with occupancies below 1
        if remove_occ_less_than_one:
            for site in structure:
                if any(occ < 1 for occ in site.species_and_occu.values()):
                    raise Exception()

        # Get stratification key
        strat_key = create_stratification_key(struct, spacegroup_group_size)

        # Get raw content of CIF in string
        cif_content = CifWriter(struct=struct, symprec=0.1).__str__()

        # Extract formula units and remove if Z=0
        formula_units = extract_formula_units(cif_content)
        if formula_units == 0:
            raise Exception()

        # Remove oxidation state information
        cif_content = remove_oxidation_loop(cif_content)

        # Number precision rounding
        cif_content = round_numbers(cif_content, decimal_places)
        cif_content = format_occupancies(cif_content, decimal_places)

        return (name, strat_key, cif_content)
    except Exception as e:
        if debug:
            #raise e
            print(f"Error processing {path}: {e}")

        logger.exception(f"Exception in worker function pre-processing CIF with path {path}, with error:\n {e}\n\n")

        return None

def preprocess(data_dir, seed, spacegroup_group_size, decimal_places=4, remove_occ_less_than_one=False, debug_max=None, debug=False):
    """
    Preprocess CIF files by extracting their structure and creating a train/val/test split.

    Args:
        data_dir (str): Directory containing the raw CIF files.
        seed (int): Random seed for reproducibility.
        spacegroup_group_size (int): Group size for spacegroup stratification.
        decimal_places (int): Number of decimal places to be enforced in floats in CIF file.
        remove_occ_less_than_one (bool): If True, structures with occupancy less than 1.0 will be removed.
        debug_max (int, optional): Maximum number of CIFs to process (for debugging).
        debug (bool, optional): If True, print debugging information.

    Returns:
        None
    """

    # Find raw CIF files
    raw_dir = os.path.join("/".join(data_dir.split("/")[:-1]), "raw")
    cif_paths = sorted(glob(os.path.join(raw_dir, "*.cif")))[:debug_max]
    assert len(cif_paths) > 0, f"Cannot find any .cif files in {raw_dir}"

    # Output directory
    pre_dir = os.path.join(data_dir, "preprocessed")
    os.makedirs(pre_dir, exist_ok=True)

    # Prepare arguments for parallel processing
    tasks = [(path, spacegroup_group_size, decimal_places, remove_occ_less_than_one, debug) for path in cif_paths]
    
    # Create a queue for logging and start the logging process
    log_queue = mp.Queue()
    listener = log_listener(log_queue, pre_dir)

    # Parallel processing of CIF files using multiprocessing
    num_workers = min(cpu_count(), len(cif_paths))  # Use available CPU cores, limited to number of files
    with Pool(processes=num_workers, initializer=init_worker, initargs=(log_queue,)) as pool:
        results = list(tqdm(pool.imap_unordered(
            process_single_cif, tasks
        ), total=len(cif_paths), desc="Preprocessing CIFs...", leave=False))
        
    # Stop log listener and flush
    listener.stop()
    logging.shutdown()

    # Filter out None results from failed processing
    valid_results = [result for result in results if result is not None]
    names, strat_keys, cif_contents = zip(*valid_results)

    print("-"*20)
    print("PRE-PROCESSING")
    print("-"*20)
    print(f"Reduction in dataset: {len(cif_paths)} samples --> {len(cif_contents)} samples")

    # Create data splits
    train_size = int(0.8 * len(cif_contents))
    val_size = int(0.1 * len(cif_contents))
    test_size = len(cif_contents) - train_size - val_size
    print("Train size:", train_size)
    print("Val size:", val_size)
    print("Test size:", test_size)

    # Split data using stratification
    train_data, test_data, train_names, test_names = train_test_split(
        cif_contents, names, test_size=test_size, stratify=strat_keys, random_state=seed
    )
    train_data, val_data, train_names, val_names = train_test_split(
        train_data, train_names, test_size=val_size, stratify=[strat_keys[cif_contents.index(f)] for f in train_data], random_state=seed
    )

    train = [(n, d) for (n, d) in zip(train_names, train_data)]
    val = [(n, d) for (n, d) in zip(val_names, val_data)]
    test = [(n, d) for (n, d) in zip(test_names, test_data)]

    # Save the train/val/test data splits
    for data, prefix in zip([train, val, test], ["train", "val", "test"]):
        print(f"Saving {prefix} dataset...", end="")
        with gzip.open(os.path.join(pre_dir, f"{prefix}_dataset.pkl.gz"), "wb") as pkl:
            pickle.dump(data, pkl)
        print("DONE.")

    # Centralized metadata for preprocessing
    metadata = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "total_files_processed": len(cif_paths),
        "total_valid_files": len(cif_contents),
        "decimal_places": decimal_places,
        "remove_occ_less_than_one": remove_occ_less_than_one,
        "seed": seed,
        "spacegroup_group_size": spacegroup_group_size
    }
    save_metadata(metadata, data_dir)
        
    print(f"Prepared CIF files have been saved to {pre_dir}")
    print()

def generate_single_xrd(args):

    # Extract arguments
    name, cif_content, wavelength, qmin, qmax, qstep, fwhm, snr, debug = args
    logger = logging.getLogger()
    
    # Generate structure from cif_content
    try:
        struct = CifParser.from_str(cif_content).get_structures()[0]
    except AttributeError:
        struct = CifParser.from_string(cif_content).get_structures()[0]
    except Exception as e:
        if debug:
            print(f"Error processing {name}: {e}")
        logger.exception(f"Exception in worker function with CifParser for CIF with name {name}, with error:\n {e}\n\n")
        return None

    try:
        # Init calculator object
        xrd_calc = XRDCalculator(wavelength=wavelength)

        # Get XRD pattern
        xrd_pattern = xrd_calc.get_pattern(struct)

    except Exception as e:
        if debug:
            print(f"Error processing {name}: {e}")
        logger.exception(f"Exception in worker function with xrd calculation for CIF with name {name}, with error:\n {e}\n\n")
        return None

    # Convert to Q
    theta = np.radians(xrd_pattern.x / 2)
    q_discrete = 4 * np.pi * np.sin(theta) / xrd_calc.wavelength # Q = 4 pi sin theta / lambda
    i_discrete = xrd_pattern.y / (np.max(xrd_pattern.y) + 1e-16)

    # Define Q grid
    q_cont = np.arange(qmin, qmax, qstep)

    # Init itensity array
    i_cont = np.zeros_like(q_cont)

    # Apply Gaussian broadening
    for q_peak, intensity in zip(q_discrete, xrd_pattern.y):
        gaussian_peak = intensity * np.exp(-0.5 * ((q_cont - q_peak) / fwhm) ** 2)
        i_cont += gaussian_peak

    # Normalize intensities
    i_cont /= (np.max(i_cont) + 1e-16)

    # Add noise based on SNR
    noise = np.random.normal(0, np.max(i_cont) / snr, size=i_cont.shape)
    i_cont = i_cont + noise
    i_cont[i_cont < 0] = 0 # Strictly positive signal (counts)

    return name, np.vstack([q_discrete, i_discrete]), np.vstack([q_cont, i_cont]), cif_content


def generate_xrd(data_dir, wavelength='CuKa', qmin=0., qmax=10., qstep=0.01, fwhm=0.05, snr=80., debug_max=None, debug=False):
    """
    Calculate the XRD pattern from a CIF string using pytmatgen with Gaussian peak broadening and noise.

    Args:
        data_dir (str): Directory containing the raw CIF files.
        wavelength (str): X-ray wavelength (default is 'CuKa').
        qmin (float): Minimum Q value (default is 0).
        qmax (float): Maximum Q value (default is 20).
        qstep (float): Step size for Q values (default is 0.01).
        fwhm (float): Full-width at half-maximum for Gaussian broadening (default is 0.1).
        snr (float): Signal-to-noise ratio for adding noise the the pattern (default is 20).
        debug_max (int, optional): Maximum number of CIFs to process (for debugging).
        debug (bool, optional): If True, print debugging information.

    Returns:
        None
    """
    
    # Find train / val / test
    preprocessed_dir = os.path.join(data_dir, "preprocessed")
    datasets = glob(os.path.join(preprocessed_dir, '*.pkl.gz'))
    assert len(datasets) > 0, f"Cannot find any preprocessed files in {preprocessed_dir}"

    # Make sure that there is an output
    xrd_dir = os.path.join(data_dir, "xrd")
    os.makedirs(xrd_dir, exist_ok=True)
        
    print("-"*20)
    print("XRD CALCULATION")
    print("-"*20)

    for dataset_path in datasets:
        dataset_name = dataset_path.split("/")[-1].split(".")[0].split("_")[0]

        # Open pkl
        with gzip.open(dataset_path, "rb") as f:
            data = pickle.load(f)

        ## Debug max
        data = data[:debug_max]

        # Prepare arguments for parallel processing
        tasks = [(name, cif_content, wavelength, qmin, qmax, qstep, fwhm, snr, debug) for (name, cif_content) in data]
    
        # Create a queue for logging and start the logging process
        log_queue = mp.Queue()
        listener = log_listener(log_queue, xrd_dir)

        # Parallel processing of CIF files using multiprocessing
        num_workers = min(cpu_count(), len(data))  # Use available CPU cores, limited to number of files
        with Pool(processes=num_workers, initializer=init_worker, initargs=(log_queue,)) as pool:
            results = list(tqdm(pool.imap_unordered(
                generate_single_xrd, tasks
            ), total=len(data), desc="Calculating XRD...", leave=False))

        # Stop log listener and flush
        listener.stop()
        logging.shutdown()
    
        # Filter out None results from failed processing
        valid_results = [result for result in results if result is not None]
        print(f"Reduction in {dataset_name} dataset: {len(results)} --> {len(valid_results)}")

        # Save
        print(f"Saving {dataset_name} dataset with XRD data...", end="")
        output_path = os.path.join(xrd_dir, f"{dataset_name}_dataset_xrd.pkl.gz")
        with gzip.open(output_path, "wb") as pkl:
            pickle.dump(valid_results, pkl)
        print("DONE.")

    # Initialize metadata for XRD calculation
    metadata = {
        "xrd_parameters": {
            "wavelength": wavelength,
            "qmin": qmin,
            "qmax": qmax,
            "qstep": qstep,
            "qgrid_len": int((qmax - qmin) / qstep),
            "fwhm": fwhm,
            "snr": snr
        },
    }
    # Save XRD metadata to centralized file
    save_metadata(metadata, data_dir)
        
    print(f"Generated XRD have been saved to {xrd_dir}")
    print()

def tokenize_single_datum(args):
    
    # Extract arguments
    name, xrd_discrete, xrd_cont, cif_content, debug = args
    logger = logging.getLogger()

    # Convert discrete xrd to string
    xrd_discrete_str = "\n".join([f"{x:5.4f}, {y:5.4f}" for (x,y) in zip(*xrd_discrete)])
        
    # Remove symmetries and header from cif_content before tokenizing
    cif_content_noheader = remove_cif_header(cif_content)
    cif_content_nosym = replace_symmetry_loop(cif_content_noheader)

    # Initialise Tokenizer
    tokenizer = Tokenizer()
    tokenize = tokenizer.tokenize_cif
    encode = tokenizer.encode

    try:
        cif_tokenized = encode(tokenize(cif_content_nosym))
        xrd_tokenized = encode(tokenize(xrd_discrete_str))
    except Exception as e:
        if debug:
            print(f"Error processing {name}: {e}")
        logger.exception(f"Exception in worker function with tokenization for CIF with name {name}, with error:\n {e}\n\n")
        return None

    return name, xrd_discrete, xrd_cont, xrd_tokenized, cif_content_noheader, cif_tokenized

def tokenize_datasets(data_dir, debug_max=None, debug=False):
    # Find train / val / test
    xrd_dir = os.path.join(data_dir, "xrd")
    datasets = glob(os.path.join(xrd_dir, '*.pkl.gz'))
    assert len(datasets) > 0, f"Cannot find any files with xrd data in {xrd_dir}"

    # Make sure that there is an output
    tokenized_dir = os.path.join(data_dir, "tokenized")
    os.makedirs(tokenized_dir, exist_ok=True)
        
    print("-"*20)
    print("TOKENIZATION")
    print("-"*20)

    for dataset_path in datasets:
        dataset_name = dataset_path.split("/")[-1].split(".")[0].split("_")[0]

        # Open pkl
        with gzip.open(dataset_path, "rb") as f:
            data = pickle.load(f)

        # Debug max
        data = data[:debug_max]

        # Prepare arguments for parallel processing
        tasks = [(name, xrd_discrete, xrd_cont, cif_content, debug) for (name, xrd_discrete, xrd_cont, cif_content) in data]
        
        # Create a queue for logging and start the logging process
        log_queue = mp.Queue()
        listener = log_listener(log_queue, tokenized_dir)

        # Parallel processing of CIF files using multiprocessing
        num_workers = min(cpu_count(), len(data))  # Use available CPU cores, limited to number of files
        with Pool(processes=num_workers, initializer=init_worker, initargs=(log_queue,)) as pool:
            results = list(tqdm(pool.imap_unordered(
                tokenize_single_datum, tasks
            ), total=len(data), desc="Tokenizing CIF and XRD...", leave=False))
        
        # Stop log listener and flush
        listener.stop()
        logging.shutdown()
    
        # Filter out None results from failed processing
        valid_results = [result for result in results if result is not None]
        print(f"Reduction in {dataset_name} dataset: {len(results)} --> {len(valid_results)}")

        # Save
        print(f"Saving {dataset_name} tokenized data...", end="")
        output_path = os.path.join(tokenized_dir, f"{dataset_name}_dataset_xrd_tokenized.pkl.gz")
        with gzip.open(output_path, "wb") as pkl:
            pickle.dump(valid_results, pkl)
        print("DONE.")

    # Initialize metadata for tokenization
    metadata = {
        "vocab_size": Tokenizer().vocab_size
    }    

    # Save tokenization metadata
    save_metadata(metadata, data_dir)
        
    print(f"Tokenized data have been saved to {tokenized_dir}")
    print()

def serialize(data_dir):
    '''
    Converts the pkl.gz dataset files to HDF5 file format.
    
    Args:
        data_dir (str): Directory containing the raw CIF files.
    
    Returns:
        None
    '''
    
    # Find train / val / test
    tokenized_dir = os.path.join(data_dir, "tokenized")
    datasets = glob(os.path.join(tokenized_dir, '*.pkl.gz'))
    assert len(datasets) > 0, f"Cannot find any tokenized data in {tokenized_dir}"

    # Make sure that there is an output
    hdf5_dir = os.path.join(data_dir, "hdf5")
    os.makedirs(hdf5_dir, exist_ok=True)

    print("-"*20)
    print("SERIALIZATION")
    print("-"*20)
    for dataset_path in datasets:
        dataset_name = dataset_path.split("/")[-1].split(".")[0].split("_")[0]

        with gzip.open(dataset_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)

        # Convert to dictionary
        data_dict = {"name": [], "xrd_discrete_x": [], "xrd_discrete_y": [], "xrd_cont_x": [], "xrd_cont_y": [], "xrd_tokenized": [], "cif_content": [], "cif_tokenized": []}
        for (name, xrd_discrete, xrd_cont, xrd_tokenized, cif_content, cif_tokenized) in data:
            data_dict['name'].append(name)
            data_dict['xrd_discrete_x'].append(xrd_discrete[0])
            data_dict['xrd_discrete_y'].append(xrd_discrete[1])
            data_dict['xrd_cont_x'].append(xrd_cont[0])
            data_dict['xrd_cont_y'].append(xrd_cont[1])
            data_dict['xrd_tokenized'].append(np.array(xrd_tokenized))
            data_dict['cif_content'].append(cif_content)
            data_dict['cif_tokenized'].append(np.array(cif_tokenized))

        with h5py.File(os.path.join(hdf5_dir, f"{dataset_name}_dataset.h5"), 'w') as hdf5_file:
            hdf5_file.create_dataset('name', data=data_dict['name'])
            hdf5_file.create_dataset('xrd_discrete_x', data=data_dict['xrd_discrete_x'], dtype=h5py.special_dtype(vlen=np.dtype('float32')))
            hdf5_file.create_dataset('xrd_discrete_y', data=data_dict['xrd_discrete_y'], dtype=h5py.special_dtype(vlen=np.dtype('float32')))
            hdf5_file.create_dataset('xrd_cont_x', data=data_dict['xrd_cont_x'])#, dtype=h5py.special_dtype(vlen=np.dtype('float32')))
            hdf5_file.create_dataset('xrd_cont_y', data=data_dict['xrd_cont_y'])#, dtype=h5py.special_dtype(vlen=np.dtype('float32')))
            hdf5_file.create_dataset('xrd_tokenized', data=data_dict['xrd_tokenized'], dtype=h5py.special_dtype(vlen=np.dtype('int32')))
            hdf5_file.create_dataset('cif_content', data=data_dict['cif_content'])
            hdf5_file.create_dataset('cif_tokenized', data=data_dict['cif_tokenized'], dtype=h5py.special_dtype(vlen=np.dtype('int32')))

    print(f"Successfully created HDF5 files at {hdf5_dir}.")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare custom CIF files and save to a tar.gz file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, help="Path to the outer data directory", required=True)
    parser.add_argument("--name", "-n", type=str, help="Name of data preparation", required=True)
    parser.add_argument("--group_size", type=int, help="Spacegroup group size for stratification", default=10)
    parser.add_argument("--decimal_places", type=int, help="Number of decimal places for floats in CIF files", default=4)
    parser.add_argument("--remove_occ", help="Remove structures with occupancies less than one", action="store_true")

    parser.add_argument("--preprocess", help="preprocess files", action="store_true")
    parser.add_argument("--xrd", help="calculate XRD", action="store_true")  # Placeholder for future implementation
    parser.add_argument("--tokenize", help="tokenize CIFs and XRD patterns", action="store_true")  # Placeholder for future implementation
    parser.add_argument("--serialize", help="serialize data by hdf5 convertion", action="store_true")  # Placeholder for future implementation

    parser.add_argument("--debug_max", help="Debug-feature: max number of files to process", type=int, default=0)
    parser.add_argument("--debug", help="Debug-feature: whether to print debug messages", action="store_true")

    args = parser.parse_args()

    # Make data prep directory and update data_dir
    args.data_dir = os.path.join(args.data_dir, args.name)
    os.makedirs(args.data_dir, exist_ok=True)

    # Adjust debug_max if no limit is specified
    if args.debug_max == 0:
        args.debug_max = None

    if args.preprocess:
        preprocess(args.data_dir, args.seed, args.group_size, args.decimal_places, args.remove_occ, args.debug_max, args.debug)

    if args.xrd:
        generate_xrd(args.data_dir, debug_max=args.debug_max, debug=args.debug)

    if args.tokenize:
        tokenize_datasets(args.data_dir, debug_max=args.debug_max, debug=args.debug)
    
    if args.serialize:
        serialize(args.data_dir)

    # Store all arguments passed to the main function in centralized metadata
    metadata = {
        "arguments": vars(args)
    }

    # Finalize metadata saving after all processing steps
    save_metadata(metadata, args.data_dir)
