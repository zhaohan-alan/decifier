import sys
sys.path.append("./")
import os
import io
import gc
import pickle
import argparse
from pymatgen.io.cif import CifWriter, Structure, CifParser
from pymatgen.core import Composition

from typing import List

from collections import Counter

from dscribe.descriptors import SOAP, ACSF

try:
    parser_from_string = CifParser.from_str
except:
    parser_from_string = CifParser.from_string

from pymatgen.analysis.diffraction.xrd import XRDCalculator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, TimeoutError, Manager
import numpy as np
import time
import random

import h5py
import gzip
import json

import logging
from logging import handlers

from decifer import (
    Tokenizer,
    replace_symmetry_loop_with_P1,
    remove_cif_header,
    remove_oxidation_loop,
    format_occupancies,
    extract_formula_units,
    extract_formula_nonreduced,
    extract_space_group_symbol,
    replace_data_formula_with_nonreduced_formula,
    round_numbers,
    add_atomic_props_block,
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

def safe_pkl_gz(output, output_path):
    temp_path = output_path + '.tmp' # Ensuring that only fully written files are considered when collecting
    with gzip.open(temp_path, 'wb') as f:
        pickle.dump(output, f)
    os.rename(temp_path, output_path) # File is secure, renaming

def run_subtasks(
    root: str,
    worker_function,
    get_from: str,
    save_to: str,
    add_metadata: List[str] = [],
    task_kwargs_dict = {},
    announcement: str = None,
    debug: bool = False,
    debug_max: int = None,
    workers: int = cpu_count() - 1,
    from_gzip: bool = False,
):
    if announcement:
        print("-"*20)
        print(announcement)
        print("-"*20)

    # Locate pickles
    from_dir = os.path.join(root, get_from)
    pickles = glob(os.path.join(from_dir, '*.pkl.gz'))
    if not from_gzip:
        assert len(pickles) > 0, f"Cannot locate any files in {from_dir}"
        paths = pickles[:debug_max]
        assert len(paths) > 1, f"Flagging suspicious behaviour, only 1 or less CIFs present in {from_dir}: {paths}"
    else:
        assert len(pickles) == 1, f"from_gzip flag is raised, but more than one gzip file found in directory"
        with gzip.open(pickles[0], 'rb') as f:
            paths = pickle.load(f)[:debug_max]

    # Make output folder
    to_dir = os.path.join(root, save_to)
    os.makedirs(to_dir, exist_ok=True)

    # Open metadata if specified
    if add_metadata:
        metadata_path = os.path.join(root, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        for key in add_metadata:
            try:
                task_kwargs_dict[key] = metadata[key]
            except NameError:
                print(f"Could not locate metadata with key: {key} in {metadata_path}")
                pass
    
    # Check for exisiting files
    existing_files = set(os.path.basename(f) for f in glob(os.path.join(to_dir, "*.pkl.gz")))

    # Collect tasks
    tasks = []
    pbar = tqdm(desc="Collecting tasks...", total=len(paths), leave=False)
    for path in paths:
        if isinstance(path, str):
            name = os.path.basename(path)
        else:
            name, _ = path
        if name in existing_files:
            pbar.update(1)
            continue
        tasks.append(
            (path, task_kwargs_dict, debug, to_dir)
        )
        pbar.update(1)
    pbar.close()

    if not tasks:
        print("All task have already been executed for {save_to}, skipping...", end='\n')
    else:
        # Keep track of optional worker metadata
        worker_metadata = {}
        # Create a queue for logging and start the logging process
        log_queue = mp.Queue()
        listener = log_listener(log_queue, to_dir)

        # Parallel processing of CIF files using multiprocessing
        with Pool(processes=workers, initializer=init_worker, initargs=(log_queue,)) as pool:
            results_iterator = pool.imap_unordered(worker_function, tasks)
            
            for _ in tqdm(range(len(tasks)), total=len(tasks), desc="Executing tasks...", leave=False):
                try:
                    worker_metadata_dict = results_iterator.next(timeout = 60)
                    if worker_metadata_dict is not None:
                        for key, values in worker_metadata_dict.items():
                            if not key in worker_metadata:
                                worker_metadata[key] = []
                            # Extend if more than one entry else append
                            if len(values) > 1:
                                worker_metadata[key].extend(values)
                            else:
                                worker_metadata[key].append(values)
                except TimeoutError as e:
                    continue

        # Add worker metadata
        for key in worker_metadata.keys():
            worker_metadata[key] = Counter(worker_metadata[key])
        save_metadata(worker_metadata, root)

        # Stop log listener and flush
        listener.stop()
        logging.shutdown()

    n_files = len(paths)
    n_successful_files = len(glob(os.path.join(to_dir, '*.pkl.gz')))
    print(f"Reduction in dataset: {n_files} samples --> {n_successful_files} samples")
    
    additional_metadata = {save_to: task_kwargs_dict}
    save_metadata(additional_metadata, root)
    
    # Free up memory
    del tasks
    gc.collect()

def preprocess_worker(args):
    
    # Extract arguments
    obj, task_dict, debug, to_dir = args

    try:
        # Make structure
        if isinstance(obj, str): # Path
            structure = Structure.from_file(obj)
            cif_name = os.path.basename(obj)
        elif isinstance(obj, tuple): # cif string from pkl
            cif_name, cif_string = obj
            structure = parser_from_string(cif_string).get_structures()[0]
        else:
            raise Exception("Unexpected format found in preprocessing step of {obj}")
        
        # Option for removing structures with occupancies below 1
        if task_dict['remove_occ_less_than_one']:
            for site in structure:
                occ = list(site.species.as_dict().values())[0]
                if occ < 1:
                    raise Exception("Occupancy below 1.0 found")

        # Get stratification key
        strat_key = create_stratification_key(structure, task_dict['spacegroup_strat_group_size'])

        # Get raw content of CIF in string
        cif_string = CifWriter(struct=structure, symprec=0.1).__str__()

        # Extract formula units and remove if Z=0
        formula_units = extract_formula_units(cif_string)
        if formula_units == 0:
            raise Exception()

        # Remove oxidation state information
        cif_string = remove_oxidation_loop(cif_string)

        # Number precision rounding
        cif_string = round_numbers(cif_string, task_dict['decimal_places'])
        cif_string = format_occupancies(cif_string, task_dict['decimal_places'])

        # Add atomic props block
        cif_string = add_atomic_props_block(cif_string)

        # Extract species, spacegroup and composition of crystal structure
        composition = Composition(extract_formula_nonreduced(cif_string)).as_dict()
        species = set(composition.keys())
        spacegroup = extract_space_group_symbol(cif_string)

        # Save output to pickle file
        output_dict = {
            'cif_name': cif_name,
            'cif_string': cif_string,
            'strat_key': strat_key,
            'species': species,
            'spacegroup': spacegroup,
            'composition': composition,
        }
        output_path = os.path.join(to_dir, cif_name + '.pkl.gz')
        safe_pkl_gz(output_dict, output_path)

        return {'species': species, 'spacegroups': spacegroup}

    except Exception as e:
        if debug:
            print(f"Error processing {cif_name}: {e}")

        logger = logging.getLogger()
        logger.exception(f"Exception in worker function pre-processing CIF {cif_name}, with error:\n {e}\n\n")
        
        # Append the file name the failed_files manager list
        #failed_files.append(cif_name)

        return None

def descriptor_worker(args):

    # Extract arguments
    from_path, task_dict, debug, to_dir = args
    
    # Open pkl and extract
    with gzip.open(from_path, "rb") as f:
        data = pickle.load(f)
    cif_name = data['cif_name']
    cif_string = data['cif_string']
    
    # Get species
    species = list(task_dict['species'].keys())

    try:
        # Load structure and parse to ASE
        ase_structure = parser_from_string(cif_string).get_structures()[0].to_ase_atoms()

        # Setup SOAP object
        soap = SOAP(
            species = species,
            r_cut = task_dict['soap']['r_cut'],
            n_max = task_dict['soap']['n_max'],
            l_max = task_dict['soap']['l_max'],
            sigma = task_dict['soap']['sigma'],
            rbf = task_dict['soap']['rbf'],
            compression = {
                'mode': task_dict['soap']['compression_mode'],
                'weighting': None,
            },
            periodic = task_dict['soap']['periodic'],
            sparse = task_dict['soap']['sparse'],
        )

        # Setup ACSF
        acsf = ACSF(
            species = species,
            r_cut = task_dict['acsf']['r_cut'],
            periodic = task_dict['acsf']['periodic'],
            sparse = task_dict['acsf']['sparse'],
        )

        # Calculate descriptors and save to pkl.gz
        output_dict = {
            'soap': soap.create(ase_structure, centers=[0])[0],
            'acsf': acsf.create(ase_structure, centers=[0])[0],
        }
        output_path = os.path.join(to_dir, cif_name + '.pkl.gz')
        safe_pkl_gz(output_dict, output_path)

    except Exception as e:
        if debug:
            print(f"Error processing {cif_name}: {e}")
        logger = logging.getLogger()
        logger.exception(f"Exception in worker function calculating descriptors for CIF {cif_name}, with error:\n {e}\n\n")
    

def xrd_worker(args):

    # Extract arguments
    from_path, task_dict, debug, to_dir = args

    # Open pkl and extract
    with gzip.open(from_path, "rb") as f:
        data = pickle.load(f)
    cif_name = data['cif_name']
    cif_string = data['cif_string']

    try:
        # Load structure and parse to ASE
        structure = parser_from_string(cif_string).get_structures()[0]

        # Init calculator object and calculate XRD pattern
        xrd_calc = XRDCalculator(wavelength=task_dict['wavelength'])
        xrd_pattern = xrd_calc.get_pattern(structure)

        # Convert units to Q
        theta = np.radians(xrd_pattern.x / 2)
        q_disc = 4 * np.pi * np.sin(theta) / xrd_calc.wavelength # Q = 4 pi sin theta / lambda
        iq_disc = xrd_pattern.y / (np.max(xrd_pattern.y) + 1e-16)

        # Define Q grid
        q_cont = np.arange(task_dict['qmin'], task_dict['qmax'], task_dict['qstep'])

        # Init itensity array
        iq_cont = np.zeros_like(q_cont)

        # Apply Gaussian broadening
        for q_peak, iq_peak in zip(q_disc, xrd_pattern.y):
            gaussian_peak = iq_peak * np.exp(-0.5 * ((q_cont - q_peak) / task_dict['fwhm']) ** 2)
            iq_cont += gaussian_peak

        # Normalize intensities
        eps = 1e-16
        iq_cont /= (np.max(iq_cont) + eps)

        # Add noise based on SNR
        if task_dict['snr'] < 100.:
            noise = np.random.normal(0, np.max(iq_cont) / task_dict['snr'], size=iq_cont.shape)
            iq_cont = iq_cont + noise

        # Strictly positive singals (counts)
        iq_cont[iq_cont < 0] = 0

        # Save output to pkl.gz file
        output_dict = {
            'cif_name': cif_name,
            'xrd_disc': {
                'q': q_disc,
                'iq': iq_disc,
            },
            'xrd_cont': {
                'q': q_cont,
                'iq': iq_cont,
            },
        }
        
        output_path = os.path.join(to_dir, cif_name + '.pkl.gz')
        safe_pkl_gz(output_dict, output_path)

    except Exception as e:
        if debug:
            print(f"Error processing {cif_name}: {e}")
        logger = logging.getLogger()
        logger.exception(f"Exception in worker function with xrd calculation for CIF with name {cif_name}, with error:\n {e}\n\n")

def cif_tokenizer_worker(args):
    
    # Extract arguments
    from_path, task_dict, debug, to_dir = args

    # Open pkl and extract
    with gzip.open(from_path, "rb") as f:
        data = pickle.load(f)
    cif_name = data['cif_name']
    cif_string = data['cif_string']

    try:
        # Remove symmetries and header from cif_string before tokenizing
        cif_string = remove_cif_header(cif_string)
        cif_string_reduced = replace_data_formula_with_nonreduced_formula(cif_string)
        cif_string_nosym = replace_symmetry_loop_with_P1(cif_string_reduced)

        # Initialise Tokenizer
        tokenizer = Tokenizer()
        tokenize = tokenizer.tokenize_cif
        encode = tokenizer.encode

        cif_tokenized = encode(tokenize(cif_string_nosym))
    
        # Save output to pickle file
        output_dict = {
            'cif_tokenized': cif_tokenized,
        }
        output_path = os.path.join(to_dir, cif_name + '.pkl.gz')
        safe_pkl_gz(output_dict, output_path)

    except Exception as e:
        if debug:
            print(f"Error processing {cif_name}: {e}")
        logger = logging.getLogger()
        logger.exception(f"Exception in worker function with tokenization for CIF with name {cif_name}, with error:\n {e}\n\n")

def xrd_tokenizer_worker(args):
    
    # Extract arguments
    from_path, task_dict, debug, to_dir = args

    # Open pkl and extract
    with gzip.open(from_path, "rb") as f:
        data = pickle.load(f)
    cif_name = data['cif_name']
    xrd_disc = data['xrd_disc']

    try:
        # Convert discrete xrd to string
        xrd_disc_str = "\n".join([f"{x:5.4f}, {y:5.4f}" for (x,y) in zip(xrd_disc['q'], xrd_disc['iq'])])

        # Initialise Tokenizer
        tokenizer = Tokenizer()
        tokenize = tokenizer.tokenize_cif
        encode = tokenizer.encode

        xrd_tokenized = encode(tokenize(xrd_disc_str))
    
        # Save output to pickle file
        output_dict = {
            'xrd_tokenized': xrd_tokenized,
        }
        output_path = os.path.join(to_dir, cif_name + '.pkl.gz')
        safe_pkl_gz(output_dict, output_path)

    except Exception as e:
        if debug:
            print(f"Error processing {cif_name}: {e}")
        logger = logging.getLogger()
        logger.exception(f"Exception in worker function with tokenization for CIF with name {cif_name}, with error:\n {e}\n\n")

def serialize(data_dir):
    '''
    Combines the dataset files to HDF5 file format.
    
    Args:
        data_dir (str): Directory containing the dataset files.
    
    Returns:
        None
    '''
    # Create data splits
    #train_size = int(0.8 * len(cif_contents))
    #val_size = int(0.1 * len(cif_contents))
    #test_size = len(cif_contents) - train_size - val_size
    #print("Train size:", train_size)
    #print("Val size:", val_size)
    #print("Test size:", test_size)

    # Split data using stratification
    #train_data, test_data, train_names, test_names = train_test_split(
    #    cif_contents, names, test_size=test_size, stratify=strat_keys, random_state=seed
    #)
    #train_data, val_data, train_names, val_names = train_test_split(
    #    train_data, train_names, test_size=val_size, stratify=[strat_keys[cif_contents.index(f)] for f in train_data], random_state=seed
    #)

    #train = [(n, d) for (n, d) in zip(train_names, train_data)]
    #val = [(n, d) for (n, d) in zip(val_names, val_data)]
    #test = [(n, d) for (n, d) in zip(test_names, test_data)]

    # Save the train/val/test data splits
    #for data, prefix in zip([train, val, test], ["train", "val", "test"]):
    #    print(f"Saving {prefix} dataset...", end="")
    #    with gzip.open(os.path.join(pre_dir, f"{prefix}_dataset.pkl.gz"), "wb") as pkl:
    #        pickle.dump(data, pkl)
    #    del data
    #    print("DONE.")

    # Centralized metadata for preprocessing
    #metadata = {
        #"train_size": train_size,
        #"val_size": val_size,
        #"test_size": test_size,

    #pre_dir = os.path.join()
    #dataset_size = len(os.listdir()

    #try:
    
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
        data_dict = {"name": [], "xrd_discrete_x": [], "xrd_discrete_y": [], "xrd_cont_x": [], "xrd_cont_y": [], "xrd_tokenized": [], "cif_content": [], "cif_tokenized": [],
                     "soap": [], "acsf": []}
        for (name, xrd_discrete, xrd_cont, xrd_tokenized, cif_content, cif_tokenized, soap, acsf) in data:
            data_dict['name'].append(name)
            data_dict['xrd_discrete_x'].append(xrd_discrete[0])
            data_dict['xrd_discrete_y'].append(xrd_discrete[1])
            data_dict['xrd_cont_x'].append(xrd_cont[0])
            data_dict['xrd_cont_y'].append(xrd_cont[1])
            data_dict['xrd_tokenized'].append(np.array(xrd_tokenized))
            data_dict['cif_content'].append(cif_content)
            data_dict['cif_tokenized'].append(np.array(cif_tokenized))
            data_dict['soap'].append(soap)
            data_dict['acsf'].append(acsf)

        with h5py.File(os.path.join(hdf5_dir, f"{dataset_name}_dataset.h5"), 'w') as hdf5_file:
            hdf5_file.create_dataset('name', data=data_dict['name'])
            hdf5_file.create_dataset('xrd_discrete_x', data=data_dict['xrd_discrete_x'], dtype=h5py.special_dtype(vlen=np.dtype('float32')))
            hdf5_file.create_dataset('xrd_discrete_y', data=data_dict['xrd_discrete_y'], dtype=h5py.special_dtype(vlen=np.dtype('float32')))
            hdf5_file.create_dataset('xrd_cont_x', data=data_dict['xrd_cont_x'])#, dtype=h5py.special_dtype(vlen=np.dtype('float32')))
            hdf5_file.create_dataset('xrd_cont_y', data=data_dict['xrd_cont_y'])#, dtype=h5py.special_dtype(vlen=np.dtype('float32')))
            hdf5_file.create_dataset('xrd_tokenized', data=data_dict['xrd_tokenized'], dtype=h5py.special_dtype(vlen=np.dtype('int32')))
            hdf5_file.create_dataset('cif_content', data=data_dict['cif_content'])
            hdf5_file.create_dataset('cif_tokenized', data=data_dict['cif_tokenized'], dtype=h5py.special_dtype(vlen=np.dtype('int32')))
            hdf5_file.create_dataset('soap', data=data_dict['soap'])#, dtype=np.dtype('float16'))
            hdf5_file.create_dataset('acsf', data=data_dict['acsf'])#, dtype=np.dtype('float16'))

    print(f"Successfully created HDF5 files at {hdf5_dir}.")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare custom CIF files and save to a tar.gz file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, help="Path to the outer data directory", required=True)
    parser.add_argument("--name", "-n", type=str, help="Name of data preparation", required=True)
    parser.add_argument("--group-size", type=int, help="Spacegroup group size for stratification", default=10)
    parser.add_argument("--decimal-places", type=int, help="Number of decimal places for floats in CIF files", default=4)
    parser.add_argument("--include-occ", help="Include structures with occupancies less than one", action="store_false")

    parser.add_argument("--preprocess", help="preprocess files", action="store_true")
    parser.add_argument("--desc", help="calculate descriptors", action="store_true")  # Placeholder for future implementation
    parser.add_argument("--xrd", help="calculate XRD", action="store_true")  # Placeholder for future implementation
    parser.add_argument("--tokenize_cif", help="tokenize CIFs", action="store_true")  # Placeholder for future implementation
    parser.add_argument("--tokenize_xrd", help="tokenize XRDs", action="store_true")  # Placeholder for future implementation
    parser.add_argument("--serialize", help="serialize data by hdf5 convertion", action="store_true")  # Placeholder for future implementation

    parser.add_argument("--debug-max", help="Debug-feature: max number of files to process", type=int, default=0)
    parser.add_argument("--debug", help="Debug-feature: whether to print debug messages", action="store_true")
    parser.add_argument("--workers", help="Number of workers for each processing step", type=int, default=0)
    parser.add_argument("--raw_from_gzip", help="Whether raw CIFs come packages in gzip pkl", action="store_true")

    args = parser.parse_args()

    # Remove occ
    args.remove_occ = not args.include_occ

    # workers
    if args.workers == 0:
        args.workers = cpu_count() - 1
    else:
        args.workers = min(cpu_count() - 1, args.workers)

    # Make data prep directory and update data_dir
    args.data_dir = os.path.join(args.data_dir, args.name)
    os.makedirs(args.data_dir, exist_ok=True)

    # Adjust debug_max if no limit is specified
    if args.debug_max == 0:
        args.debug_max = None

    if args.preprocess:
        #cif, spacegroup_group_size, decimal_places, remove_occ_less_than_one, debug, pre_dir, failed_files = args
        preprocess_dict = {
            'spacegroup_strat_group_size': args.group_size,
            'decimal_places': args.decimal_places,
            'remove_occ_less_than_one': args.remove_occ,
        }
        run_subtasks(
            root = args.data_dir, 
            worker_function = preprocess_worker,
            get_from = "../",
            save_to = "preprocessed",
            task_kwargs_dict = preprocess_dict,
            announcement = "PREPROCESSING",
            debug = True,
            workers = args.workers,
            from_gzip = args.raw_from_gzip,
            debug_max = args.debug_max,
        )
        #preprocess(args.data_dir, args.seed, args.group_size, args.decimal_places, args.remove_occ, args.debug_max, args.debug, args.workers)
    
    if args.desc:
        descriptor_dict = {
            'soap': {
                'r_cut': 6.0,
                'n_max': 2,
                'l_max': 5,
                'sigma': 1.0,
                'rbf': 'gto',
                'compression_mode': 'crossover',
                'periodic': True,
                'sparse': False,
            },
            'acsf': {
                'r_cut': 6.0,
                'periodic': True,
                'sparse': False,
            },
        }
        run_subtasks(
            root = args.data_dir, 
            worker_function = descriptor_worker,
            get_from = "preprocessed",
            save_to = "descriptors",
            add_metadata = ["species"],
            task_kwargs_dict = descriptor_dict,
            announcement = "DESCRIPTOR CALCULATIONS",
            debug = True,
            workers = args.workers,
        )
        #generate_descriptors(args.data_dir, debug_max=args.debug_max, debug=args.debug, workers=args.workers)

    if args.xrd:
        xrd_dict = {
            'wavelength': 'CuKa',
            'qmin': 0.0,
            'qmax': 10.0,
            'qstep': 0.01,
            'fwhm': 0.05,
            'snr': 100.0,
        }
        run_subtasks(
            root = args.data_dir, 
            worker_function = xrd_worker,
            get_from = "preprocessed",
            save_to = "xrd",
            task_kwargs_dict = xrd_dict,
            announcement = "XRD CALCULATIONS",
            debug = True,
            workers = args.workers,
        )
        #generate_xrd(args.data_dir, debug_max=args.debug_max, debug=args.debug, workers=args.workers)

    if args.tokenize_cif:
        run_subtasks(
            root = args.data_dir, 
            worker_function = cif_tokenizer_worker,
            get_from = "preprocessed",
            save_to = "cif_tokenized",
            announcement = "TOKENIZING CIFs",
            debug = True,
            workers = args.workers,
        )

    if args.tokenize_xrd:
        run_subtasks(
            root = args.data_dir, 
            worker_function = xrd_tokenizer_worker,
            get_from = "xrd",
            save_to = "xrd_tokenized",
            announcement = "TOKENIZING XRDs",
            debug = True,
            workers = args.workers,
        )
        #tokenize_datasets(args.data_dir, debug_max=args.debug_max, debug=args.debug, workers=args.workers)
    
    if args.serialize:
        serialize(args.data_dir)

    # Store all arguments passed to the main function in centralized metadata
    metadata = {
        "arguments": vars(args)
    }

    # Finalize metadata saving after all processing steps
    save_metadata(metadata, args.data_dir)
