#!/usr/bin/env python3

import os
import sys
import argparse
import yaml
from omegaconf import OmegaConf
import torch
import multiprocessing as mp
from time import time
from queue import Empty
import h5py
from tqdm.auto import tqdm
import pandas as pd
from glob import glob
import pickle
import json

import numpy as np

from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.cif import CifWriter, Structure, CifParser
try:
    parser_from_string = CifParser.from_str
except:
    parser_from_string = CifParser.from_string

# descriptors
from dscribe.descriptors import SOAP, MBTR, ACSF

# Import custom modules
from decifer import (
    HDF5Dataset,
    DeciferDataset,
    load_model_from_checkpoint,
    Tokenizer,
    extract_prompt,
    replace_symmetry_loop_with_P1,
    extract_space_group_symbol,
    reinstate_symmetry_loop,
    is_sensible,
    evaluate_syntax_validity,
    extract_numeric_property,
    get_unit_cell_volume,
    extract_volume,
)

def evaluate_cif(cif):
    eval_dict = {
        'cif': cif,
        'syntax_validity': None,
        'spacegroup': None,
        'cell_params': {
            'a': None,
            'b': None,
            'c': None,
            'alpha': None,
            'beta': None,
            'gamma': None,
            'implied_vol': None,
            'gen_vol': None,
        },
    }
    try:
        eval_dict['syntax_validity'] = evaluate_syntax_validity(cif)
        eval_dict['spacegroup'] = extract_space_group_symbol(cif)
        
        a = extract_numeric_property(cif, "_cell_length_a")
        b = extract_numeric_property(cif, "_cell_length_b")
        c = extract_numeric_property(cif, "_cell_length_c")
        alpha = extract_numeric_property(cif, "_cell_angle_alpha")
        beta = extract_numeric_property(cif, "_cell_angle_beta")
        gamma = extract_numeric_property(cif, "_cell_angle_gamma")
        
        eval_dict['cell_params'].update({
            'a': a,
            'b': b,
            'c': c,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
        })
        eval_dict['cell_params']['implied_vol'] = get_unit_cell_volume(a, b, c, alpha, beta, gamma)
        eval_dict['cell_params']['gen_vol'] = extract_volume(cif)
        
    except Exception:
        return eval_dict
    return eval_dict

def calculate_xrd(name, cif, xrd_parameters, debug=False):
    try:
        struct = parser_from_string(cif).get_structures()[0]
    except Exception as e:
        if debug:
            print(f"Error processing {name}: {e}")
        return None

    try:
        # Init calculator object
        xrd_calc = XRDCalculator(wavelength=xrd_parameters['wavelength'])

        # Get XRD pattern
        xrd_pattern = xrd_calc.get_pattern(struct)

    except Exception as e:
        if debug:
            print(f"Error processing {name}: {e}")
        return None

    # Convert to Q
    theta = np.radians(xrd_pattern.x / 2)
    q_discrete = 4 * np.pi * np.sin(theta) / xrd_calc.wavelength # Q = 4 pi sin theta / lambda
    i_discrete = xrd_pattern.y / (np.max(xrd_pattern.y) + 1e-16)

    # Define Q grid
    q_cont = np.arange(xrd_parameters['qmin'], xrd_parameters['qmax'], xrd_parameters['qstep'])

    # Init itensity array
    i_cont = np.zeros_like(q_cont)

    # Apply Gaussian broadening
    for q_peak, intensity in zip(q_discrete, xrd_pattern.y):
        gaussian_peak = intensity * np.exp(-0.5 * ((q_cont - q_peak) / xrd_parameters['fwhm']) ** 2)
        i_cont += gaussian_peak

    # Normalize intensities
    i_cont /= (np.max(i_cont) + 1e-16)

    # Add noise based on SNR
    if xrd_parameters['snr'] < 100.:
        noise = np.random.normal(0, np.max(i_cont) / xrd_parameters['snr'], size=i_cont.shape)
        i_cont = i_cont + noise
    i_cont[i_cont < 0] = 0 # Strictly positive signal (counts)

    return {'q': q_cont, 'iq': i_cont}

def calculate_crystal_descriptors(name, cif, species, desc_parameters, debug=False):
        
    try:
        # Load structure and parse to ASE
        ase_structure = parser_from_string(cif).get_structures()[0].to_ase_atoms()

        # Setup SOAP object
        soap = SOAP(
            species = species,
            r_cut = desc_parameters['soap']['r_cut'],
            n_max = desc_parameters['soap']['n_max'],
            l_max = desc_parameters['soap']['l_max'],
            sigma = desc_parameters['soap']['sigma'],
            rbf = desc_parameters['soap']['rbf'],
            compression = {
                'mode': desc_parameters['soap']['compression_mode'],
                'weighting': None,
            },
            periodic = desc_parameters['soap']['periodic'],
            sparse = desc_parameters['soap']['sparse'],
        )

        # Setup ACSF
        acsf = ACSF(
            species = species,
            r_cut = desc_parameters['acsf']['r_cut'],
            periodic = desc_parameters['acsf']['periodic'],
        )

        # Calculate descriptors and return dict
        descriptor_dict = {}
        descriptor_dict['SOAP'] = soap.create(ase_structure, centers=[0])[0]
        descriptor_dict['ACSF'] = acsf.create(ase_structure, centers=[0])[0]
    
        return descriptor_dict

    except Exception as e:
        if debug:
            print(f"Error processing {name}: {e}")
        return None
    

def worker(input_queue, output_queue, eval_files_dir):

    # Create tokenizer
    decode = Tokenizer().decode

    while True:
        task = input_queue.get()
        if task is None:
            break
        token_ids = task['token_ids']
        name = task['name']
        idx = task['index']
        rep = task['rep']
        dataset_name = task.get('dataset', 'UnknownDataset')
        model_name = task.get('model', 'UnknownModel')
        xrd_parameters = task.get(
            'xrd_parameters', 
            {
                'wavelength': 'CuKa',
                'qmin': 0.0,
                'qmax': 20.0,
                'qstep': 0.01,
                'fwhm': 0.1,
                'snr': 100.0,
            }
        )
        xrd_from_sample = task['xrd_from_sample']
        soap_from_sample = task['soap_from_sample']
        acsf_from_sample = task['acsf_from_sample']
        desc_parameters = task.get(
            'descriptors_parameters',
            {
                'soap': {
                    "r_cut": 6.0,
                    "n_max": 2,
                    "l_max": 5,
                    "sigma": 1.0,
                    "rbf": 'gto',
                    "compression_mode": 'crossover',
                    "periodic": True,
                    "sparse": False,
                },
                'acsf': {
                    "r_cut": 6.0,
                    "periodic": True,
                    "sparse": False,
                },
                
            }
        )
        species = list(task['species'].keys())
        debug = task['debug']
        try:
            # Preprocessing steps
            cif = decode(token_ids)
            cif = replace_symmetry_loop_with_P1(cif)
            spacegroup_symbol = extract_space_group_symbol(cif)
            if spacegroup_symbol != "P 1":
                cif = reinstate_symmetry_loop(cif, spacegroup_symbol)
            print(cif)
            if is_sensible(cif):

                # Evaluate the CIF
                eval_result = evaluate_cif(cif)
            
                # Add index and rep
                eval_result['index'] = idx
                eval_result['rep'] = rep

                # Calculate CIF descriptors
                eval_result['descriptors'] = {}
                eval_result['descriptors']['xrd_gen'] = calculate_xrd(name, cif, xrd_parameters, debug)
                eval_result['descriptors']['xrd_sample'] = xrd_from_sample
                crystal_descriptors = calculate_crystal_descriptors(name, cif, species, desc_parameters, debug)
                eval_result['descriptors']['soap_gen'] = crystal_descriptors['SOAP']
                eval_result['descriptors']['soap_sample'] = soap_from_sample
                eval_result['descriptors']['acsf_gen'] = crystal_descriptors['ACSF']
                eval_result['descriptors']['acsf_sample'] = acsf_from_sample

                # Add 'Dataset' and 'Model' to eval_result
                eval_result['dataset_name'] = dataset_name
                eval_result['model_name'] = model_name
                eval_result['seq_len'] = len(token_ids)
                eval_result['status'] = 'success'

                # Save the results to pickle
                output_filename = os.path.join(eval_files_dir, f"{name}_{rep}.pkl")
                temp_filename = output_filename + '.tmp' # Ensuring that only fully written files are considered when collecting
                with open(temp_filename, 'wb') as f:
                    pickle.dump(eval_result, f)
                os.rename(temp_filename, output_filename) # File is secure, renaming
                output_queue.put(eval_result)
            else:
                output_queue.put({'status': 'fail', 'index': idx, 'rep': rep})
        except Exception as e:
            raise e
            output_queue.put({'status': 'error', 'error_msg': str(e), 'index': idx, 'rep': rep})

def process_dataset(h5_test_path, block_size, model, input_queue, output_queue, eval_files_dir, num_workers,
                    out_folder_path='./', debug_max=None, debug=False,
                    add_composition=False, add_spacegroup=False, max_new_tokens=1000,
                    dataset_name='DefaultDataset', model_name='DefaultModel',
                    num_reps=1,):
    evaluations = []
    invalid_cifs = 0
    error_cifs = 0
    start = time()
    
    # Check which files have already been processed
    existing_eval_files = set(os.path.basename(f) for f in glob(os.path.join(eval_files_dir, "*.pkl")))
    
    # Make dataset from test
    test_dataset = DeciferDataset(h5_test_path, ["cif_name", "cif_tokenized", "xrd_cont.q", "xrd_cont.iq", "soap", "acsf"])

    # Extract metadata
    metadata_path = os.path.join(os.path.dirname(os.path.dirname(h5_test_path)), "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Padding token
    padding_id = Tokenizer().padding_id

    num_samples = len(test_dataset) * num_reps
    num_gens = min(num_samples, debug_max * num_reps) if debug_max is not None else num_samples
    pbar = tqdm(total=num_gens, desc='Generating...', leave=True)
    n_sent = 0
    for i, (name, sample, xrd_cont_x, xrd_cont_y, soap, acsf) in enumerate(test_dataset):

        # Start rep number
        start_rep = 0
        
        # Skip if eval file is already present
        if any(file_name.startswith(name) for file_name in existing_eval_files):
            # Count how many and adjust the num_rep
            num_evals = len(glob(os.path.join(eval_files_dir, name + '*.pkl')))
            num_adjusted_reps = num_reps - num_evals
            start_rep = num_evals
            pbar.update(num_evals)
            if num_adjusted_reps == 0:
                continue
        else:
            num_adjusted_reps = num_reps

        if debug_max and (i + 1) > debug_max:
            break  # Stop after reaching the debug_max limit
        if model is not None:
            prompt = extract_prompt(
                sample,
                model.device,
                add_composition=add_composition,
                add_spacegroup=add_spacegroup
            ).unsqueeze(0)
            if prompt is not None:
                prompt = prompt.repeat(num_adjusted_reps,1)
                token_ids = model.generate_batched_reps(prompt, max_new_tokens=max_new_tokens, disable_pbar=False).cpu().numpy()
                token_ids = [ids[ids != padding_id] for ids in token_ids]
                
                for j in range(num_adjusted_reps):
                    # Send the token ids and index to the worker without preprocessing
                    task = {
                        'token_ids': token_ids[j],
                        'name': name,
                        'index': i,
                        'rep': start_rep + j,
                        'xrd_parameters': metadata['xrd'],
                        'xrd_from_sample': {'q': xrd_cont_x.numpy(), 'iq': xrd_cont_y.numpy()},
                        'soap_from_sample': soap.numpy(),
                        'acsf_from_sample': acsf.numpy(),
                        'desc_parameters': metadata['descriptors'],
                        'species': metadata['species'],
                        'dataset': dataset_name,
                        'model': model_name,
                        'debug': debug,
                    }
                    input_queue.put(task)
                    n_sent += 1
        else:
            # We don't prompt, we simply pass the cifs to the workers
            #sample = sample[0]
            token_ids = sample[sample != padding_id].cpu().numpy()
            task = {
                'token_ids': token_ids,
                'name': name,
                'index': i,
                'rep': 0,
                'xrd_parameters': metadata['xrd'],
                'xrd_from_sample': {'q': xrd_cont_x.numpy(), 'iq': xrd_cont_y.numpy()},
                'soap_from_sample': soap.numpy(),
                'acsf_from_sample': acsf.numpy(),
                'desc_parameters': metadata['descriptors'],
                'species': metadata['species'],
                'dataset': dataset_name,
                'model': "NoModel",
                'debug': debug,
            }
            input_queue.put(task)
            n_sent += 1

        pbar.update(1)
    pbar.close()

    # Terminate workers
    for _ in range(num_workers):
        input_queue.put(None)  # Poison pill

    # Collect results
    n_received = 0
    pbar = tqdm(total=n_sent, desc='Evaluating...', leave=True)
    while n_received < n_sent:
        try:
            message = output_queue.get(timeout=1)
            idx = message['index']
            status = message['status']
            if status == 'fail':
                invalid_cifs += 1
            elif status == 'error':
                error_cifs += 1
                if debug:
                    print(f"Worker error for index {idx}: {message['error_msg']}")
            n_received += 1
        except Empty:
            continue
        finally:
            pbar.update(1)
    pbar.close()

    # Collect evaluated data from pickles
    eval_files = glob(os.path.join(eval_files_dir, '*.pkl'))
    for f in eval_files:
        with open(f, 'rb') as infile:
            eval_result = pickle.load(infile)
            evaluations.append(eval_result)
    
    print(f"Processed {n_sent} samples in {(time() - start):.3f} seconds.") 
    print(f"Total number of successful evaluations: {len(evaluations)}.")

    # Convert evaluations to DataFrame
    df = pd.json_normalize(evaluations)

    # Write to Parquet file
    out_file_path = os.path.join(out_folder_path, dataset_name + '.eval')
    df.to_parquet(out_file_path, compression='snappy')

    return evaluations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and evaluate CIF files using multiprocessing.')

    parser.add_argument('--config-path', type=str, required=True,
                        help='Path to the configuration YAML file (default: ../testconfig.yaml).')

    parser.add_argument('--root', type=str, default='./',
                        help='Root directory path (default: ./).')

    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                        help='Number of worker processes to use (default: number of CPU cores minus one).')

    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to the dataset HDF5 file. If not specified, it is derived from the config.')

    parser.add_argument('--out-folder', type=str, default=None,
                    help='Path to the output folder (default: None).')

    parser.add_argument('--debug-max', type=int, default=None,
                        help='Maximum number of samples to process for debugging purposes (default: process all samples).')

    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional output.')

    # No model evaluation
    parser.add_argument("--no-model", action='store_true', 
                        help='Enable no-model mode, evaluation from samples from given dataset')

    # add_composition, add_spacegroup, and max_new_tokens
    parser.add_argument('--add-composition', action='store_true',
                        help='Include composition in the prompt (default: exclude).')

    parser.add_argument('--add-spacegroup', action='store_true',
                        help='Include spacegroup in the prompt (default: exclude).')

    parser.add_argument('--max-new-tokens', type=int, default=1000,
                        help='Maximum number of new tokens to generate (default: 1000).')

    parser.add_argument('--dataset-name', type=str, default='DefaultDataset',
                    help='Name of the dataset, will name the eval file (default: DefaultDataset).')

    parser.add_argument('--model-name', type=str, default='DefaultModel',
                    help='Name of the model (default: DefaultModel).')

    parser.add_argument('--num-reps', type=int, default=1, 
                        help='Number of repetitions of generation attempt per sample')
    
    parser.add_argument('--block-size', type=int, default=10000, 
                        help='Block size of the dataset class')

    # Set defaults for add_composition and add_spacegroup
    parser.set_defaults(add_composition=False, add_spacegroup=False)

    args = parser.parse_args()

    # Get config
    with open(args.config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    config = OmegaConf.create(yaml_config)

    # Determine dataset path
    h5_test_path = args.dataset_path
    block_size = args.block_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine checkpoint path
    ckpt_path = os.path.join(args.root, config["out_dir"], 'ckpt.pt')  # Update with your actual checkpoint path

    if args.no_model is False:
        if os.path.exists(ckpt_path):
            model = load_model_from_checkpoint(ckpt_path, device)
            model.eval()  # Set the model to evaluation mode
        else:
            print(f"Checkpoint file not found at {ckpt_path}")
            sys.exit(1)
    else:
        model = None

    # Set up multiprocessing queues
    input_queue = mp.Queue()
    output_queue = mp.Queue(maxsize=1000)
    manager = mp.Manager()

    # Directory for evaluation
    if args.out_folder is not None:
        out_folder = args.out_folder
        os.makedirs(out_folder, exist_ok=True)
    else:
        if args.no_model is False:
            out_folder = os.path.dirname(ckpt_path)
        else:
            out_folder = os.path.dirname(h5_test_path)
    
    # Directory for individual files
    eval_files_dir = os.path.join(out_folder, "eval_files", args.dataset_name)
    os.makedirs(eval_files_dir, exist_ok=True)

    # Start worker processes
    num_workers = args.num_workers
    processes = [mp.Process(target=worker, args=(input_queue, output_queue, eval_files_dir)) for _ in range(num_workers)]
    for p in processes:
        p.start()

    # Call process_dataset with new arguments
    evaluations = process_dataset(
        h5_test_path,
        block_size,
        model,
        input_queue,
        output_queue,
        eval_files_dir,
        num_workers,
        out_folder_path=out_folder,
        debug_max=args.debug_max,
        debug=args.debug,
        add_composition=args.add_composition,
        add_spacegroup=args.add_spacegroup,
        max_new_tokens=args.max_new_tokens,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        num_reps=args.num_reps,
    )

    # Join worker processes
    for p in processes:
        p.join()
