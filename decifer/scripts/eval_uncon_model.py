#!/usr/bin/env python3

# Standard library imports
import os
import sys
import argparse
import multiprocessing as mp
from multiprocessing import Queue
from queue import Empty
from time import time
from queue import Empty
from glob import glob
import pickle
import json

# Third-party library imports
import yaml
from omegaconf import OmegaConf
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.cif import CifParser

# Conditional imports for backwards compatibility with older pymatgen versions
try:
    parser_from_string = CifParser.from_str
except AttributeError:
    parser_from_string = CifParser.from_string

# Dscribe descriptors (used for crystal descriptor calculations)
from dscribe.descriptors import SOAP, ACSF

# Local module imports (custom modules)
from decifer import (
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
    is_space_group_consistent,
    is_atom_site_multiplicity_consistent,
    is_formula_consistent,
    bond_length_reasonableness_score,
    extract_species,
    extract_composition,
)

def safe_extract(extract_function, *args, return_boolean=False):
    """
    Safely executes a given extraction function, handling any exceptions that may occur during execution.

    Args:
        extract_function (callable): The function to be called for extracting a value. This function may 
                                     take multiple arguments, provided in *args.
        *args: Arguments to pass to the `extract_function`.
        return_boolean (bool, optional): If True, the function returns a tuple (result, success_flag), 
                                         where success_flag is True if the function executed successfully 
                                         and False if an exception occurred. Defaults to False.

    Returns:
        result: The result of `extract_function(*args)` if successful, otherwise `None`.
        tuple: If `return_boolean=True`, returns a tuple `(result, success_flag)` where success_flag is a 
               boolean indicating whether the function executed successfully.
    """
    try:
        # Attempt to execute the provided extraction function with the given arguments
        result = extract_function(*args)
        
        # Return result along with success flag if return_boolean is set to True
        return (result, True) if return_boolean else result
    except Exception:
        # Return None along with failure flag if return_boolean is set to True
        return (None, False) if return_boolean else None

def get_statistics(cif_string):
    """
    Extracts statistical properties and validity checks for a given CIF (Crystallographic Information File) string.

    This function evaluates a CIF string to determine the validity of key structural properties (e.g., formula, 
    site multiplicity, bond lengths, and spacegroup), and extracts numeric properties of the unit cell, such as
    cell lengths, angles, and volume. Safe extraction methods are used to handle missing or invalid data.

    Args:
        cif_string (str): The CIF data string representing the crystal structure.

    Returns:
        dict: A dictionary containing the following keys:
            - 'cif_string': The original CIF string.
            - 'validity': A dictionary indicating the validity of the formula, site multiplicity, bond lengths, 
              and spacegroup.
            - 'cell_params': A dictionary with numeric properties of the unit cell (a, b, c, alpha, beta, gamma, 
              implied volume, and generated volume).
            - 'spacegroup': Extracted spacegroup symbol.
            - 'composition': Chemical composition of the structure (if extractable).
            - 'species': List of atomic species in the structure.
    """
    
    # Define the dictionary to hold the CIF statistics and validity checks
    stat_dict = {
        'cif_string': cif_string,
        'validity': {
            'formula': False,
            'site_multiplicity': False,
            'bond_length': False,
            'spacegroup': False,
        },
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
        'spacegroup': None,
        'composition': None,
        'species': None,
    }
    
    # Extract validity checks and update the validity dictionary
    stat_dict['validity']['formula'], _ = safe_extract(is_formula_consistent, cif_string, return_boolean=True)
    stat_dict['validity']['site_multiplicity'], _ = safe_extract(is_atom_site_multiplicity_consistent, cif_string, return_boolean=True)
    stat_dict['validity']['bond_length'], _ = safe_extract(lambda cif: bond_length_reasonableness_score(cif) >= 1.0, cif_string, return_boolean=True)
    stat_dict['validity']['spacegroup'], _ = safe_extract(is_space_group_consistent, cif_string, return_boolean=True)
    
    # Safely extract numeric properties of the unit cell
    a = safe_extract(extract_numeric_property, cif_string, "_cell_length_a")
    b = safe_extract(extract_numeric_property, cif_string, "_cell_length_b")
    c = safe_extract(extract_numeric_property, cif_string, "_cell_length_c")
    alpha = safe_extract(extract_numeric_property, cif_string, "_cell_angle_alpha")
    beta = safe_extract(extract_numeric_property, cif_string, "_cell_angle_beta")
    gamma = safe_extract(extract_numeric_property, cif_string, "_cell_angle_gamma")
    
    # Safely compute volumes
    implied_vol = safe_extract(get_unit_cell_volume, a, b, c, alpha, beta, gamma)
    gen_vol = safe_extract(extract_volume, cif_string)
    
    # Update the 'cell_params' dictionary with the extracted properties
    stat_dict['cell_params'].update({
        'a': a,
        'b': b,
        'c': c,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'implied_vol': implied_vol,
        'gen_vol': gen_vol
    })
    stat_dict['spacegroup'] = safe_extract(extract_space_group_symbol, cif_string)
    stat_dict['species'] = safe_extract(extract_species, cif_string)
    stat_dict['composition'] = safe_extract(extract_composition, cif_string)
    
    return stat_dict


def calculate_xrd(structure_name, cif_string, xrd_params, debug=False):
    """
    Calculates the X-ray diffraction (XRD) pattern for a given CIF structure.

    This function takes a CIF string, extracts its atomic structure, and computes the 
    XRD pattern based on the provided parameters. It applies Gaussian broadening 
    to the XRD peaks and optionally adds noise based on a signal-to-noise ratio (SNR).

    Args:
        structure_name (str): The name of the structure being processed (for debugging purposes).
        cif_string (str): The CIF data string representing the structure.
        xrd_params (dict): Dictionary containing XRD parameters with the following keys:
            - 'wavelength': The wavelength used for XRD calculation.
            - 'qmin': The minimum value of Q (momentum transfer) for the output grid.
            - 'qmax': The maximum value of Q for the output grid.
            - 'qstep': The step size for the Q grid.
            - 'fwhm': The full width at half maximum for the Gaussian peak broadening.
            - 'snr': The signal-to-noise ratio for adding noise (lower values add more noise).
        debug (bool, optional): If True, additional debugging information will be printed. Defaults to False.

    Returns:
        dict: A dictionary with two keys:
            - 'q': The Q values (momentum transfer) on a regular grid.
            - 'iq': The corresponding intensities (normalized), with noise applied if specified.
            Returns None if an error occurs during the calculation.
    """
    try:
        # Parse the CIF string to get the structure
        structure = parser_from_string(cif_string).get_structures()[0]
        
        # Initialize the XRD calculator using the specified wavelength
        xrd_calculator = XRDCalculator(wavelength=xrd_params['wavelength'])
        
        # Calculate the XRD pattern from the structure
        xrd_pattern = xrd_calculator.get_pattern(structure)
    
    except Exception as e:
        if debug:
            print(f"Error processing {structure_name}: {e}")
        return None
    
    # Convert 2θ (xrd_pattern.x) to Q (momentum transfer)
    theta_radians = np.radians(xrd_pattern.x / 2)
    q_values = 4 * np.pi * np.sin(theta_radians) / xrd_calculator.wavelength
    
    # Normalize intensities
    intensities_normalized = xrd_pattern.y / (np.max(xrd_pattern.y) + 1e-16)
    
    # Define the continuous Q grid
    q_continuous = np.arange(xrd_params['qmin'], xrd_params['qmax'], xrd_params['qstep'])
    
    # Initialize the continuous intensity array
    intensities_continuous = np.zeros_like(q_continuous)
    
    # Apply Gaussian broadening to the peaks
    for q_peak, intensity in zip(q_values, xrd_pattern.y):
        gaussian_broadening = intensity * np.exp(-0.5 * ((q_continuous - q_peak) / xrd_params['fwhm']) ** 2)
        intensities_continuous += gaussian_broadening
    
    # Normalize the continuous intensities
    intensities_continuous /= (np.max(intensities_continuous) + 1e-16)
    
    # Add noise to the signal if the SNR is less than 100
    if xrd_params['snr'] < 100.:
        noise = np.random.normal(0, np.max(intensities_continuous) / xrd_params['snr'], size=intensities_continuous.shape)
        intensities_continuous = np.clip(intensities_continuous + noise, 0, None)

    return {'q': q_continuous, 'iq': intensities_continuous}

def calculate_crystal_descriptors(structure_name, cif_string, species_list, descriptor_params, debug=False):
    """
    Calculates crystal structure descriptors (SOAP and ACSF) for a given CIF structure.

    This function processes a CIF string to extract the atomic structure, then calculates 
    two types of descriptors: SOAP (Smooth Overlap of Atomic Positions) and ACSF (Atom-Centered 
    Symmetry Functions).

    Args:
        structure_name (str): The name of the structure being processed (for debugging purposes).
        cif_string (str): The CIF data string representing the structure.
        species_list (list): List of atomic species present in the structure.
        descriptor_params (dict): Dictionary containing descriptor calculation parameters:
            - 'soap': Dictionary with SOAP descriptor parameters:
                - 'r_cut': Cutoff radius.
                - 'n_max': Maximum number of radial basis functions.
                - 'l_max': Maximum number of spherical harmonics.
                - 'sigma': Width of the Gaussian broadening.
                - 'rbf': Radial basis function type.
                - 'compression_mode': Mode of compression (e.g., crossover).
                - 'periodic': Whether the system is periodic.
                - 'sparse': Whether the result should be sparse.
            - 'acsf': Dictionary with ACSF descriptor parameters:
                - 'r_cut': Cutoff radius.
                - 'periodic': Whether the system is periodic.
        debug (bool, optional): If True, prints additional debug information. Defaults to False.

    Returns:
        dict: A dictionary containing the calculated descriptors:
            - 'SOAP': The SOAP descriptor array.
            - 'ACSF': The ACSF descriptor array.
        Returns None if an error occurs during the calculation.
    """
    try:
        # Parse the CIF string and convert the structure to ASE format
        ase_structure = parser_from_string(cif_string).get_structures()[0].to_ase_atoms()
        
        # Initialize SOAP descriptor with specified parameters
        soap_descriptor = SOAP(
            species=species_list,
            r_cut=descriptor_params['soap']['r_cut'],
            n_max=descriptor_params['soap']['n_max'],
            l_max=descriptor_params['soap']['l_max'],
            sigma=descriptor_params['soap']['sigma'],
            rbf=descriptor_params['soap']['rbf'],
            compression={'mode': descriptor_params['soap']['compression_mode']},
            periodic=descriptor_params['soap']['periodic'],
            sparse=descriptor_params['soap']['sparse'],
        )

        # Initialize ACSF descriptor with specified parameters
        acsf_descriptor = ACSF(
            species=species_list,
            r_cut=descriptor_params['acsf']['r_cut'],
            periodic=descriptor_params['acsf']['periodic'],
        )

        # Calculate the descriptors for the atomic structure
        descriptor_dict = {
            'SOAP': soap_descriptor.create(ase_structure, centers=[0])[0],
            'ACSF': acsf_descriptor.create(ase_structure, centers=[0])[0]
        }

        return descriptor_dict

    except Exception as e:
        if debug:
            print(f"Error processing {structure_name}: {e}")
        return None

def worker(input_queue, eval_files_dir, done_queue):
    """
    Worker process that retrieves tasks from the input queue, processes CIF files, calculates descriptors, 
    and saves evaluation results to the specified directory.

    This function operates in an infinite loop, processing each task from the input queue until it encounters
    a `None` task, signaling the end of the queue. For each CIF file, it extracts the space group, applies
    symmetry modifications, evaluates its validity, and calculates descriptors like XRD, SOAP, and ACSF. 
    The results are then saved to the disk.

    Args:
        input_queue (multiprocessing.Queue): Queue that provides tasks containing CIF data and metadata.
        eval_files_dir (str): Directory where evaluation results will be saved as pickle files.
    
    Task Format (dict):
        Each task in the queue is expected to have the following structure:
            - 'token_ids': List of token IDs representing the CIF structure.
            - 'name': Name of the structure being processed.
            - 'index': Index of the task for tracking purposes.
            - 'rep': Repetition number for multiple attempts.
            - 'xrd_parameters': Parameters for XRD calculation.
            - 'xrd_from_sample': XRD data from the sample for comparison.
            - 'species': Atomic species involved in the structure.
            - 'desc_parameters': Parameters for descriptor (SOAP, ACSF) calculations.
            - 'dataset': Name of the dataset (optional, defaults to 'UnknownDataset').
            - 'model': Name of the model (optional, defaults to 'UnknownModel').
            - 'debug': Boolean flag for enabling debug mode.
    
    Returns:
        None: The worker function continuously processes tasks until a `None` task is encountered, 
              signaling termination.
    """
    
    # Initialize the tokenizer decoder function
    decode = Tokenizer().decode

    while True:
        # Fetch task from the input queue
        task = input_queue.get()

        # If a `None` task is received, terminate the worker
        if task is None:
            break

        try:
            # Decode tokenized CIF structure into a string
            cif_string = decode(task['token_ids'])

            # Replace symmetry loop with primitive cell ("P1")
            cif_string = replace_symmetry_loop_with_P1(cif_string)

            # Extract space group symbol from the CIF string
            spacegroup_symbol = extract_space_group_symbol(cif_string)

            # If the space group is not "P1", reinstate symmetry
            if spacegroup_symbol != "P 1":
                cif_string = reinstate_symmetry_loop(cif_string, spacegroup_symbol)

            # Check if the CIF structure is sensible (valid)
            if is_sensible(cif_string):
                # Evaluate CIF validity and structure
                eval_result = get_statistics(cif_string)

                # Update evaluation result with metadata and calculated descriptors
                eval_result.update({
                    'index': task['index'],
                    'rep': task['rep'],
                    'descriptors': {
                        'xrd_gen': calculate_xrd(task['name'], cif_string, task['xrd_parameters'], task['debug']),
                        'xrd_sample': task['xrd_from_sample'],
                        'soap_gen': calculate_crystal_descriptors(task['name'], cif_string, task['species'], task['desc_parameters'], task['debug'])['SOAP'],
                        'acsf_gen': calculate_crystal_descriptors(task['name'], cif_string, task['species'], task['desc_parameters'], task['debug'])['ACSF'],
                    },
                    'dataset_name': task.get('dataset', 'UnknownDataset'),
                    'model_name': task.get('model', 'UnknownModel'),
                    'seq_len': len(task['token_ids']),
                    'status': 'success'
                })

                # Save the evaluation result to a file
                save_evaluation(eval_result, task['name'], task['rep'], eval_files_dir)

            else:
                # If the CIF is not sensible, mark the task as failed
                save_evaluation({'status': 'fail', 'index': task['index'], 'rep': task['rep']}, task['name'], task['rep'], eval_files_dir)

        except Exception as e:
            # In case of error, save error information
            save_evaluation({'status': 'error', 'error_msg': str(e), 'index': task['index'], 'rep': task['rep']}, task['name'], task['rep'], eval_files_dir)
        finally:
            # Signal task completion
            done_queue.put(1)

def save_evaluation(eval_result, structure_name, repetition_num, eval_files_dir):
    """
    Saves the evaluation result to disk as a pickle file, ensuring atomicity.

    This function saves the evaluation result to a temporary file first, then renames it 
    to the final output file. This method ensures that incomplete files are not left 
    in the destination folder in case of a failure during the writing process.

    Args:
        eval_result (dict): The evaluation result to be saved. It contains metadata, 
                            descriptors, and status of the evaluation.
        structure_name (str): The name of the structure being processed. This is used 
                              in the filename.
        repetition_num (int): The repetition number for the current evaluation, used 
                              to differentiate multiple runs for the same structure.
        eval_files_dir (str): The directory where the evaluation result will be saved.

    Returns:
        None: The function writes the result to disk and does not return any value.
    """
    
    # Construct the final output filename and a temporary filename
    output_filename = os.path.join(eval_files_dir, f"{structure_name}_{repetition_num}.pkl")
    temp_filename = output_filename + '.tmp'

    try:
        # Write evaluation result to a temporary file
        with open(temp_filename, 'wb') as temp_file:
            pickle.dump(eval_result, temp_file)
        
        # Rename the temporary file to the final output file to ensure atomic save
        os.rename(temp_filename, output_filename)

    except Exception as e:
        # Log the error if writing to disk fails (optional: add logging or debug prints)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)  # Clean up incomplete temporary file
        raise IOError(f"Failed to save evaluation for {structure_name} (rep {repetition_num}): {e}")

def process_dataset(h5_test_path, model, input_queue, eval_files_dir, num_workers, override, **kwargs):
    """
    Processes a dataset, generates tokenized CIF prompts, and dispatches tasks to worker processes.

    This function loads a dataset from an HDF5 file, iterates over its samples, and generates token sequences 
    from CIF structures using a given model. It creates a task for each sample (and its repetitions) and 
    sends these tasks to the input queue for worker processes to consume. The function also manages existing 
    evaluations to avoid redundant work.

    Args:
        h5_test_path (str): Path to the HDF5 file containing the test dataset.
        model (torch.nn.Module): The model used to generate tokenized prompts from the CIF samples.
        input_queue (multiprocessing.Queue): Queue to which tasks are submitted for worker processes.
        eval_files_dir (str): Directory where evaluation files are saved. This is used to check for already processed files.
        num_workers (int): Number of worker processes that will consume tasks from the input queue.
        **kwargs: Additional keyword arguments, including:
            - 'num_reps' (int): Number of repetitions for generating new sequences for each sample.
            - 'debug_max' (int, optional): Maximum number of samples to process in debug mode.
            - 'add_composition' (bool): Whether to include composition information in the prompt.
            - 'add_spacegroup' (bool): Whether to include spacegroup information in the prompt.
            - 'max_new_tokens' (int): Maximum number of new tokens to generate.
            - 'xrd_parameters' (dict): Parameters for the XRD calculation.
            - 'desc_parameters' (dict): Parameters for the descriptor (SOAP, ACSF) calculations.
            - 'species' (list): List of species in the dataset.
            - 'dataset_name' (str): Name of the dataset for saving purposes.
            - 'model_name' (str): Name of the model for saving purposes.
            - 'debug' (bool): If True, enables debug mode with additional output.
            - 'override' (bool): If True, ignores check of exisiting files.
    
    Returns:
        None: The function puts tasks into the input queue for workers and terminates when the queue is exhausted.
    """
    
    # Load the test dataset from the HDF5 file, including necessary features like CIF, XRD
    test_dataset = DeciferDataset(h5_test_path, ["cif_name", "cif_tokenized", "xrd_cont.q", "xrd_cont.iq"])

    # Set of existing evaluation files to avoid redundant processing
    existing_eval_files = set(os.path.basename(f) for f in glob(os.path.join(eval_files_dir, "*.pkl")))
    
    # Calculate the number of generations to process, constrained by `debug_max` if provided
    num_generations = min(len(test_dataset) * kwargs['num_reps'], kwargs.get('debug_max', len(test_dataset)) * kwargs['num_reps'])
    num_send = num_generations

    # Progress bar to track the processing status
    pbar = tqdm(total=num_generations, desc='Generating and parsing evaluation tasks...', leave=True)

    # Tokenizer (for padding id)
    padding_id = Tokenizer().padding_id

    # Iterate over the dataset samples
    for i, (structure_name, sample, xrd_cont_x, xrd_cont_y) in enumerate(test_dataset):
        if i >= num_generations:
            break  # Stop processing if we've reached the generation limit (in debug mode)
        
        # Skip processing if evaluation for this structure already exists
        if not override:
            if any(f.startswith(structure_name) for f in existing_eval_files):
                pbar.update(1)
                num_send -= 1
                continue

        if model is None:
            prompt = None
        else:
            # Generate tokenized prompt from the sample using the model
            prompt = extract_prompt(sample, model.device, add_composition=kwargs['add_composition'], add_spacegroup=kwargs['add_spacegroup']).unsqueeze(0)
        
        if prompt is not None:
            # Generate token sequences from the model's output
            token_ids = model.generate_batched_reps(prompt, max_new_tokens=kwargs['max_new_tokens']).cpu().numpy()
            token_ids = [ids[ids != padding_id] for ids in token_ids]  # Remove padding tokens
        else:
            token_ids = [sample[sample != padding_id].cpu().numpy()]

        # Create tasks for each repetition
        for rep_num in range(kwargs['num_reps']):
            task = {
                'token_ids': token_ids[rep_num],
                'name': structure_name,
                'index': i,
                'rep': rep_num,
                'xrd_parameters': kwargs['xrd_parameters'],
                'xrd_from_sample': {'q': xrd_cont_x.numpy(), 'iq': xrd_cont_y.numpy()},
                'desc_parameters': kwargs['desc_parameters'],
                'species': kwargs['species'],
                'dataset': kwargs['dataset_name'],
                'model': kwargs['model_name'],
                'debug': kwargs['debug'],
            }
            
            # Put the task into the input queue for worker processes to consume
            input_queue.put(task)

        # Update the progress bar after each sample
        pbar.update(1)
    
    pbar.close()  # Close the progress bar after processing is complete

    # Send termination signals (None) to each worker to indicate that the tasks are complete
    for _ in range(num_workers):
        input_queue.put(None)

    return num_generations, num_send

def collect_results(eval_files_dir):
    """
    Collects all evaluation results from the saved pickle files in the specified directory.

    Args:
        eval_files_dir (str): Directory where evaluation results are stored as pickle files.

    Returns:
        list: A list of evaluation results (dictionaries) loaded from the pickle files.
    """
    eval_files = glob(os.path.join(eval_files_dir, '*.pkl'))
    evaluations = []
    
    # Load each pickle file and append the evaluation result to the evaluations list
    for eval_file in eval_files:
        with open(eval_file, 'rb') as infile:
            eval_result = pickle.load(infile)
            evaluations.append(eval_result)
    
    return evaluations

def save_evaluations_to_parquet(evaluations, out_folder, dataset_name, duration):
    """
    Saves the collected evaluation results to a Parquet file.

    Args:
        evaluations (list): List of evaluation results (dictionaries) to be saved.
        out_folder (str): Directory where the Parquet file will be saved.
        dataset_name (str): Name of the dataset, used as part of the Parquet filename.
        duration (float): Time duration (in seconds) it took to process the evaluations.

    Returns:
        None: The function writes the evaluations to a Parquet file and prints a summary.
    """
    # Convert evaluation results to a pandas DataFrame
    df = pd.json_normalize(evaluations)
    
    # Construct the output Parquet file path
    out_file_path = os.path.join(out_folder, dataset_name + '.eval')
    
    # Save the DataFrame to Parquet format with snappy compression
    df.to_parquet(out_file_path, compression='snappy')
    
    # Print the summary of the processed samples and the time taken
    print(f"Processed {len(evaluations)} samples in {duration:.3f} seconds.")

def main():
    """
    Main function to process and evaluate CIF files using multiprocessing.

    Command-line Arguments:
        --config-path (str): Path to the YAML configuration file.
        --root (str): Root directory path (default: current directory).
        --num-workers (int): Number of worker processes to use (default: number of CPU cores minus one).
        --dataset-path (str): Path to the dataset HDF5 file.
        --out-folder (str): Directory where output files will be saved.
        --debug-max (int): Maximum number of samples to process in debug mode.
        --debug (bool): Enable debug mode with additional output.
        --no-model (bool): Disable model usage and evaluate from dataset only.
        --add-composition (bool): Include composition information in the prompt.
        --add-spacegroup (bool): Include spacegroup information in the prompt.
        --max-new-tokens (int): Maximum number of new tokens to generate for CIF structures.
        --dataset-name (str): Name of the dataset for saving evaluation results.
        --model-name (str): Name of the model for saving evaluation results.
        --num-reps (int): Number of repetitions per sample for CIF generation.
        --collect (bool): Flag to collect evaluation files and combine them.

    Returns:
        None: The function manages dataset processing, task distribution, and evaluation collection.
    """
    parser = argparse.ArgumentParser(description='Process and evaluate CIF files using multiprocessing.')

    # Argument parsing for required and optional arguments
    parser.add_argument('--config-path', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument('--root', type=str, default='./', help='Root directory path.')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1, help='Number of worker processes.')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset HDF5 file.')
    parser.add_argument('--out-folder', type=str, default=None, help='Path to the output folder.')
    parser.add_argument('--debug-max', type=int, default=None, help='Maximum number of samples to process for debugging.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional output.')
    parser.add_argument('--no-model', action='store_true', help='Enable no-model mode.')
    parser.add_argument('--add-composition', action='store_true', help='Include composition in the prompt.')
    parser.add_argument('--add-spacegroup', action='store_true', help='Include spacegroup in the prompt.')
    parser.add_argument('--max-new-tokens', type=int, default=1000, help='Maximum number of new tokens to generate.')
    parser.add_argument('--dataset-name', type=str, default='default_dataset', help='Name of the dataset.')
    parser.add_argument('--model-name', type=str, default='default_model', help='Name of the model.')
    parser.add_argument('--num-reps', type=int, default=1, help='Number of repetitions per sample.')
    parser.add_argument('--collect-only', action='store_true', help='Just collect eval files and combine.')
    parser.add_argument('--override', action='store_true', help='Overrides the presence of existing files, effectively generating everything from scratch')

    # Argument parsing for required and optional arguments
    parser.add_argument('--soap-r_cut', type=float, default=None, help='SOAP: Cutoff radius.')
    parser.add_argument('--soap-n_max', type=int, default=None, help='SOAP: Maximum number of radial basis functions.')
    parser.add_argument('--soap-l_max', type=int, default=None, help='SOAP: Maximum number of spherical harmonics.')
    parser.add_argument('--soap-sigma', type=float, default=None, help='SOAP: Width of Gaussian broadening.')
    parser.add_argument('--soap-rbf', type=str, default=None, help='SOAP: Radial basis function type.')
    parser.add_argument('--soap-compression_mode', type=str, default=None, help='SOAP: Compression mode (e.g., crossover).')
    parser.add_argument('--soap-periodic', type=bool, default=None, help='SOAP: Periodicity of the system.')
    parser.add_argument('--soap-sparse', type=bool, default=None, help='SOAP: Whether the result should be sparse.')

    parser.add_argument('--acsf-r_cut', type=float, default=None, help='ACSF: Cutoff radius.')
    parser.add_argument('--acsf-periodic', type=bool, default=None, help='ACSF: Periodicity of the system.')

    # Parse command-line arguments
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    config = OmegaConf.create(yaml_config)

    # Set paths and initialize model/device
    h5_test_path = args.dataset_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model from checkpoint, unless in no-model mode
    ckpt_path = os.path.join(args.root, config["out_dir"], 'ckpt.pt')
    model = None
    if not args.no_model and os.path.exists(ckpt_path):
        model = load_model_from_checkpoint(ckpt_path, device)
        model.eval()
    elif not args.no_model:
        print(f"Checkpoint not found at {ckpt_path}")
        sys.exit(1)
    
    # Extract metadata and set default values for descriptor parameters
    metadata_path = os.path.join(os.path.dirname(os.path.dirname(h5_test_path)), "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Set default descriptor values if they are not present in the metadata
    default_descriptor_params = {
        'soap': {
            'r_cut': 6.0,
            'n_max': 8,
            'l_max': 6,
            'sigma': 1.0,
            'rbf': 'gto',
            'compression_mode': 'crossover',
            'periodic': True,
            'sparse': False,
        },
        'acsf': {
            'r_cut': 6.0,
            'periodic': True,
        }
    }

    # Set default descriptor values if they are not present in the metadata
    metadata['descriptors'] = metadata.get('descriptors', default_descriptor_params)
    metadata['descriptors']['soap'] = {**default_descriptor_params['soap'], **metadata['descriptors'].get('soap', {})}
    metadata['descriptors']['acsf'] = {**default_descriptor_params['acsf'], **metadata['descriptors'].get('acsf', {})}

    # Update descriptor parameters based on command-line arguments (if provided)
    if args.soap_r_cut is not None:
        metadata['descriptors']['soap']['r_cut'] = args.soap_r_cut
    if args.soap_n_max is not None:
        metadata['descriptors']['soap']['n_max'] = args.soap_n_max
    if args.soap_l_max is not None:
        metadata['descriptors']['soap']['l_max'] = args.soap_l_max
    if args.soap_sigma is not None:
        metadata['descriptors']['soap']['sigma'] = args.soap_sigma
    if args.soap_rbf is not None:
        metadata['descriptors']['soap']['rbf'] = args.soap_rbf
    if args.soap_compression_mode is not None:
        metadata['descriptors']['soap']['compression_mode'] = args.soap_compression_mode
    if args.soap_periodic is not None:
        metadata['descriptors']['soap']['periodic'] = args.soap_periodic
    if args.soap_sparse is not None:
        metadata['descriptors']['soap']['sparse'] = args.soap_sparse

    if args.acsf_r_cut is not None:
        metadata['descriptors']['acsf']['r_cut'] = args.acsf_r_cut
    if args.acsf_periodic is not None:
        metadata['descriptors']['acsf']['periodic'] = args.acsf_periodic

    # Directory for evaluation
    if args.out_folder is not None:
        out_folder = args.out_folder
        os.makedirs(out_folder, exist_ok=True)
    else:
        if args.no_model is False:
            out_folder = os.path.dirname(ckpt_path)
        else:
            out_folder = os.path.dirname(h5_test_path)

    # Set up multiprocessing queue and directories for evaluation files
    input_queue = mp.Queue()
    done_queue = mp.Queue()
    eval_files_dir = os.path.join(out_folder or os.path.dirname(h5_test_path), "eval_files", args.dataset_name)
    os.makedirs(eval_files_dir, exist_ok=True)

    # Collect and combine evaluation files if the --collect-only flag is used
    if args.collect_only:
        evaluations = collect_results(eval_files_dir)
        save_evaluations_to_parquet(evaluations, out_folder, args.dataset_name, time() - start)
    else:
        # Start worker processes for processing
        processes = [
            mp.Process(target=worker, args=(input_queue, eval_files_dir, done_queue))
            for _ in range(args.num_workers)
        ]
        
        for process in processes:
            process.start()

        # Start processing the dataset
        start = time()
        num_gens, num_send = process_dataset(
            h5_test_path,
            model,
            input_queue,
            eval_files_dir,
            args.num_workers,
            out_folder=out_folder,
            debug_max=args.debug_max,
            debug=args.debug,
            add_composition=args.add_composition,
            add_spacegroup=args.add_spacegroup,
            max_new_tokens=args.max_new_tokens,
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            num_reps=args.num_reps,
            xrd_parameters=metadata['xrd'],
            desc_parameters=metadata['descriptors'],
            species=metadata['species'],
            override=args.override,
        )

        if num_send > 0:
            # Create a new progress bar for task completion
            pbar = tqdm(total=num_gens, desc='Evaluating...', leave=True)
            # Monitor the done_queue and update the progress bar
            completed_tasks = 0
            while completed_tasks < num_gens:
                try:
                    # Wait for a task completion signal
                    done_queue.get(timeout=1)
                    # Update the progress bar
                    pbar.update(1)
                    completed_tasks += 1
                except Empty:
                    pass

            pbar.close()

        # Join worker processes after processing is complete
        for process in processes:
            process.join()

        # Collect results and save them to a Parquet file
        evaluations = collect_results(eval_files_dir)
        save_evaluations_to_parquet(evaluations, out_folder, args.dataset_name, time() - start)


if __name__ == '__main__':
    main()
