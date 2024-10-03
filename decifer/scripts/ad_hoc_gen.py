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
from torch.utils.data import DataLoader

# Import custom modules
from decifer import (
    HDF5Dataset,
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

    except Exception as e:
        raise e
        
    finally:
        return eval_dict

def main():
    # Initialize argparse for input handling
    parser = argparse.ArgumentParser(description="Generate CIF based on composition and/or space group")
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset HDF5 file')
    parser.add_argument('--model', type=str, help='Path to model checkpoint', required=True)
    parser.add_argument('--max-new-tokens', type=int, help='Max num. tokens to generate', default=3_000)
    parser.add_argument('--model-block-size', type=int, help='Model block size', default=10_000)
    parser.add_argument('--add-composition', action='store_true', help='Adding compostion')
    parser.add_argument('--add-spacegroup', action='store_true', help='Adding spacegroup')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if os.path.exists(args.model):
        model = load_model_from_checkpoint(args.model, args.device)
    else:
        print(f"Checkpoint file not found at {args.model}")
        sys.exit(1)
        
    model.eval()  # Set the model to evaluation mode
    
    # Make dataset from test
    dataset = HDF5Dataset(
        args.dataset_path,
        ["cif_tokenized"], 
        block_size=args.model_block_size,
    )
    
    # Create DataLoader to load data from the dataset
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Get a random sample from the dataset
    sample = next(iter(data_loader))
    
    # Extract prompt
    prompt = extract_prompt(
            sample[0][0],
            model.device,
            add_composition=args.add_composition,
            add_spacegroup=args.add_spacegroup
        ).unsqueeze(0)

    # Call the CIF generation function with the provided arguments
    out_tokens = model.generate(prompt, max_new_tokens=args.max_new_tokens)[0].cpu().numpy()
    decode = Tokenizer().decode
    
    try:
        cif = decode(out_tokens)
        print(cif)
        cif = replace_symmetry_loop_with_P1(cif)
        spacegroup_symbol = extract_space_group_symbol(cif)
        if spacegroup_symbol != "P 1":
            cif = reinstate_symmetry_loop(cif, spacegroup_symbol)
    
        print(cif)
        sensible = is_sensible(cif)
        print("sensible", sensible)
        if sensible:
            eval_results = evaluate_cif(cif)
            print('syntax', evaluate_syntax_validity(cif))
    except:
        raise Exception()

    #print(evaluate_cif(cif))
    
#     if is_sensible(cif):
#         # Evaluate the CIF
#         eval_result = evaluate_cif(cif)
#         # Add 'Dataset' and 'Model' to eval_result
# #         eval_result['seq_len'] = len(out_tokens)
# #         print(eval_result)

if __name__ == "__main__":
    main()
