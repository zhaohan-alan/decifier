import os
import re
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob

from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .cif_utils import (
    extract_data_formula,
)

from .tokenizer import Tokenizer
from ..models import DeciferConfig, Decifer

import warnings
warnings.filterwarnings('ignore')

def bond_length_reasonableness_score(cif_str, tolerance=0.32, h_factor=2.5):
    """
    If a bond length is 30% shorter or longer than the sum of the atomic radii, the score is lower.
    """
    structure = Structure.from_str(cif_str, fmt="cif")
    crystal_nn = CrystalNN()

    min_ratio = 1 - tolerance
    max_ratio = 1 + tolerance

    # calculate the score based on bond lengths and covalent radii
    score = 0
    bond_count = 0
    for i, site in enumerate(structure):
        bonded_sites = crystal_nn.get_nn_info(structure, i)
        for connected_site_info in bonded_sites:
            j = connected_site_info['site_index']
            if i == j:  # skip if they're the same site
                continue
            connected_site = connected_site_info['site']
            bond_length = site.distance(connected_site)

            is_hydrogen_bond = "H" in [site.specie.symbol, connected_site.specie.symbol]

            electronegativity_diff = abs(site.specie.X - connected_site.specie.X)
            """
            According to the Pauling scale, when the electronegativity difference 
            between two bonded atoms is less than 1.7, the bond can be considered 
            to have predominantly covalent character, while a difference greater 
            than or equal to 1.7 indicates that the bond has significant ionic 
            character.
            """
            if electronegativity_diff >= 1.7:
                # use ionic radii
                if site.specie.X < connected_site.specie.X:
                    expected_length = site.specie.average_cationic_radius + connected_site.specie.average_anionic_radius
                else:
                    expected_length = site.specie.average_anionic_radius + connected_site.specie.average_cationic_radius
            else:
                expected_length = site.specie.atomic_radius + connected_site.specie.atomic_radius

            bond_ratio = bond_length / expected_length

            # penalize bond lengths that are too short or too long;
            #  check if bond involves hydrogen and adjust tolerance accordingly
            if is_hydrogen_bond:
                if bond_ratio < h_factor:
                    score += 1
            else:
                if min_ratio < bond_ratio < max_ratio:
                    score += 1

            bond_count += 1

    normalized_score = score / bond_count

    return normalized_score

def is_space_group_consistent(cif_str):
    structure = Structure.from_str(cif_str, fmt="cif")
    try:
        parser = CifParser.from_string(cif_str)
    except:
        parser = CifParser.from_str(cif_str)
    cif_data = parser.as_dict()

    # Extract the stated space group from the CIF file
    stated_space_group = cif_data[list(cif_data.keys())[0]]['_symmetry_space_group_name_H-M']

    # Analyze the symmetry of the structure
    spacegroup_analyzer = SpacegroupAnalyzer(structure, symprec=0.1)

    # Get the detected space group
    detected_space_group = spacegroup_analyzer.get_space_group_symbol()

    # Check if the detected space group matches the stated space group
    is_match = stated_space_group.strip() == detected_space_group.strip()

    return is_match

def is_formula_consistent(cif_str):
    try:
        parser = CifParser.from_string(cif_str)
    except:
        parser = CifParser.from_str(cif_str)

    cif_data = parser.as_dict()

    formula_data = Composition(extract_data_formula(cif_str))
    formula_sum = Composition(cif_data[list(cif_data.keys())[0]]["_chemical_formula_sum"])
    formula_structural = Composition(cif_data[list(cif_data.keys())[0]]["_chemical_formula_structural"])

    return formula_data.reduced_formula == formula_sum.reduced_formula == formula_structural.reduced_formula

def is_atom_site_multiplicity_consistent(cif_str):
    # Parse the CIF string
    try:
        parser = CifParser.from_string(cif_str)
    except:
        parser = CifParser.from_str(cif_str)
    cif_data = parser.as_dict()

    # Extract the chemical formula sum from the CIF data
    formula_sum = cif_data[list(cif_data.keys())[0]]["_chemical_formula_sum"]

    # Convert the formula sum into a dictionary
    expected_atoms = Composition(formula_sum).as_dict()

    # Count the atoms provided in the _atom_site_type_symbol section
    actual_atoms = {}
    for key in cif_data:
        if "_atom_site_type_symbol" in cif_data[key] and "_atom_site_symmetry_multiplicity" in cif_data[key]:
            for atom_type, multiplicity in zip(cif_data[key]["_atom_site_type_symbol"],
                                               cif_data[key]["_atom_site_symmetry_multiplicity"]):
                
                if atom_type in actual_atoms:
                    actual_atoms[atom_type] += int(multiplicity)
                else:
                    actual_atoms[atom_type] = int(multiplicity)

    return expected_atoms == actual_atoms

def is_sensible(cif_str, length_lo=0.5, length_hi=1000., angle_lo=10., angle_hi=170.):
    
    try: 
        Structure.from_str(cif_str, fmt="cif")
    except:
        return False
    
    cell_length_pattern = re.compile(r"_cell_length_[abc]\s+([\d\.]+)")
    cell_angle_pattern = re.compile(r"_cell_angle_(alpha|beta|gamma)\s+([\d\.]+)")

    cell_lengths = cell_length_pattern.findall(cif_str)
    for length_str in cell_lengths:
        length = float(length_str)
        if length < length_lo or length > length_hi:
            return False

    cell_angles = cell_angle_pattern.findall(cif_str)
    for _, angle_str in cell_angles:
        angle = float(angle_str)
        if angle < angle_lo or angle > angle_hi:
            return False

    return True

def is_valid(cif_str, bond_length_acceptability_cutoff=1.0):
    if not is_formula_consistent(cif_str):
        return False
    if not is_atom_site_multiplicity_consistent(cif_str):
        return False

    bond_length_score = bond_length_reasonableness_score(cif_str)
    if bond_length_score < bond_length_acceptability_cutoff:
         return False
    if not is_space_group_consistent(cif_str):
        return False
    return True

def evaluate_syntax_validity(cif_str, bond_length_acceptability_cutoff=1.0):
    validity_dict = {
        "formula": False,
        "site_multiplicity": False, 
        "bond_length": False, 
        "spacegroup": False,
    }
    if is_formula_consistent(cif_str):
        validity["formula"] = True
    if is_atom_site_multiplicity_consistent(cif_str):
        validity["site_multiplicity"] = True
    bond_length_score = bond_length_reasonableness_score(cif_str)
    if bond_length_score >= bond_length_acceptability_cutoff:
        validity["bond_length"] = True
    if is_space_group_consistent(cif_str):
        validity["spacegroup"] = True
    
    return validity_dict

def space_group_symbol_to_number(symbol):
    # Dictionary mapping space group symbols to their corresponding numbers
    space_group_mapping = {
        "P1": 1, "P-1": 2, "P2": 3, "P21": 4, "C2": 5, "Pm": 6, "Pc": 7, "Cm": 8, "Cc": 9,
        "P2/m": 10, "P21/m": 11, "C2/m": 12, "P2/c": 13, "P21/c": 14, "C2/c": 15,
        "P222": 16, "P2221": 17, "P21212": 18, "P212121": 19, "C2221": 20, "C222": 21,
        "F222": 22, "I222": 23, "I212121": 24, "Pmm2": 25, "Pmc21": 26, "Pcc2": 27,
        "Pma2": 28, "Pca21": 29, "Pnc2": 30, "Pmn21": 31, "Pba2": 32, "Pna21": 33,
        "Pnn2": 34, "Cmm2": 35, "Cmc21": 36, "Ccc2": 37, "Amm2": 38, "Abm2": 39,
        "Ama2": 40, "Aba2": 41, "Fmm2": 42, "Fdd2": 43, "Imm2": 44, "Iba2": 45,
        "Ima2": 46, "Pmmm": 47, "Pnnn": 48, "Pccm": 49, "Pban": 50, "Pmma": 51,
        "Pnna": 52, "Pmna": 53, "Pcca": 54, "Pbam": 55, "Pccn": 56, "Pbcm": 57,
        "Pnnm": 58, "Pmmn": 59, "Pbcn": 60, "Pbca": 61, "Pnma": 62, "Cmcm": 63,
        "Cmca": 64, "Cmmm": 65, "Cccm": 66, "Cmme": 67, "Ccce": 68, "Fmmm": 69,
        "Fddd": 70, "Immm": 71, "Ibam": 72, "Ibca": 73, "Imma": 74, "P4": 75,
        "P41": 76, "P42": 77, "P43": 78, "I4": 79, "I41": 80, "P-4": 81, "I-4": 82,
        "P4/m": 83, "P42/m": 84, "P4/n": 85, "P42/n": 86, "I4/m": 87, "I41/a": 88,
        "P422": 89, "P4212": 90, "P4122": 91, "P41212": 92, "P4222": 93, "P42212": 94,
        "P4322": 95, "P43212": 96, "I422": 97, "I4122": 98, "P4mm": 99, "P4bm": 100,
        "P42cm": 101, "P42nm": 102, "P4cc": 103, "P4nc": 104, "P42mc": 105, "P42bc": 106,
        "I4mm": 107, "I4cm": 108, "I41md": 109, "I41cd": 110, "P-42m": 111, "P-42c": 112,
        "P-421m": 113, "P-421c": 114, "P-4m2": 115, "P-4c2": 116, "P-4b2": 117, "P-4n2": 118,
        "I-4m2": 119, "I-4c2": 120, "I-42m": 121, "I-42d": 122, "P4/mmm": 123, "P4/mcc": 124,
        "P4/nbm": 125, "P4/nnc": 126, "P4/mbm": 127, "P4/mnc": 128, "P4/nmm": 129, "P4/ncc": 130,
        "I4/mmm": 131, "I4/mcm": 132, "I41/amd": 133, "I41/acd": 134, "P3": 143, "P31": 144,
        "P32": 145, "R3": 146, "P-3": 147, "R-3": 148, "P312": 149, "P321": 150, "P3112": 151,
        "P3121": 152, "P3212": 153, "P3221": 154, "R32": 155, "P3m1": 156, "P31m": 157,
        "P3c1": 158, "P31c": 159, "R3m": 160, "R3c": 161, "P-31m": 162, "P-31c": 163,
        "P-3m1": 164, "P-3c1": 165, "R-3m": 166, "R-3c": 167, "P6": 168, "P61": 169,
        "P65": 170, "P62": 171, "P64": 172, "P63": 173, "P-6": 174, "P6/m": 175,
        "P63/m": 176, "P622": 177, "P6122": 178, "P6522": 179, "P6222": 180, "P6422": 181,
        "P6322": 182, "P6mm": 183, "P6cc": 184, "P63cm": 185, "P63mc": 186, "P-62m": 187,
        "P-62c": 188, "P-6m2": 189, "P-6c2": 190, "P6/mmm": 191, "P6/mcc": 192, "P63/mcm": 193,
        "P63/mmc": 194, "P23": 195, "F23": 196, "I23": 197, "P213": 198, "I213": 199,
        "Pm-3": 200, "Pn-3": 201, "Fm-3": 202, "Fd-3": 203, "Im-3": 204, "Pa-3": 205,
        "Ia-3": 206, "P432": 207, "P4232": 208, "F432": 209, "F4132": 210, "I432": 211,
        "P4332": 212, "P4132": 213, "I4132": 214, "P-43m": 215, "F-43m": 216, "I-43m": 217,
        "P-43n": 218, "F-43c": 219, "I-43d": 220, "Pm-3m": 221, "Pn-3n": 222, "Pm-3n": 223,
        "Pn-3m": 224, "Fm-3m": 225, "Fm-3c": 226, "Fd-3m": 227, "Fd-3c": 228, "Im-3m": 229,
        "Ia-3d": 230
    }
    
    # Convert the space group symbol to the space group number
    return space_group_mapping.get(symbol, None)

def get_metrics(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    metrics = checkpoint['metrics']
    fname = '/'.join(ckpt_path.split('/')[:-1])
    return metrics, fname

def plot_loss_curves(paths, ylog=True, xlog=False, xmin=None, xmax=None, ymin=None, ymax=None, offset=0.02, plot_metrics=True, figsize=(10, 5)):
    # Apply Seaborn style
    sns.set(style="whitegrid")
    #plt.figure(figsize=figsize, dpi=150)
    fig, ax_loss = plt.subplots(figsize=figsize, dpi=150)
    text_positions = []

    ax_loss.set(xlabel='Epoch', ylabel='Cross-Entropy Loss')

    if xmin is not None or xmax is not None:
        ax_loss.set_xlim(xmin, xmax)
    if ymin is not None or ymax is not None:
        ax_loss.set_ylim(ymin, ymax)
    if ylog:
        ax_loss.set_yscale('log')
    if xlog:
        ax_loss.set_xscale('log')
    
    # Seaborn grid
    ax_loss.grid(alpha=0.4, which="both")

    for path in paths:
        metrics, fname = get_metrics(path)
        legend_fname = '/'.join(fname.split("/")[-2:])
        
        # Convert lists to numpy arrays for plotting
        losses_train = np.array(metrics['train_losses'])
        losses_val = np.array(metrics['val_losses'])
        epochs = np.array(metrics['epoch_losses'])
        
        # Train and validation loss plots
        p = sns.lineplot(x=epochs, y=losses_train, label=f'{legend_fname} [Train]', ax=ax_loss, linewidth=2)
        sns.lineplot(x=epochs, y=losses_val, label=f'{legend_fname} [Validation]', ax=ax_loss, linewidth=2, linestyle='--', color=p.get_lines()[-1].get_color())
        
        # Find the minimum value in losses_val and its corresponding epoch
        try:
            val_line_min = epochs[np.argmin(losses_val)]
            min_loss_val = np.min(losses_val)
        
            # Plot the dotted line
            ax_loss.plot([val_line_min, ax_loss.get_xlim()[1]], [min_loss_val, min_loss_val],
                        c=p.get_lines()[-1].get_color(), ls=':', alpha=1.0)

            # Adjust text position if overlapping
            text_x = ax_loss.get_xlim()[1]
            text_y = min_loss_val
        
            vert_align = 'bottom'
            for pos in text_positions:
                if abs(pos[1] - text_y) < offset:  # Check for overlap
                    vert_align = 'top'
                else:
                    vert_align = 'bottom'

            # Add text at the end of the dotted line
            ax_loss.text(text_x, text_y, f'{min_loss_val:.4f}', 
                    verticalalignment=vert_align, horizontalalignment='right', color=p.get_lines()[-1].get_color(),
                    fontsize=10)
            text_positions.append((text_x, text_y))
        except Exception as e:
            print(f"Error plotting validation loss for {legend_fname}: {e}")
            pass
     
    # Add the legend
    ax_loss.legend(fontsize=8)
    
    # Make the layout tight
    fig.tight_layout()

    # Show the plot
    plt.show()

def extract_prompt(sequence, device, add_composition=True, add_spacegroup=False):
    
    tokenizer = Tokenizer()
    cif_start_id = tokenizer.token_to_id["data_"]
    new_line_id = tokenizer.token_to_id["\n"]
    spacegroup_id = tokenizer.token_to_id["_symmetry_space_group_name_H-M"]

    # Find "data_" and slice
    try:
        end_prompt_index = np.argwhere(sequence == cif_start_id)[0][0] + 1
    except IndexError:
        #return 
        raise ValueError(f"'data_' id: {cif_start_id} not found in sequence", tokenizer.decode(sequence))

    cond_ids = torch.tensor(sequence[:end_prompt_index].long())
    cond_ids_len = len(cond_ids) - 1

    # Add composition (and spacegroup)
    if add_composition:
        end_prompt_index += np.argwhere(sequence[end_prompt_index:] == new_line_id)[0][0]

        if add_spacegroup:
            end_prompt_index += np.argwhere(sequence[end_prompt_index:] == spacegroup_id)[0][0]
            end_prompt_index += np.argwhere(sequence[end_prompt_index:] == new_line_id)[0][0]
            
        end_prompt_index += 1
    
    prompt_ids = torch.tensor(sequence[:end_prompt_index].long()).to(device=device)

    return prompt_ids

def extract_prompt_batch(sequences, device, add_composition=True, add_spacegroup=False):
    """
    Extract prompt sequences from a batch of sequences and apply front padding (left-padding).
    """
    tokenizer = Tokenizer()
    cif_start_id = tokenizer.token_to_id["data_"]
    new_line_id = tokenizer.token_to_id["\n"]
    spacegroup_id = tokenizer.token_to_id["_symmetry_space_group_name_H-M"]

    prompts = []
    prompt_lengths = []
    max_len = 0

    # Loop through each sequence in the batch
    for sequence in sequences:
        # Find "data_" and slice
        try:
            end_prompt_index = np.argwhere(sequence == cif_start_id)[0][0] + 1
        except IndexError:
            raise ValueError(f"'data_' id: {cif_start_id} not found in sequence", tokenizer.decode(sequence))

        # Handle composition and spacegroup additions
        if add_composition:
            end_prompt_index += np.argwhere(sequence[end_prompt_index:] == new_line_id)[0][0]
            if add_spacegroup:
                end_prompt_index += np.argwhere(sequence[end_prompt_index:] == spacegroup_id)[0][0]
                end_prompt_index += np.argwhere(sequence[end_prompt_index:] == new_line_id)[0][0]

        # Extract the prompt for this sequence
        prompt_ids = torch.tensor(sequence[:end_prompt_index + 1].long()).to(device)
        prompts.append(prompt_ids)
        prompt_lengths.append(len(prompt_ids))
        max_len = max(max_len, len(prompt_ids))

    # Apply left-padding: pad from the front to ensure all prompts are the same length
    padded_prompts = torch.full((len(sequences), max_len), tokenizer.padding_id, dtype=torch.long).to(device)
    for i, prompt in enumerate(prompts):
        prompt_len = len(prompt)
        # Left-pad by inserting the prompt at the end of the padded sequence
        padded_prompts[i, -prompt_len:] = prompt

    return padded_prompts, prompt_lengths


# Function to load model from a checkpoint
def load_model_from_checkpoint(ckpt_path, device):
    
    # Checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)  # Load checkpoint
    state_dict = checkpoint["best_model"]
    model_args = checkpoint["model_args"]

    # TEMP TODO
    if "condition_with_emb" in model_args:
        model_args['condition_with_mlp_emb'] = model_args.pop('condition_with_emb')
    
    # Load the model and checkpoint
    model_config = DeciferConfig(**model_args)
    model = Decifer(model_config).to(device)
    model.device = device
    
    # Fix the keys of the state dict per CrystaLLM
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    #model.load_state_dict(state_dict)  # Load modified state_dict into the model
    model.load_state_dict(state_dict)
    return model
