import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from glob import glob
import os
import warnings

# from decifer import (
#     Tokenizer,
#     DeciferConfig,
#     UnconditionedDecifer,
# )

warnings.filterwarnings('ignore')

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

def plot_loss_curves(paths, ylog=True, xlog=False, xmin=None, xmax=None, offset = 0.02, plot_metrics=True):
    fig, ax_loss = plt.subplots(figsize=(10,5), dpi=150)
    text_positions = []

    ax_loss.set(xlabel='Epoch', ylabel='CE-Loss')

    if xmin is not None or xmax is not None:
        ax_loss.set_xlim(xmin, xmax)
    if ylog:
        ax_loss.set_yscale('log')
    if xlog:
        ax_loss.set_xscale('log')
    ax_loss.grid(alpha=0.2, which="both")

    for path in paths:
        metrics, fname = get_metrics(path)
        legend_fname = '/'.join(fname.split("/")[-2:])
        losses_train, losses_val, epochs = metrics['train_losses'], metrics['val_losses'], metrics['epoch_losses']
        
        # Losses
        p = ax_loss.plot(epochs, losses_train, label=legend_fname + f' [{losses_val[-1].item():1.3f}]')
        ax_loss.plot(epochs, losses_val, c=p[0].get_color(), ls='--')
        
        # Find the minimum value in losses_val and its corresponding epoch
        try:
            val_line_min = epochs[np.argmin(losses_val)].item()
            min_loss_val = torch.min(losses_val).item()
        
            # Plot the dotted line
            ax_loss.plot([val_line_min, ax_loss.get_xlim()[1]], [min_loss_val, min_loss_val],
                        c=p[0].get_color(), ls=':', alpha=1.0)

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
                    verticalalignment=vert_align, horizontalalignment='right', color=p[0].get_color(),
                    fontsize=10)
            text_positions.append((text_x, text_y))
        except:
            pass
     
    ax_loss.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    

def extract_prompt(sequence, device, add_composition=True, add_spacegroup=False):
    from decifer import Tokenizer
    
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
    
    prompt_ids = torch.tensor(sequence[:end_prompt_index+1].long()).to(device=device)

    return prompt_ids

# Function to load model from a checkpoint
def load_model_from_checkpoint(ckpt_path, device):
    import torch
    from decifer import DeciferConfig, UnconditionedDecifer  # Local import here
    
    # Checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)  # Load checkpoint
    state_dict = checkpoint["best_model"]
    model_args = checkpoint["model_args"]
    
    # Load the model and checkpoint
    model_config = DeciferConfig(**model_args)
    model = UnconditionedDecifer(model_config).to(device)
    model.device = device
    
    # Fix the keys of the state dict per CrystaLLM
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    #model.load_state_dict(state_dict)  # Load modified state_dict into the model
    model.load_state_dict(state_dict)
    return model