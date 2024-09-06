import math
import re
import pandas as pd
import numpy as np

from pymatgen.core import Composition
from pymatgen.io.cif import CifBlock
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.core.operations import SymmOp

def replace_symmetry_loop(cif_str):
    start = cif_str.find("_symmetry_equiv_pos_site_id")
    end = cif_str.find("loop_", start+1)

    replacement = """_symmetry_equiv_pos_site_id\n_symmetry_equiv_pos_as_xyz\n1  'x, y, z'\n"""

    return cif_str.replace(cif_str[start:end], replacement)

def remove_cif_header(cif_str):
    lines = cif_str.split('\n')
    cif_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 0 and not line.startswith("#") and "pymatgen" not in line:
            cif_lines.append(line)

    cif_str = '\n'.join(cif_lines)
    return cif_str

OXI_LOOP_PATTERN = r'loop_[^l]*(?:l(?!oop_)[^l]*)*_atom_type_oxidation_number[^l]*(?:l(?!oop_)[^l]*)*'
OXI_STATE_PATTERN = r'(\n\s+)([A-Za-z]+)[\d.+-]*'

# Regular expression to match occupancy values
OCCU_PATTERN = re.compile(
        r'(_atom_site_type_symbol[\s\S]+?_atom_site_occupancy\s+)((?:\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\d+\.\d+\s*)+)', re.MULTILINE)

def remove_oxidation_loop(cif_str):
    cif_str = re.sub(OXI_LOOP_PATTERN, '', cif_str)
    cif_str = re.sub(OXI_STATE_PATTERN, r'\1\2', cif_str)
    return cif_str

def format_occupancies(cif_str, decimal_places=4):

    # Search for the occupancy block
    occupancy_block_match = OCCU_PATTERN.search(cif_str)
    
    if occupancy_block_match:
        # Extract header and atoms
        header = occupancy_block_match.group(1)
        atoms = occupancy_block_match.group(2).strip().split('\n')
        
        # Check occupancies and format them
        formatted_atoms = []
        occupancies = []
        for atom in atoms:
            parts = atom.split()
            occupancy = float(parts[-1])
            occupancies.append(occupancy)
            formatted_occupancy = "{:.{}f}".format(occupancy,decimal_places)
            formatted_atoms.append('  '.join(parts[:-1]) + f"  {formatted_occupancy}")

        formatted_block = header + '\n  '.join(formatted_atoms)
        cif_str = OCCU_PATTERN.sub(formatted_block, cif_str)

    return cif_str

def extract_formula_units(cif_str):
    return extract_numeric_property(cif_str, "_cell_formula_units_Z", numeric_type=int)

def extract_space_group_symbol(cif_str):
    match = re.search(r"_symmetry_space_group_name_H-M\s+('([^']+)'|(\S+))", cif_str)
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract space group from:\n{cif_str}")

def extract_numeric_property(cif_str, prop, numeric_type=float):
    match = re.search(rf"{prop}\s+([.0-9]+)", cif_str)
    if match:
        return numeric_type(match.group(1))
    raise Exception(f"could not find {prop} in:\n{cif_str}")

def extract_data_formula(cif_str):
    #match = re.search(r"data_([A-Za-z0-9]+)\n", cif_str)
    match = re.search(r"data_([A-Za-z0-9.()]+)\n", cif_str) # Including paranthesis
    if match:
        return match.group(1)
    raise Exception(f"could not find data_ in:\n{cif_str}")

def extract_formula_nonreduced(cif_str):
    match = re.search(r"_chemical_formula_sum\s+('([^']+)'|(\S+))", cif_str)
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract _chemical_formula_sum value from:\n{cif_str}")

def replace_data_formula_with_nonreduced_formula(cif_str):
    pattern = r"_chemical_formula_sum\s+(.+)\n"
    pattern_2 = r"(data_)(.*?)(\n)"
    match = re.search(pattern, cif_str)
    if match:
        chemical_formula = match.group(1)
        chemical_formula = chemical_formula.replace("'", "").replace(" ", "")

        modified_cif = re.sub(pattern_2, r'\1' + chemical_formula + r'\3', cif_str)

        return modified_cif
    else:
        raise Exception(f"Chemical formula not found {cif_str}")

def round_numbers(cif_str, decimal_places=4):
    # Pattern to match a floating point number in the CIF file
    # It also matches numbers in scientific notation
    # pattern = r"[-+]?\d*\.\d+([eE][-+]?\d+)?"
    pattern = r"[-+]?\d*\.?\d+([eE][-+]?\d+)?" # Including occupancies

    # Function to round the numbers
    def round_number(match):
        number_str = match.group()
        number = float(number_str)
        # Check if number of digits after decimal point is less than 'decimal_places'
        if len(number_str.split('.')[-1]) <= decimal_places:
            return number_str
        rounded = round(number, decimal_places)
        return format(rounded, '.{}f'.format(decimal_places))

    # Replace all occurrences of the pattern using a regex sub operation
    cif_string_rounded = re.sub(pattern, round_number, cif_str)

    return cif_string_rounded

