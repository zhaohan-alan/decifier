import math
import re
import pandas as pd

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
