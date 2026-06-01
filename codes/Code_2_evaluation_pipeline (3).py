from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import itertools
import json
import math
import os
import subprocess
import sys
import re 
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

try:
    from scipy.spatial.transform import Rotation
except Exception:  # pragma: no cover
    Rotation = None

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None

# ==========================================================
# SETTINGS
# ==========================================================
# Define constants, paths, and configuration settings for the evaluation pipeline.
# ==========================================================

# Defines for directory paths and file naming conventions
BASE_DIR = Path(__file__).resolve().parent
STRUCT_ROOT = BASE_DIR / "structures"
RESULTS_DIR = BASE_DIR / "results"
ALIGNED_ROOT = RESULTS_DIR / "aligned_xyz"
OUT_DIR = BASE_DIR / "combined_rmsd_analysis"

# Subdirectories for each model's structures
STRUCT_DIRS = {
    "gemini": STRUCT_ROOT / "gemini",
    "openai": STRUCT_ROOT / "openai",
    "claude": STRUCT_ROOT / "claude",
    "pubchem": STRUCT_ROOT / "pubchem",
}

# Evaluation parameters and settings
MODELS = ["gemini", "openai", "claude"]
MODEL_ORDER = ["claude", "gemini", "openai"]
MAX_WORKERS = min(8, os.cpu_count() or 4)
USE_REFLECTIONS = True
FERMI_MIDPOINT = 1.0
FERMI_K = 0.2
THRESHOLDS = [0.25, 0.5, 1.0, 2.0]
HIST_BIN_WIDTH = 0.2
HIST_XMAX = 5.6
ZOOM_XMAX = 2.0
AUTO_LAUNCH_VIEWER = True
VIEWER_FILENAME = "Code_3_molecule_viewer.py"

# Create necessary directories if they don't exist
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(ALIGNED_ROOT / "gemini").mkdir(parents=True, exist_ok=True)
(ALIGNED_ROOT / "openai").mkdir(parents=True, exist_ok=True)
(ALIGNED_ROOT / "claude").mkdir(parents=True, exist_ok=True)
(ALIGNED_ROOT / "pubchem_reference").mkdir(parents=True, exist_ok=True)

# Define paths for intermediate and output files
MAPPING_CSV = RESULTS_DIR / "atom_mapping_table.csv"
RESULTS_CSV = RESULTS_DIR / "rmsd_results_parallel.csv"
OUTPUT_XLSX = RESULTS_DIR / "all_models_vs_pubchem.xlsx"

# Covalent radii for common elements, used for bond inference
COVALENT_RADII = {
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}

# Mapping of error codes to human-readable labels for classification of evaluation errors.
ERROR_CODE_MAP = {
    0: "ok",
    1: "mismatched_atom_count",
    2: "element_mismatch",
    3: "atom_mapping_failed",
    4: "kabsch_failed",
    5: "failure_marker_from_generation",
    6: "invalid_xyz",
    7: "missing_pubchem_reference",
    8: "unknown_evaluation_error",
    9: "json_parse_failed",
    10: "no_json_found",
    11: "api_or_network_error",
}

# Convertes error message into error code and label based on known patterns and status indicators.
def classify_evaluation_error(message: str | None, status: str | None = None):
    msg = (message or "").strip()
    lowered = msg.lower()
    status = (status or "").strip().lower()

    # Status-level classifications. These are only used when the status itself
    # is more important than the text message, e.g. a missing PubChem reference.
    if status == "ok":
        return 0, ERROR_CODE_MAP[0]
    if status == "missing_pubchem":
        return 7, ERROR_CODE_MAP[7]

    # Message-level classifications. These allow Code 2 to recover the actual
    # reason stored inside a Code 1 failure marker, e.g. atom count mismatch.
    if "atom count mismatch" in lowered:
        return 1, ERROR_CODE_MAP[1]
    if "element mismatch" in lowered or "element count mismatch" in lowered:
        return 2, ERROR_CODE_MAP[2]
    if "no json found" in lowered:
        return 10, ERROR_CODE_MAP[10]
    if (
        "json" in lowered
        or "expecting" in lowered
        or "unterminated" in lowered
        or "extra data" in lowered
        or "invalid control character" in lowered
    ):
        return 9, ERROR_CODE_MAP[9]
    if (
        "timeout" in lowered
        or "timed out" in lowered
        or "connection" in lowered
        or "rate limit" in lowered
        or "authentication" in lowered
        or "api" in lowered
        or "status=" in lowered
    ):
        return 11, ERROR_CODE_MAP[11]
    if "rdkit could not produce a full atom mapping" in lowered or "hungarian" in lowered or "mapped atom order" in lowered or "mapping" in lowered:
        return 3, ERROR_CODE_MAP[3]
    if "kabsch" in lowered or "align_vectors" in lowered or "alignment failed" in lowered:
        return 4, ERROR_CODE_MAP[4]
    if "xyz" in lowered or "malformed xyz" in lowered or "invalid xyz" in lowered:
        return 6, ERROR_CODE_MAP[6]

    # If it is a marker but the message does not match a known pattern,
    # keep it as a generation-marker failure.
    if status == "not_found" or lowered.startswith("failed") or "error_code=" in lowered:
        return 5, ERROR_CODE_MAP[5]
    return 8, ERROR_CODE_MAP[8]

# Extracts failure marker details from code 1
def parse_failure_marker_details(note: str):
    details = {}
    for line in (note or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        details[key.strip()] = value.strip()
    return details

# Normalizes error messages by stripping whitespace, collapsing multiple spaces, and truncating to a reasonable length for display.
def normalize_error_message(message: str | None) -> str:
    msg = str(message or "").strip()
    if not msg:
        return ""
    msg = re.sub(r"\s+", " ", msg)
    return msg[:160]

# Created readable label for error combining the error code, type and message.
def build_error_display_label(code, label, message) -> str:
    code_text = "?" if pd.isna(code) else str(int(code))
    label_text = str(label or "unknown_error").strip()
    message_text = normalize_error_message(message)
    if message_text:
        return f"{code_text}: {label_text} | {message_text}"
    return f"{code_text}: {label_text}"

# ==========================================================
# HELPERS
# ==========================================================
# Helper functions for reading XYZ files, inferring bonds, 
# computing signatures, performing atom mapping, aligning structures, and classifying errors.
# ==========================================================

# Cleans the value collumbs keeping only numbers, checks if >1 values are left, and calculatede the SEM (standart error mean)
def safe_sem(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(values) <= 1:
        return np.nan
    return values.std(ddof=1) / np.sqrt(len(values))


# Convertes RMSD value to a score between 0 and 1 using a Fermi-Dirac-like function. K determinds the slope.
def rmsd_to_score(value: float, midpoint: float = FERMI_MIDPOINT, k: float = FERMI_K):
    if value is None or pd.isna(value): # Checks if none or NaN yielding score of 0
        return 0.0
    return float(1.0 / (1.0 + math.exp((float(value) - float(midpoint)) / float(k))))


# Extracts the base stem of a file path by removing a specified suffix if it is present, otherwise returns the full stem.
def safe_stem_base(path: Path, suffix: str) -> str:
    stem = path.stem # remove extension
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem # Remove suffix if present, otherwise return full stem


# Find all XYZ structues, group them by molecule name, and organize paths by source
def scan_structure_files() -> Dict[str, Dict[str, Path]]:
    files: Dict[str, Dict[str, Path]] = defaultdict(dict) # creates empty dict atumatically

    # Scan each model's directory for XYZ files, extract the molecule name by removing the model-specific suffix, 
    # and store the paths in a nested dictionary structure.
    for path in STRUCT_DIRS["pubchem"].glob("*_pubchem.xyz"):
        files[safe_stem_base(path, "_pubchem")]["pubchem"] = path
    for path in STRUCT_DIRS["gemini"].glob("*_gemini.xyz"):
        files[safe_stem_base(path, "_gemini")]["gemini"] = path
    for path in STRUCT_DIRS["openai"].glob("*_openai.xyz"):
        files[safe_stem_base(path, "_openai")]["openai"] = path
    for path in STRUCT_DIRS["claude"].glob("*_claude.xyz"):
        files[safe_stem_base(path, "_claude")]["claude"] = path

    return dict(sorted(files.items()))


# Extracts the atom symbol adn coordinates from the xyz. files
def read_xyz(path: Path) -> Tuple[List[str], np.ndarray]: 
    lines = path.read_text(encoding="utf-8").strip().splitlines() # removes spaces and split lines
    
    # Checks if file has at least 3 lines (atom count, comment, and at least one atom) and returns error if not.
    if len(lines) < 3:
        raise ValueError(f"Invalid XYZ file: {path}")

    try:
        n_atoms = int(lines[0].strip()) # Number of atoms should be in the first line, if not error is raised
    except ValueError as exc:
        raise ValueError(f"XYZ atom count missing in {path}") from exc

    body = lines[2:2 + n_atoms] # extract only atom coordinates
    if len(body) != n_atoms: # checks if expected atom count and actual lines with coordinates match, if not error is raised
        raise ValueError(f"XYZ atom count mismatch in {path}")

    # Empty arrays
    symbols: List[str] = []
    coords: List[List[float]] = []

    # 
    for line in body:
        parts = line.split() # split line into parts, expecting symbol and 3 coordinates
        if len(parts) < 4: # check if there are at least 4 parts (symbol + x, y, z), if not error is raised
            raise ValueError(f"Malformed XYZ line in {path}: {line}")
        symbols.append(parts[0]) # save symbol
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])]) # save coordinates as floats

    return symbols, np.asarray(coords, dtype=float) # return symbols and coordinates as numpy array

# checks if  a file is a failure marker "FAILED" and extracts the message if it is, otherwise returns False and empty string.
def is_failure_marker(path: Path) -> tuple[bool, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip() # read file content, ignoring encoding errors, and strip whitespace
    except Exception as exc:
        return False, f"could_not_read_marker: {exc}"
    if text.startswith("FAILED"):
        return True, text
    return False, ""


# Converts atom symbols and coordinates into XYZ file format text
def xyz_to_text(symbols: List[str], coords: np.ndarray, comment: str = "") -> str:
    lines = [str(len(symbols)), comment] # First line is atom count, second line is comment
    for sym, (x, y, z) in zip(symbols, coords): # Each subsequent line contains the symbol and coordinates formatted to 8 decimal places
        lines.append(f"{sym:2s} {x: .8f} {y: .8f} {z: .8f}")
    return "\n".join(lines) + "\n" # Join all lines into a single string with newline characters, and add a final newline at the end


# Groups atom indexes by element type, ex. [O, H, H,] -> H: [1, 2], O: [0]
def grouped_indices(symbols: List[str]) -> Dict[str, List[int]]: 
    out: Dict[str, List[int]] = defaultdict(list) 
    for idx, sym in enumerate(symbols):
        out[sym].append(idx)
    return dict(out)


# Checks if Pubchem and LLm strucute have the same atom count and element types. If tnot, mismatch error. 
# If they match, returns the grouped indices by element type for both structures.
def validate_element_counts(ref_symbols: List[str], model_symbols: List[str]):
    if len(ref_symbols) != len(model_symbols): # count mismatch
        raise ValueError("Atom count mismatch")
    ref_groups = grouped_indices(ref_symbols)
    model_groups = grouped_indices(model_symbols)
    if set(ref_groups) != set(model_groups): # element type mismatch
        raise ValueError("Element mismatch")
    for sym in ref_groups:
        if len(ref_groups[sym]) != len(model_groups.get(sym, [])): # element count mismatch
            raise ValueError(f"Element count mismatch for {sym}")
    return ref_groups, model_groups


# Guesses bonds based on covalent radii and interatomic distances, returning a set of bonded atom index pairs.
def infer_bonds(symbols: List[str], coords: np.ndarray, scale: float = 1.25) -> set[Tuple[int, int]]:
    bonds: set[Tuple[int, int]] = set() # Initialize an empty set to store bonded atom index pairs
    n = len(symbols) 

    # Compare every atom
    for i in range(n):
        for j in range(i + 1, n):

            # Get covalent radii for each atom
            ri = COVALENT_RADII.get(symbols[i], 0.8)
            rj = COVALENT_RADII.get(symbols[j], 0.8)

            # Calculate the distance cutoff for bonding based on the sum of covalent radii multiplied by a scale factor
            cutoff = scale * (ri + rj)
            d = float(np.linalg.norm(coords[i] - coords[j])) # calc structure distance

            # If distance smaller than cutoff, create bond
            if d <= cutoff:
                bonds.add((i, j))
    return bonds


# Using "bonds" from above funtion, it creates an adjacency list 
# representation of the molecule where each atom index maps to a list of neighboring atom indices.
def adjacency_from_bonds(n: int, bonds: set[Tuple[int, int]]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    return adj


# Retuns tumple of number of neighbors for the atom, and symbol of neighbors sorted alphabetically.
def local_env_signature(idx: int, symbols: List[str], adj: List[List[int]]) -> Tuple[int, Tuple[str, ...]]:
    nbr_syms = sorted(symbols[j] for j in adj[idx]) # find the atoms neighbour and extracts its symbol, sorted alphabetically
    return len(adj[idx]), tuple(nbr_syms) 

# Calculates a penalty when two atoms have different local bonding environments
def env_penalty(ref_sig, model_sig) -> float:

    # degree = bonded neighbours, nbrs = type of neighbour
    ref_degree, ref_nbrs = ref_sig 
    model_degree, model_nbrs = model_sig
    penalty = 6.0 * abs(ref_degree - model_degree) # Penalty from dree of ref and LLM structure

    # Checks atom type
    if ref_nbrs != model_nbrs: 

        # Coutn negibour types
        ref_counts = defaultdict(int)
        model_counts = defaultdict(int)
        for sym in ref_nbrs:
            ref_counts[sym] += 1
        for sym in model_nbrs:
            model_counts[sym] += 1
            
        all_syms = set(ref_counts) | set(model_counts) # combinded set of element types
        penalty += 4.0 * sum(abs(ref_counts[s] - model_counts[s]) for s in all_syms) # Coutn difference i neighbour types and adds penealty
    return float(penalty)


# Computes teh distance from every atom to the other using the xyz coordinates
def sorted_distance_signature(coords: np.ndarray, idx: int, take: int | None = None) -> np.ndarray:
    d = np.linalg.norm(coords - coords[idx], axis=1) # calc distance
    d = np.sort(np.delete(d, idx)) # removes self distance
    if take is not None:
        d = d[:take]
    return d


# compares atoms from PubChem ref to LLM strucute based on location, returining a penalty score based 
# on the difference in sorted distance signatures and optionally the same-element neighbor distances.
def signature_penalty(ref_coords, model_coords, ref_idx, model_idx, ref_symbols=None, model_symbols=None) -> float:
    ref_all = sorted_distance_signature(ref_coords, ref_idx, take=8) # gets distance from pubhchem atom to nearest atom
    model_all = sorted_distance_signature(model_coords, model_idx, take=8) # get distance form LLma tom to nerast atom
    n = min(len(ref_all), len(model_all)) # take up to 8 nearest atoms, but if there are less than 8 atoms in total, take all of them
    total = float(np.mean(np.abs(ref_all[:n] - model_all[:n]))) if n > 0 else 0.0 # compares distance patterns

    if ref_symbols is not None and model_symbols is not None: # if symbols provided

        # Finds distance form one atom type to others of same type for pubchem
        ref_same = np.sort([ 
            float(np.linalg.norm(ref_coords[k] - ref_coords[ref_idx]))
            for k, sym in enumerate(ref_symbols)
            if k != ref_idx and sym == ref_symbols[ref_idx]
        ])[:6]

         # Finds distance form one atom type to others of same type for LLM
        model_same = np.sort([
            float(np.linalg.norm(model_coords[k] - model_coords[model_idx]))
            for k, sym in enumerate(model_symbols)
            if k != model_idx and sym == model_symbols[model_idx]
        ])[:6]

        m = min(len(ref_same), len(model_same)) # finds count of same elements for comparison
        if m > 0:
            # adds extra penenlty points if same element patterns are different
            total += 1.5 * float(np.mean(np.abs(ref_same[:m] - model_same[:m]))) 
    return total



# 
def build_rdkit_mol_from_inferred_bonds(symbols: List[str], coords: np.ndarray):
    if Chem is None: # if RDkit not available return nothing
        return None
    rw = Chem.RWMol() # empty RDkit molecule (reat wrote molecule) to which we will add atoms and bonds

    # add each atoms to the molecule based on the symbols
    for sym in symbols: 
        rw.AddAtom(Chem.Atom(sym))

    # Use function infer_bonds to guess bonds based on distance and covalent radii, and add them to the RDKit molecule
    for i, j in infer_bonds(symbols, coords):
        rw.AddBond(int(i), int(j), Chem.BondType.SINGLE) # add single bond between atoms i and j
    mol = rw.GetMol() # Convert molecule from editable to standard RDKit molecule
    try:
        Chem.SanitizeMol(mol) # Check if molecule makes chemical sence based on RDkit rules
    except Exception: # Dosnt stop due to complains from RDKit
        pass
    return mol


# Uses RDKit to find possible atom mappings between PubChem and model structures
def rdkit_substructure_permutations(ref_symbols, ref_coords, model_symbols, model_coords):
    if Chem is None: # empty list ift RDKit not available
        return []
    
    ref_mol = build_rdkit_mol_from_inferred_bonds(ref_symbols, ref_coords) # build RDKit molecule for PubChem reference
    model_mol = build_rdkit_mol_from_inferred_bonds(model_symbols, model_coords) # build RDKit molecule for LLM structure

    if ref_mol is None or model_mol is None:
        return [] # If not moelcules return empty list
    try:
        matches = list(model_mol.GetSubstructMatches(ref_mol, uniquify=False)) # Matches of Pubvhem pattersn seen in LLM molecules
    except Exception: # dosnt crash if fail
        matches = []
    perms = []

    for match in matches: 
        if len(match) != len(ref_symbols): # Use only molecules with same atom count
            continue
        perm = np.asarray(match, dtype=int) # Converts matches to numpy arrays
        if [model_symbols[i] for i in perm] == ref_symbols: # Checks if the matched model atoms has same order as Pubchem
            perms.append(perm) # Store structure
    return perms


# Selects the best RDKit atom mapping by choosing the mapping with the lowest coordinate difference
def rdkit_atom_mapping(ref_symbols, ref_coords, model_symbols, model_coords):
    perms = rdkit_substructure_permutations(ref_symbols, ref_coords, model_symbols, model_coords)
    if not perms: # if no valid mappings found, raise error
        raise ValueError("RDKit could not produce a full atom mapping")

    best_perm = None
    best_cost = None
    ref_centered = ref_coords - ref_coords.mean(axis=0) # center Pubchem coordinates around 0

    for perm in perms: # for each mapping
        reordered = model_coords[perm] # reorder LLM coordinates correct based on mapping
        reordered_centered = reordered - reordered.mean(axis=0) # center reordered coordinates around 0
        cost = float(np.mean(np.linalg.norm(reordered_centered - ref_centered, axis=1))) # calc distance between the centerede strucutres
        
        # Finds the best mapping with lowest distance
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm

    # raise error if no valid mapping found, otherwise return the best mapping and source label
    if best_perm is None:
        raise ValueError("RDKit returned no valid atom permutation")
    return best_perm, "rdkit" 

# Backup atom mapping method. Matches atoms by element type, distance pattern,
# and local bonding environment using the Hungarian algorithm.
def hungarian_atom_mapping(ref_symbols, ref_coords, model_symbols, model_coords):
    if linear_sum_assignment is None: # Check if SciPy linear_sum_assignment is available, if not raise error
        raise ValueError("SciPy linear_sum_assignment is not available")

    
    # Check if atom counts and element types match, and get grouped indices by element type for both structures.
    # If not, raise error.
    ref_groups, model_groups = validate_element_counts(ref_symbols, model_symbols)

    # Infers bonds for both molecules and creates neighvour lists
    ref_adj = adjacency_from_bonds(len(ref_symbols), infer_bonds(ref_symbols, ref_coords))
    model_adj = adjacency_from_bonds(len(model_symbols), infer_bonds(model_symbols, model_coords))

    # Creates local enviorment signetures for each atom
    ref_sigs = [local_env_signature(i, ref_symbols, ref_adj) for i in range(len(ref_symbols))]
    model_sigs = [local_env_signature(i, model_symbols, model_adj) for i in range(len(model_symbols))]

    perm = np.full(len(ref_symbols), -1, dtype=int) # Empty mapping list (-1 means "no yet assigned")

    # Center cooridnates
    ref_centered = ref_coords - ref_coords.mean(axis=0)  
    model_centered = model_coords - model_coords.mean(axis=0)

    # For each element type
    for sym in sorted(ref_groups):
        ref_idx = ref_groups[sym] # atom index for PubChem structure
        model_idx = model_groups[sym] # atom index for LLM structure

        # Cost matrix for Hungarian algorithm, where cost[i, j] is the penalty for mapping ref_idx[i] to model_idx[j]
        cost = np.zeros((len(ref_idx), len(model_idx)), dtype=float) 
        for i, r_idx in enumerate(ref_idx):
            for j, m_idx in enumerate(model_idx):

                # cost of matching LLM model with pubchem based on distance pattern and local enviorment
                cost[i, j] = (
                    signature_penalty(ref_centered, model_centered, r_idx, m_idx, ref_symbols, model_symbols)
                    + env_penalty(ref_sigs[r_idx], model_sigs[m_idx])
                )

        row_ind, col_ind = linear_sum_assignment(cost) # choses the best mappin with lowest cost
        for i, j in zip(row_ind, col_ind): # stores matches
            perm[ref_idx[i]] = model_idx[j]

    if np.any(perm < 0): # Checks if mapping was assigned
        raise ValueError("Hungarian mapping failed to assign all atoms")
    if [model_symbols[i] for i in perm.tolist()] != ref_symbols: # Checks element order preservation
        raise ValueError("Hungarian permutation did not preserve element order")
    return perm, "hungarian"


# allignes the LLM structure to PubChem reference using the Kabsch algorithm implemented in SciPy, 
# allowing for reflections.
# Mobile is the movable stucture and target is the reference structure
def align_with_library_kabsch(mobile: np.ndarray, target: np.ndarray, allow_reflection: bool = True):
    if Rotation is None: # chechs if rotaiton available in SciPy, if not raise error
        raise ValueError("SciPy Rotation.align_vectors is not available")

    # Convert to numpy arrays and check shapes
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if mobile.shape != target.shape: # Checsk for same xyz molecule shape, if not raise error
        raise ValueError("Shape mismatch between mobile and target coordinates")

    # Center both structures around their centroids
    mobile_centroid = mobile.mean(axis=0)
    target_centroid = target.mean(axis=0)
    mobile_centered = mobile - mobile_centroid
    target_centered = target - target_centroid

    # No rotation for single-atom structures, just translation
    if len(mobile) == 1:
        aligned = mobile - mobile_centroid + target_centroid
        return aligned, np.eye(3), target_centroid - mobile_centroid, False

    reflection_options = [np.array([1.0, 1.0, 1.0])]

    # Check both non-reflected and reflected versions of the mobile structure to find the best alignment
    if allow_reflection:
        reflection_options = [np.array(v, dtype=float) for v in itertools.product([-1.0, 1.0], repeat=3)]

    best = None

    # For mirrored and non mirrored
    for refl in reflection_options:
        reflected = mobile_centered * refl #Applies reflection
        try:
            rotation_obj, _ = Rotation.align_vectors(target_centered, reflected) # Finds best rotatation for alignment using Kabsch algorithm implemented in SciPy
        except Exception:
            continue
        rotation = rotation_obj.as_matrix()

        aligned_centered = rotation_obj.apply(reflected) # applies rotation
        aligned = aligned_centered + target_centroid # moces aligned structure back to original location of PubChem reference
        current_rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1)))) # calculates RMSD of the aligned structure to the target
        used_reflection = bool(np.any(refl < 0)) # checks if reflection was used in this alignment
        candidate = (aligned, rotation, target_centroid - mobile_centroid, used_reflection, current_rmsd) #Store allignment candidate details
        
        # Keep alligment with lowest RMSD
        if best is None or current_rmsd < best[-1]: 
            best = candidate

    #If no valid alignment found, raise error, otherwise return the best aligned coordinates, rotation matrix, translation vector, and whether reflection was used.
    if best is None:
        raise ValueError("Library Kabsch alignment failed")
    return best[0], best[1], best[2], best[3]


# Computes RMSD (Root mean square deviation) between two aligned structures
def compute_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((np.asarray(a) - np.asarray(b)) ** 2, axis=1))))


# Evaluates one model structure against its PubChem reference:
# checks files, reads XYZ, maps atoms, aligns structures, computes RMSD/score,
# saves aligned files, and returns a result record.
def process_one_pair(task):
    # define task 
    molecule, model, ref_path, model_path, aligned_root = task
    
    # Creates results dict, assumed unsucessfull
    record = {
        "molecule": molecule,
        "model": model,
        "atom_count": np.nan,
        "rmsd": np.nan,
        "score": 0.0,
        "status": "unknown",
        "note": None,
        "mapping_source": None,
        "used_reflection": False,
        "aligned_model_file": None,
        "pubchem_file": str(ref_path),
        "model_file": str(model_path),
        "valid": False,
        "permutation_ref_to_model": None,
        "error_code": "",
        "error_label": "",
        "error_message": "",
    }

    try:
        ref_failed, ref_note = is_failure_marker(ref_path) #Checks if Pubchem reference file is a failure marker.
        
        # return error record if PubChem reference is missing or marked as failure.
        if ref_failed: 
            error_code, error_label = classify_evaluation_error(ref_note, status="missing_pubchem")
            record.update({
                "status": "missing_pubchem",
                "note": ref_note,
                "score": 0.0,
                "error_code": error_code,
                "error_label": error_label,
                "error_message": ref_note,
            })
            return record

        # Check if the model file failed in Code 1 generation, and if so, extract the failure details 
        # and return an error record without attempting mapping or alignment.
        model_failed, model_note = is_failure_marker(model_path)
        if model_failed:
            try:
                ref_symbols, _ = read_xyz(ref_path)
                record["atom_count"] = len(ref_symbols)
            except Exception:
                pass
            marker_details = parse_failure_marker_details(model_note)
            marker_error_code = marker_details.get("error_code")
            marker_error_label = marker_details.get("error_label")
            error_message = marker_details.get("message", model_note)

            # IMPORTANT:
            # A Code 1 failure marker starts with "FAILED", but the real reason
            # is stored in message=... . Classify that message first, so an atom
            # count mismatch from Code 1 becomes error code 1 instead of always
            # being collapsed into code 5: failure_marker_from_generation.
            inferred_code, inferred_label = classify_evaluation_error(error_message, status="not_found")

            record.update({
                "status": "generation_failed",
                "note": model_note,
                "score": 0.0,
                "error_code": int(marker_error_code) if str(marker_error_code).strip().isdigit() else inferred_code,
                "error_label": marker_error_label or inferred_label,
                "error_message": error_message,
            })
            return record

        ref_symbols, ref_coords = read_xyz(ref_path) # read PubChem reference structure
        model_symbols, model_coords = read_xyz(model_path) # read LLM structure
        record["atom_count"] = len(ref_symbols) # store atom count from PubChem reference
        validate_element_counts(ref_symbols, model_symbols) # validate that atom counts and element types match between PubChem and LLM structures, raise error if not

        # Tries RDitk mapping first, and if it fails, falls back to the Hungarian algorithm mapping. Raise error if failure in both methods.
        try:
            perm, mapping_source = rdkit_atom_mapping(ref_symbols, ref_coords, model_symbols, model_coords)
        except Exception:
            perm, mapping_source = hungarian_atom_mapping(ref_symbols, ref_coords, model_symbols, model_coords)

        # Rorders model atom symbols using mapping and checks if the order matches PubChem reference, raise error if not
        reordered_symbols = [model_symbols[idx] for idx in perm.tolist()]
        if reordered_symbols != ref_symbols:
            raise ValueError("Mapped atom order does not match PubChem symbol order")

        reordered_coords = model_coords[perm] # Reorders model coordinates into pubchem order using the mapping

        # Create alligned coordinates using Kabasch
        aligned_coords, _, _, used_reflection = align_with_library_kabsch(
            reordered_coords,
            ref_coords,
            allow_reflection=USE_REFLECTIONS,
        )
        kabsch_rmsd = compute_rmsd(aligned_coords, ref_coords) # Calc kabsch RMSD
        score = rmsd_to_score(kabsch_rmsd) # Converte RMSD into score

        # Defines where the PubChem reference structure should be saved in the aligned output folder.
        pubchem_out = aligned_root / "pubchem_reference" / f"{molecule}_pubchem_reference.xyz"
        if not pubchem_out.exists():
            pubchem_out.parent.mkdir(parents=True, exist_ok=True)
            pubchem_out.write_text(
                xyz_to_text(ref_symbols, ref_coords, comment=f"{molecule} pubchem reference"),
                encoding="utf-8",
            )

        # Defines where the aligned model structure should be saved in the aligned output folder, 
        # and saves it there in XYZ format with a comment indicating the molecule, model, and that it is aligned to PubChem.
        aligned_out = aligned_root / model / f"{molecule}_{model}_aligned.xyz"
        aligned_out.parent.mkdir(parents=True, exist_ok=True)
        aligned_out.write_text(
            xyz_to_text(reordered_symbols, aligned_coords, comment=f"{molecule} {model} aligned to pubchem"),
            encoding="utf-8",
        )

        # Update result dict
        record.update({
            "rmsd": float(kabsch_rmsd),
            "score": float(score),
            "status": "ok",
            "note": mapping_source,
            "mapping_source": mapping_source,
            "used_reflection": bool(used_reflection),
            "aligned_model_file": str(aligned_out),
            "valid": True,
            "permutation_ref_to_model": json.dumps([int(x) for x in perm.tolist()]),
            "error_code": 0,
            "error_label": ERROR_CODE_MAP[0],
            "error_message": "",
        })
        return record
    
    # If any unexpected error occurs during the process, catch it and return an error record with the details of the exception.
    except Exception as exc:
        error_message = str(exc)
        error_code, error_label = classify_evaluation_error(error_message, status="failed")
        record.update({
            "status": "failed",
            "note": error_message,
            "score": 0.0,
            "error_code": error_code,
            "error_label": error_label,
            "error_message": error_message,
        })
        return record


# Function to launch the Streamlit viewer for visualizing the results. It checks if the viewer script exists, 
# and if so, it attempts to launch it using subprocess. If it fails to launch, it prints an error message.
def launch_viewer():
    viewer_path = BASE_DIR / VIEWER_FILENAME
    if not viewer_path.exists():
        return
    try:
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(viewer_path)])
        print(f"Launched viewer: python -m streamlit run {viewer_path.name}")
    except Exception as exc:
        print(f"Could not launch viewer automatically: {exc}")


# Finds all run folders created by Code 1, such as run_001, run_002, etc.
# The folders are sorted so the evaluation processes them in order.
def discover_run_dirs() -> list[Path]:
    runs_dir = BASE_DIR / "runs"
    if not runs_dir.exists():
        return []
    run_dirs = []
    for p in runs_dir.iterdir():
        if p.is_dir():
            m = re.fullmatch(r"run_(\d{3})", p.name)
            if m:
                run_dirs.append((int(m.group(1)), p))
    run_dirs.sort(key=lambda x: x[0])
    return [p for _, p in run_dirs]

# Scans one run folder and groups the PubChem, Gemini, OpenAI,
# and Claude XYZ files by molecule name so they can be compared.
def scan_structure_files_in_root(struct_root: Path) -> Dict[str, Dict[str, Path]]:
    files: Dict[str, Dict[str, Path]] = defaultdict(dict)

    for path in (struct_root / "pubchem").glob("*_pubchem.xyz"):
        files[safe_stem_base(path, "_pubchem")]["pubchem"] = path
    for path in (struct_root / "gemini").glob("*_gemini.xyz"):
        files[safe_stem_base(path, "_gemini")]["gemini"] = path
    for path in (struct_root / "openai").glob("*_openai.xyz"):
        files[safe_stem_base(path, "_openai")]["openai"] = path
    for path in (struct_root / "claude").glob("*_claude.xyz"):
        files[safe_stem_base(path, "_claude")]["claude"] = path

    return dict(sorted(files.items()))


def savefig_all(out_dir: Path, filename: str, dpi: int = 200) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / filename, dpi=dpi)




def add_bar_labels(ax, fmt="{:.2f}", pad=3):
    for container in ax.containers:
        try:
            bars = [bar for bar in container if bar is not None and hasattr(bar, "get_height")]
            if not bars:
                continue

            labels = []
            for bar in bars:
                h = bar.get_height()
                if h is None or (isinstance(h, float) and (np.isnan(h) or np.isinf(h))):
                    labels.append("")
                else:
                    labels.append(fmt.format(h))

            ax.bar_label(bars, labels=labels, padding=pad, fontsize=10)
        except Exception:
            continue


def add_patch_value_labels(ax, fmt="{:.0f}", pad_fraction=0.01):
    heights = []
    for patch in ax.patches:
        try:
            h = float(patch.get_height())
        except Exception:
            continue
        if np.isnan(h) or np.isinf(h):
            continue
        heights.append(h)

    if not heights:
        return

    y_min, y_max = ax.get_ylim()
    span = max(abs(y_max - y_min), 1.0)
    pad = span * pad_fraction

    for patch in ax.patches:
        try:
            h = float(patch.get_height())
        except Exception:
            continue
        if np.isnan(h) or np.isinf(h) or h <= 0:
            continue
        x = patch.get_x() + patch.get_width() / 2.0
        ax.text(x, h + pad, fmt.format(h), ha="center", va="bottom", fontsize=9)

def annotate_boxplot_medians(ax, data, fmt="{:.3f}"):
    for i, values in enumerate(data, start=1):
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        median = float(np.median(arr))
        q3 = float(np.percentile(arr, 75))
        y = q3 + max(0.03 * max(abs(q3), 1.0), 0.01)
        ax.text(i, y, fmt.format(median), ha="center", va="bottom", fontsize=10)


def add_histogram_labels(ax, counts, patches, fmt="{:.0f}"):
    if counts is None or patches is None:
        return
    finite_counts = [float(c) for c in counts if c is not None and np.isfinite(c) and float(c) > 0]
    max_count = max(finite_counts, default=0.0)
    if max_count <= 0:
        return

    y_min, y_max = ax.get_ylim()
    span = max(float(y_max) - float(y_min), max_count, 1.0)
    y_offset = max(span * 0.015, 0.08)
    ax.set_ylim(y_min, y_max + y_offset * 3.0)

    for count, patch in zip(counts, patches):
        if count is None or not np.isfinite(count) or float(count) <= 0:
            continue
        x = patch.get_x() + patch.get_width() / 2.0
        y = patch.get_height()
        ax.text(x, y + y_offset, fmt.format(float(count)), ha="center", va="bottom", fontsize=9)


def make_plots(df: pd.DataFrame, df_rmsd_valid: pd.DataFrame, threshold_df: pd.DataFrame, out_dir: Path, title_suffix: str = "") -> None:
    if df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df_rmsd_valid = df_rmsd_valid.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df_rmsd_valid["model"] = pd.Categorical(df_rmsd_valid["model"], categories=MODEL_ORDER, ordered=True)

    suffix = f" {title_suffix}" if title_suffix else ""

    fig, ax = plt.subplots(figsize=(8, 5))
    mean_vals = df_rmsd_valid.groupby("model", observed=False)["rmsd"].mean().reindex(MODEL_ORDER).dropna()
    sem_vals = df_rmsd_valid.groupby("model", observed=False)["rmsd"].apply(safe_sem).reindex(mean_vals.index)
    ax.bar(mean_vals.index, mean_vals.values, yerr=sem_vals.values, capsize=5)
    ax.set_ylabel("Mean RMSD")
    ax.set_title(f"Mean RMSD by Model{suffix}")
    add_bar_labels(ax, fmt="{:.3f}")
    fig.tight_layout()
    savefig_all(out_dir, "mean_rmsd_by_model.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    mean_score_vals = df.groupby("model", observed=False)["score"].mean().reindex(MODEL_ORDER).dropna()
    sem_score_vals = df.groupby("model", observed=False)["score"].apply(safe_sem).reindex(mean_score_vals.index)
    ax.bar(mean_score_vals.index, mean_score_vals.values, yerr=sem_score_vals.values, capsize=5)
    ax.set_ylabel("Mean score")
    ax.set_title(f"Mean Score by Model{suffix}")
    add_bar_labels(ax, fmt="{:.3f}")
    fig.tight_layout()
    savefig_all(out_dir, "mean_score_by_model.png")
    plt.close(fig)

    present_models = [m for m in MODEL_ORDER if m in df["model"].astype(str).values or m in df_rmsd_valid["model"].astype(str).values]
    box_data = [df_rmsd_valid.loc[df_rmsd_valid["model"] == model, "rmsd"].dropna().values for model in present_models]
    score_box_data = [df.loc[df["model"] == model, "score"].dropna().values for model in present_models]

    if box_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(box_data, tick_labels=present_models)
        ax.set_ylabel("RMSD")
        ax.set_title(f"RMSD Boxplot by Model{suffix}")
        annotate_boxplot_medians(ax, box_data, fmt="{:.3f}")
        fig.tight_layout()
        savefig_all(out_dir, "rmsd_boxplot.png")
        plt.close(fig)

        plt.figure(figsize=(8, 5))
        plt.violinplot(box_data, showmeans=True, showmedians=False)
        plt.xticks(range(1, len(present_models) + 1), present_models)
        plt.ylabel("RMSD")
        plt.title(f"RMSD Violin Plot by Model{suffix}")
        plt.tight_layout()
        savefig_all(out_dir, "rmsd_violinplot.png")
        plt.close()

    if score_box_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(score_box_data, tick_labels=present_models)
        ax.set_ylabel("Score")
        ax.set_title(f"Score Boxplot by Model{suffix}")
        annotate_boxplot_medians(ax, score_box_data, fmt="{:.3f}")
        fig.tight_layout()
        savefig_all(out_dir, "score_boxplot.png")
        plt.close(fig)

        plt.figure(figsize=(8, 5))
        plt.violinplot(score_box_data, showmeans=True, showmedians=False)
        plt.xticks(range(1, len(present_models) + 1), present_models)
        plt.ylabel("Score")
        plt.title(f"Score Violin Plot by Model{suffix}")
        plt.tight_layout()
        savefig_all(out_dir, "score_violinplot.png")
        plt.close()

    bins = np.arange(0, HIST_XMAX + HIST_BIN_WIDTH, HIST_BIN_WIDTH)
    zoom_bins = np.arange(0, ZOOM_XMAX + HIST_BIN_WIDTH, HIST_BIN_WIDTH)
    score_bins = np.arange(0, 1.05, 0.05)

    for model in MODEL_ORDER:
        vals = df_rmsd_valid.loc[df_rmsd_valid["model"] == model, "rmsd"].dropna().values
        if len(vals) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            counts, _, patches = ax.hist(vals, bins=bins, edgecolor="black")
            ax.set_xlim(0, HIST_XMAX)
            ax.set_xlabel("RMSD")
            ax.set_ylabel("Count")
            ax.set_title(f"RMSD Distribution - {model}{suffix}")
            add_histogram_labels(ax, counts, patches, fmt="{:.0f}")
            fig.tight_layout()
            savefig_all(out_dir, f"hist_rmsd_{model}.png")
            plt.close(fig)

            zoom_vals = vals[vals <= ZOOM_XMAX]
            if len(zoom_vals) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                counts, _, patches = ax.hist(zoom_vals, bins=zoom_bins, edgecolor="black")
                ax.set_xlim(0, ZOOM_XMAX)
                ax.set_xlabel("RMSD")
                ax.set_ylabel("Count")
                ax.set_title(f"RMSD Distribution (0-{ZOOM_XMAX}) - {model}{suffix}")
                add_histogram_labels(ax, counts, patches, fmt="{:.0f}")
                fig.tight_layout()
                savefig_all(out_dir, f"hist_rmsd_zoom_{model}.png")
                plt.close(fig)

        score_vals = df.loc[df["model"] == model, "score"].dropna().values
        if len(score_vals) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            counts, _, patches = ax.hist(score_vals, bins=score_bins, edgecolor="black")
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.set_title(f"Score Distribution - {model}{suffix}")
            add_histogram_labels(ax, counts, patches, fmt="{:.0f}")
            fig.tight_layout()
            savefig_all(out_dir, f"hist_score_{model}.png")
            plt.close(fig)

    plt.figure(figsize=(8, 5))
    for model in MODEL_ORDER:
        vals = np.sort(df_rmsd_valid.loc[df_rmsd_valid["model"] == model, "rmsd"].dropna().values)
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals) * 100.0
        plt.plot(vals, y, marker=".", linestyle="-", label=model)
    plt.xlabel("RMSD")
    plt.ylabel("Cumulative percent of molecules")
    plt.title(f"Cumulative RMSD by Model{suffix}")
    plt.legend()
    plt.tight_layout()
    savefig_all(out_dir, "cumulative_rmsd.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    for model in MODEL_ORDER:
        sub = df_rmsd_valid[df_rmsd_valid["model"] == model].copy()
        sub["atom_count"] = pd.to_numeric(sub["atom_count"], errors="coerce")
        sub["rmsd"] = pd.to_numeric(sub["rmsd"], errors="coerce")
        sub = sub.dropna(subset=["atom_count", "rmsd"])
        if sub.empty:
            continue
        plt.scatter(sub["atom_count"], sub["rmsd"], label=f"{model} data", alpha=0.35, s=35)
        if len(sub) >= 2 and sub["atom_count"].nunique() >= 2:
            x = sub["atom_count"].to_numpy(dtype=float)
            y = sub["rmsd"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, linewidth=2, label=f"{model} fit")
    plt.xlabel("Atom count")
    plt.ylabel("RMSD")
    plt.title(f"Atom Count vs RMSD with Linear Regression")
    plt.legend()
    plt.tight_layout()
    savefig_all(out_dir, "atomcount_vs_rmsd_linear_regression.png")
    plt.close()

    threshold_plot_df = threshold_df.set_index("model").reindex(MODEL_ORDER).dropna(how="all")
    x = np.arange(len(threshold_plot_df.index))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, thr in enumerate(THRESHOLDS):
        vals = threshold_plot_df[f"pct_le_{thr}"].values
        ax.bar(x + i * width, vals, width=width, label=f"≤ {thr}")
    ax.set_xticks(x + width * (len(THRESHOLDS) - 1) / 2, threshold_plot_df.index)
    ax.set_ylabel("Percent of molecules")
    ax.set_title(f"RMSD Threshold Success{suffix}")
    ax.legend()
    add_bar_labels(ax, fmt="{:.1f}%")
    fig.tight_layout()
    savefig_all(out_dir, "threshold_success.png")
    plt.close(fig)

    success_df = (
        df.groupby("model", observed=False)["valid"]
        .mean()
        .mul(100)
        .reindex(MODEL_ORDER)
        .dropna()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(success_df.index, success_df.values)
    ax.set_ylabel("Success rate (%)")
    ax.set_title(f"Successful mapping + alignment rate{suffix}")
    add_bar_labels(ax, fmt="{:.1f}%")
    fig.tight_layout()
    savefig_all(out_dir, "success_rate.png")
    plt.close(fig)


def make_error_summary_outputs(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty or "model" not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    out_dir.mkdir(parents=True, exist_ok=True)
    err_df = df.copy()
    err_df["error_code"] = pd.to_numeric(err_df.get("error_code"), errors="coerce")
    err_df["error_label"] = err_df.get("error_label")
    err_df["error_message"] = err_df.get("error_message")

    err_df = err_df[err_df["error_code"].notna()]
    err_df = err_df[err_df["error_code"] != 0].copy()
    if err_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    err_df["error_code"] = err_df["error_code"].astype(int)
    err_df["error_message_normalized"] = err_df["error_message"].apply(normalize_error_message)
    err_df["error_display"] = err_df.apply(
        lambda row: build_error_display_label(row.get("error_code"), row.get("error_label"), row.get("error_message_normalized")),
        axis=1,
    )

    error_summary = (
        err_df.groupby(["error_code", "error_label", "model"]).size()
        .reset_index(name="count")
        .sort_values(["error_code", "model", "count"], ascending=[True, True, False])
    )

    error_pivot = (
        error_summary.pivot_table(
            index=["error_code", "error_label"],
            columns="model",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=MODEL_ORDER, fill_value=0)
        .sort_index()
    )

    error_summary.to_csv(out_dir / "error_summary_by_model.csv", index=False)
    error_pivot_reset = error_pivot.reset_index()
    error_pivot_reset.to_csv(out_dir / "error_summary_pivot.csv", index=False)

    labels = [f"{int(code)}: {label}" for code, label in error_pivot.index.tolist()]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.15), 6))
    for i, model in enumerate(MODEL_ORDER):
        if model not in error_pivot.columns:
            continue
        vals = error_pivot[model].to_numpy(dtype=float)
        ax.bar(x + i * width, vals, width=width, label=model)
    ax.set_xticks(x + width * (len(MODEL_ORDER) - 1) / 2, labels, rotation=45, ha="right")
    ax.set_ylabel("Error count")
    ax.set_title("Error Counts by Model Across All Runs")
    ax.legend()
    add_bar_labels(ax, fmt="{:.0f}")
    fig.tight_layout()
    savefig_all(out_dir, "error_counts_by_model_all_runs.png")
    plt.close(fig)

    message_summary = (
        err_df.groupby(["error_code", "error_label", "error_message_normalized", "model"]).size()
        .reset_index(name="count")
        .sort_values(["error_code", "count", "model"], ascending=[True, False, True])
    )

    message_pivot = (
        message_summary.pivot_table(
            index=["error_code", "error_label", "error_message_normalized"],
            columns="model",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=MODEL_ORDER, fill_value=0)
        .sort_index()
    )

    message_summary.to_csv(out_dir / "error_message_summary_by_model.csv", index=False)
    message_pivot_reset = message_pivot.reset_index()
    message_pivot_reset.to_csv(out_dir / "error_message_summary_pivot.csv", index=False)

    message_labels = [
        build_error_display_label(code, label, message)
        for code, label, message in message_pivot.index.tolist()
    ]
    x = np.arange(len(message_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(12, len(message_labels) * 1.2), 7))
    for i, model in enumerate(MODEL_ORDER):
        if model not in message_pivot.columns:
            continue
        vals = message_pivot[model].to_numpy(dtype=float)
        ax.bar(x + i * width, vals, width=width, label=model)
    ax.set_xticks(x + width * (len(MODEL_ORDER) - 1) / 2, message_labels, rotation=60, ha="right")
    ax.set_ylabel("Error count")
    ax.set_title("Individual Error Messages by Code and Model Across All Runs")
    ax.legend()
    add_bar_labels(ax, fmt="{:.0f}")
    fig.tight_layout()
    savefig_all(out_dir, "error_messages_by_code_and_model_all_runs.png")
    plt.close(fig)

    return error_summary, error_pivot_reset, message_summary, message_pivot_reset


def make_run_aggregate_plots(run_summary_df: pd.DataFrame, out_dir: Path) -> None:
    if run_summary_df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    ordered_runs = sorted(run_summary_df["run"].dropna().astype(str).unique())

    plt.figure(figsize=(9, 5))
    for model in MODEL_ORDER:
        sub = run_summary_df[run_summary_df["model"] == model].sort_values("run")
        if sub.empty:
            continue
        plt.plot(sub["run"], sub["mean_rmsd"], marker="o", linestyle="-", label=model)
    plt.xlabel("Run")
    plt.ylabel("Mean RMSD")
    plt.title("Mean RMSD by Run")
    plt.legend()
    plt.tight_layout()
    savefig_all(out_dir, "run_mean_rmsd_by_model.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    for model in MODEL_ORDER:
        sub = run_summary_df[run_summary_df["model"] == model].sort_values("run")
        if sub.empty:
            continue
        plt.plot(sub["run"], sub["mean_score"], marker="o", linestyle="-", label=model)
    plt.xlabel("Run")
    plt.ylabel("Mean score")
    plt.title("Mean Score by Run")
    plt.legend()
    plt.tight_layout()
    savefig_all(out_dir, "run_mean_score_by_model.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    for model in MODEL_ORDER:
        sub = run_summary_df[run_summary_df["model"] == model].sort_values("run")
        if sub.empty:
            continue
        plt.plot(sub["run"], sub["success_rate_pct"], marker="o", linestyle="-", label=model)
    plt.xlabel("Run")
    plt.ylabel("Success rate (%)")
    plt.title("Success Rate by Run")
    plt.legend()
    plt.tight_layout()
    savefig_all(out_dir, "run_success_rate_by_model.png")
    plt.close()




def make_run_level_summary_outputs(run_summary_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if run_summary_df.empty:
        return pd.DataFrame()

    out_dir.mkdir(parents=True, exist_ok=True)
    rs = run_summary_df.copy()
    rs["run"] = rs["run"].astype(str)

    run_averaged_summary = (
        rs.groupby("model", as_index=False)
        .agg(
            n_runs=("run", "nunique"),
            avg_run_mean_rmsd=("mean_rmsd", "mean"),
            sem_run_mean_rmsd=("mean_rmsd", safe_sem),
            avg_run_mean_score=("mean_score", "mean"),
            sem_run_mean_score=("mean_score", safe_sem),
            avg_run_success_rate_pct=("success_rate_pct", "mean"),
            sem_run_success_rate_pct=("success_rate_pct", safe_sem),
        )
    )
    run_averaged_summary.to_csv(out_dir / "run_averaged_summary.csv", index=False)

    ordered = [m for m in MODEL_ORDER if m in run_averaged_summary["model"].values]

    if ordered:
        plot_df = run_averaged_summary.set_index("model").reindex(ordered).reset_index()

        plt.figure(figsize=(8, 5))
        plt.bar(
            plot_df["model"],
            plot_df["avg_run_mean_score"],
            yerr=plot_df["sem_run_mean_score"].to_numpy(dtype=float),
            capsize=5,
        )
        plt.ylabel("Average score across runs")
        plt.title("Run-averaged Mean Score by Model")
        plt.tight_layout()
        savefig_all(out_dir, "run_averaged_mean_score.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(
            plot_df["model"],
            plot_df["avg_run_mean_rmsd"],
            yerr=plot_df["sem_run_mean_rmsd"].to_numpy(dtype=float),
            capsize=5,
        )
        plt.ylabel("Average RMSD across runs")
        plt.title("Run-averaged Mean RMSD by Model")
        plt.tight_layout()
        savefig_all(out_dir, "run_averaged_mean_rmsd.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(
            plot_df["model"],
            plot_df["avg_run_success_rate_pct"],
            yerr=plot_df["sem_run_success_rate_pct"].to_numpy(dtype=float),
            capsize=5,
        )
        plt.ylabel("Average success rate (%) across runs")
        plt.title("Run-averaged Success Rate by Model")
        plt.tight_layout()
        savefig_all(out_dir, "run_averaged_success_rate.png")
        plt.close()

    mean_per_run = (
        rs.groupby("run", as_index=False)
        .agg(
            avg_score_all_models=("mean_score", "mean"),
            avg_rmsd_all_models=("mean_rmsd", "mean"),
            avg_success_rate_all_models=("success_rate_pct", "mean"),
        )
        .sort_values("run")
    )
    mean_per_run.to_csv(out_dir / "overall_mean_per_run.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(mean_per_run["run"], mean_per_run["avg_score_all_models"], marker="o", linestyle="-")
    plt.xlabel("Run")
    plt.ylabel("Average score across models")
    plt.title("Average Score for Each Run")
    plt.tight_layout()
    savefig_all(out_dir, "overall_mean_score_per_run.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(mean_per_run["run"], mean_per_run["avg_rmsd_all_models"], marker="o", linestyle="-")
    plt.xlabel("Run")
    plt.ylabel("Average RMSD across models")
    plt.title("Average RMSD for Each Run")
    plt.tight_layout()
    savefig_all(out_dir, "overall_mean_rmsd_per_run.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(mean_per_run["run"], mean_per_run["avg_success_rate_all_models"], marker="o", linestyle="-")
    plt.xlabel("Run")
    plt.ylabel("Average success rate (%) across models")
    plt.title("Average Success Rate for Each Run")
    plt.tight_layout()
    savefig_all(out_dir, "overall_mean_success_rate_per_run.png")
    plt.close()

    return run_averaged_summary


def _clear_directory_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except Exception:
            pass


def _copy_tree_contents(src: Path, dst: Path, clear_first: bool = False) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    if clear_first:
        _clear_directory_contents(dst)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def sync_latest_run_outputs_to_root(latest_run_dir: Path) -> None:
    latest_results_dir = latest_run_dir / "results"
    latest_aligned_root = latest_results_dir / "aligned_xyz"

    if latest_aligned_root.exists():
        if ALIGNED_ROOT.exists():
            import shutil
            shutil.rmtree(ALIGNED_ROOT)
        import shutil
        shutil.copytree(latest_aligned_root, ALIGNED_ROOT)

    for filename in ["atom_mapping_table.csv", "rmsd_results_parallel.csv", "all_models_vs_pubchem.xlsx"]:
        src = latest_results_dir / filename
        dst = RESULTS_DIR / filename
        if src.exists():
            dst.write_bytes(src.read_bytes())


def run_single_run(run_dir: Path) -> dict:
    run_name = run_dir.name
    struct_root = run_dir / "structures"
    results_dir = run_dir / "results"
    aligned_root = results_dir / "aligned_xyz"
    out_dir = run_dir / "combined_rmsd_analysis"

    for model in ["gemini", "openai", "claude", "pubchem_reference"]:
        (aligned_root / model).mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_map = scan_structure_files_in_root(struct_root)
    tasks = []
    for molecule, files in file_map.items():
        ref_path = files.get("pubchem")
        if ref_path is None:
            continue
        for model in MODELS:
            model_path = files.get(model)
            if model_path is not None:
                tasks.append((molecule, model, ref_path, model_path, aligned_root))

    if not tasks:
        return {
            "run": run_name,
            "mapping_df": pd.DataFrame(),
            "results_df": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
            "threshold_df": pd.DataFrame(),
        }

    rows = []
    print(f"\nStarting evaluation for {run_name} ...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_one_pair, task): task for task in tasks}
        for idx, future in enumerate(as_completed(future_to_task), start=1):
            row = future.result()
            row["run"] = run_name
            rows.append(row)
            print(f"[{run_name} {idx}/{len(tasks)}] {row['model']} | {row['molecule']} | RMSD={row['rmsd']} | status={row['status']}")

    df = pd.DataFrame(rows).sort_values(["molecule", "model"]).reset_index(drop=True)
    mapping_df = df[[
        "run", "molecule", "model", "atom_count", "status", "mapping_source", "permutation_ref_to_model",
        "pubchem_file", "model_file", "aligned_model_file", "note"
    ]].copy()

    df["rmsd"] = pd.to_numeric(df["rmsd"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["atom_count"] = pd.to_numeric(df["atom_count"], errors="coerce")
    df_rmsd_valid = df[df["valid"] & df["rmsd"].notna()].copy()

    summary = (
        df.groupby("model")
        .agg(
            n_total=("model", "count"),
            n_success=("valid", "sum"),
            mean_rmsd=("rmsd", "mean"),
            median_rmsd=("rmsd", "median"),
            std_rmsd=("rmsd", "std"),
            sem_rmsd=("rmsd", safe_sem),
            mean_score=("score", "mean"),
            mean_atoms=("atom_count", "mean"),
        )
        .reset_index()
        .sort_values("mean_rmsd")
    )
    summary["run"] = run_name
    summary["success_rate_pct"] = 100.0 * summary["n_success"] / summary["n_total"]

    status_summary = (
        df.groupby(["model", "status"]).size().reset_index(name="count")
        .sort_values(["model", "count"], ascending=[True, False])
    )

    error_summary_df, error_pivot_df, error_message_summary_df, error_message_pivot_df = make_error_summary_outputs(df, OUT_DIR)

    threshold_rows = []
    for model in sorted(df_rmsd_valid["model"].dropna().unique()):
        model_vals = df_rmsd_valid.loc[df_rmsd_valid["model"] == model, "rmsd"].dropna()
        n = len(model_vals)
        row = {"model": model, "n": n}
        for thr in THRESHOLDS:
            row[f"pct_le_{thr}"] = (100.0 * (model_vals <= thr).sum() / n) if n else np.nan
        threshold_rows.append(row)
    threshold_df = pd.DataFrame(threshold_rows)

    mapping_df.to_csv(results_dir / "atom_mapping_table.csv", index=False)
    df.to_csv(results_dir / "rmsd_results_parallel.csv", index=False)
    summary.to_csv(out_dir / "overall_summary.csv", index=False)
    status_summary.to_csv(out_dir / "status_summary.csv", index=False)
    threshold_df.to_csv(out_dir / "threshold_summary.csv", index=False)
    df.to_csv(out_dir / "all_rmsd_results.csv", index=False)

    with pd.ExcelWriter(results_dir / "all_models_vs_pubchem.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="per_molecule", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        mapping_df.to_excel(writer, sheet_name="atom_mapping", index=False)

    make_plots(df, df_rmsd_valid, threshold_df, out_dir, title_suffix=f"({run_name})")

    return {
        "run": run_name,
        "mapping_df": mapping_df,
        "results_df": df,
        "summary_df": summary,
        "threshold_df": threshold_df,
    }


def run_pipeline() -> dict:
    run_dirs = discover_run_dirs()
    if not run_dirs:
        raise SystemExit("No run folders found under runs/")

    all_mapping = []
    all_results = []
    all_summary = []
    all_threshold = []

    for run_dir in run_dirs:
        result = run_single_run(run_dir)
        if not result["results_df"].empty:
            all_mapping.append(result["mapping_df"])
            all_results.append(result["results_df"])
        if not result["summary_df"].empty:
            all_summary.append(result["summary_df"])
        if not result["threshold_df"].empty:
            tdf = result["threshold_df"].copy()
            tdf["run"] = result["run"]
            all_threshold.append(tdf)

    if not all_results:
        raise SystemExit("No matched model/PubChem XYZ files found across runs.")

    df = pd.concat(all_results, ignore_index=True).sort_values(["run", "molecule", "model"]).reset_index(drop=True)
    mapping_df = pd.concat(all_mapping, ignore_index=True) if all_mapping else pd.DataFrame()
    run_summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    threshold_all_runs_df = pd.concat(all_threshold, ignore_index=True) if all_threshold else pd.DataFrame()

    df["rmsd"] = pd.to_numeric(df["rmsd"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["atom_count"] = pd.to_numeric(df["atom_count"], errors="coerce")
    df_rmsd_valid = df[df["valid"] & df["rmsd"].notna()].copy()

    overall_summary = (
        df.groupby("model")
        .agg(
            n_total=("model", "count"),
            n_success=("valid", "sum"),
            mean_rmsd=("rmsd", "mean"),
            median_rmsd=("rmsd", "median"),
            std_rmsd=("rmsd", "std"),
            sem_rmsd=("rmsd", safe_sem),
            mean_score=("score", "mean"),
            mean_atoms=("atom_count", "mean"),
        )
        .reset_index()
        .sort_values("mean_rmsd")
    )
    overall_summary["success_rate_pct"] = 100.0 * overall_summary["n_success"] / overall_summary["n_total"]

    status_summary = (
        df.groupby(["model", "status"]).size().reset_index(name="count")
        .sort_values(["model", "count"], ascending=[True, False])
    )

    error_summary_df, error_pivot_df, error_message_summary_df, error_message_pivot_df = make_error_summary_outputs(df, OUT_DIR)

    threshold_rows = []
    for model in sorted(df_rmsd_valid["model"].dropna().unique()):
        model_vals = df_rmsd_valid.loc[df_rmsd_valid["model"] == model, "rmsd"].dropna()
        n = len(model_vals)
        row = {"model": model, "n": n}
        for thr in THRESHOLDS:
            row[f"pct_le_{thr}"] = (100.0 * (model_vals <= thr).sum() / n) if n else np.nan
        threshold_rows.append(row)
    threshold_df = pd.DataFrame(threshold_rows)

    mapping_df.to_csv(MAPPING_CSV, index=False)
    df.to_csv(RESULTS_CSV, index=False)
    df.to_csv(OUT_DIR / "all_rmsd_results.csv", index=False)
    overall_summary.to_csv(OUT_DIR / "overall_summary.csv", index=False)
    run_summary_df.to_csv(OUT_DIR / "per_run_summary.csv", index=False)
    status_summary.to_csv(OUT_DIR / "status_summary.csv", index=False)
    threshold_df.to_csv(OUT_DIR / "threshold_summary.csv", index=False)
    if not error_summary_df.empty:
        error_summary_df.to_csv(OUT_DIR / "error_summary_by_model.csv", index=False)
    if not error_pivot_df.empty:
        error_pivot_df.to_csv(OUT_DIR / "error_summary_pivot.csv", index=False)
    if not error_message_summary_df.empty:
        error_message_summary_df.to_csv(OUT_DIR / "error_message_summary_by_model.csv", index=False)
    if not error_message_pivot_df.empty:
        error_message_pivot_df.to_csv(OUT_DIR / "error_message_summary_pivot.csv", index=False)
    if not threshold_all_runs_df.empty:
        threshold_all_runs_df.to_csv(OUT_DIR / "per_run_threshold_summary.csv", index=False)

    run_averaged_summary = make_run_level_summary_outputs(run_summary_df, OUT_DIR)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="per_molecule_all_runs", index=False)
        overall_summary.to_excel(writer, sheet_name="overall_summary", index=False)
        run_summary_df.to_excel(writer, sheet_name="per_run_summary", index=False)
        mapping_df.to_excel(writer, sheet_name="atom_mapping", index=False)
        if not error_summary_df.empty:
            error_summary_df.to_excel(writer, sheet_name="error_summary", index=False)
        if not error_pivot_df.empty:
            error_pivot_df.to_excel(writer, sheet_name="error_pivot", index=False)
        if not error_message_summary_df.empty:
            error_message_summary_df.to_excel(writer, sheet_name="error_message_summary", index=False)
        if not error_message_pivot_df.empty:
            error_message_pivot_df.to_excel(writer, sheet_name="error_message_pivot", index=False)
        if 'run_averaged_summary' in locals() and not run_averaged_summary.empty:
            run_averaged_summary.to_excel(writer, sheet_name="run_averaged_summary", index=False)

    make_plots(df, df_rmsd_valid, threshold_df, OUT_DIR, title_suffix="(All Runs)")
    make_run_aggregate_plots(run_summary_df, OUT_DIR)

    latest_run_dir = run_dirs[-1]
    try:
        sync_latest_run_outputs_to_root(latest_run_dir)
    except Exception as exc:
        print(f"Warning: could not sync latest run outputs to project root: {exc}")

    print("\nDone.")
    print(f"Saved aggregate plots and summaries in: {OUT_DIR}")
    print(f"Saved aggregate mapping CSV in: {MAPPING_CSV}")
    print(f"Saved aggregate results CSV in: {RESULTS_CSV}")
    print(f"Latest run synced to viewer root from: {latest_run_dir.name}")

    if AUTO_LAUNCH_VIEWER:
        launch_viewer()

    return {
        "mapping_df": mapping_df,
        "results_df": df,
        "run_summary_df": run_summary_df,
        "run_averaged_summary_csv": OUT_DIR / "run_averaged_summary.csv",
        "summary_csv": OUT_DIR / "overall_summary.csv",
        "per_run_summary_csv": OUT_DIR / "per_run_summary.csv",
    }


if __name__ == "__main__":
    run_pipeline()
