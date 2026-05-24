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

BASE_DIR = Path(__file__).resolve().parent
STRUCT_ROOT = BASE_DIR / "structures"
RESULTS_DIR = BASE_DIR / "results"
ALIGNED_ROOT = RESULTS_DIR / "aligned_xyz"
OUT_DIR = BASE_DIR / "combined_rmsd_analysis"

STRUCT_DIRS = {
    "gemini": STRUCT_ROOT / "gemini",
    "openai": STRUCT_ROOT / "openai",
    "claude": STRUCT_ROOT / "claude",
    "pubchem": STRUCT_ROOT / "pubchem",
}

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

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(ALIGNED_ROOT / "gemini").mkdir(parents=True, exist_ok=True)
(ALIGNED_ROOT / "openai").mkdir(parents=True, exist_ok=True)
(ALIGNED_ROOT / "claude").mkdir(parents=True, exist_ok=True)
(ALIGNED_ROOT / "pubchem_reference").mkdir(parents=True, exist_ok=True)

MAPPING_CSV = RESULTS_DIR / "atom_mapping_table.csv"
RESULTS_CSV = RESULTS_DIR / "rmsd_results_parallel.csv"
OUTPUT_XLSX = RESULTS_DIR / "all_models_vs_pubchem.xlsx"

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

# ==========================================================
# HELPERS
# ==========================================================


def safe_sem(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(values) <= 1:
        return np.nan
    return values.std(ddof=1) / np.sqrt(len(values))



def rmsd_to_score(value: float, midpoint: float = FERMI_MIDPOINT, k: float = FERMI_K):
    if value is None or pd.isna(value):
        return 0.0
    return float(1.0 / (1.0 + math.exp((float(value) - float(midpoint)) / float(k))))



def safe_stem_base(path: Path, suffix: str) -> str:
    stem = path.stem
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem



def scan_structure_files() -> Dict[str, Dict[str, Path]]:
    files: Dict[str, Dict[str, Path]] = defaultdict(dict)

    for path in STRUCT_DIRS["pubchem"].glob("*_pubchem.xyz"):
        files[safe_stem_base(path, "_pubchem")]["pubchem"] = path
    for path in STRUCT_DIRS["gemini"].glob("*_gemini.xyz"):
        files[safe_stem_base(path, "_gemini")]["gemini"] = path
    for path in STRUCT_DIRS["openai"].glob("*_openai.xyz"):
        files[safe_stem_base(path, "_openai")]["openai"] = path
    for path in STRUCT_DIRS["claude"].glob("*_claude.xyz"):
        files[safe_stem_base(path, "_claude")]["claude"] = path

    return dict(sorted(files.items()))



def read_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid XYZ file: {path}")

    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"XYZ atom count missing in {path}") from exc

    body = lines[2:2 + n_atoms]
    if len(body) != n_atoms:
        raise ValueError(f"XYZ atom count mismatch in {path}")

    symbols: List[str] = []
    coords: List[List[float]] = []
    for line in body:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ line in {path}: {line}")
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return symbols, np.asarray(coords, dtype=float)


def is_failure_marker(path: Path) -> tuple[bool, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception as exc:
        return False, f"could_not_read_marker: {exc}"
    if text.startswith("FAILED"):
        return True, text
    return False, ""



def xyz_to_text(symbols: List[str], coords: np.ndarray, comment: str = "") -> str:
    lines = [str(len(symbols)), comment]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym:2s} {x: .8f} {y: .8f} {z: .8f}")
    return "\n".join(lines) + "\n"



def grouped_indices(symbols: List[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = defaultdict(list)
    for idx, sym in enumerate(symbols):
        out[sym].append(idx)
    return dict(out)



def validate_element_counts(ref_symbols: List[str], model_symbols: List[str]):
    if len(ref_symbols) != len(model_symbols):
        raise ValueError("Atom count mismatch")
    ref_groups = grouped_indices(ref_symbols)
    model_groups = grouped_indices(model_symbols)
    if set(ref_groups) != set(model_groups):
        raise ValueError("Element mismatch")
    for sym in ref_groups:
        if len(ref_groups[sym]) != len(model_groups.get(sym, [])):
            raise ValueError(f"Element count mismatch for {sym}")
    return ref_groups, model_groups



def infer_bonds(symbols: List[str], coords: np.ndarray, scale: float = 1.25) -> set[Tuple[int, int]]:
    bonds: set[Tuple[int, int]] = set()
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            ri = COVALENT_RADII.get(symbols[i], 0.8)
            rj = COVALENT_RADII.get(symbols[j], 0.8)
            cutoff = scale * (ri + rj)
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d <= cutoff:
                bonds.add((i, j))
    return bonds



def adjacency_from_bonds(n: int, bonds: set[Tuple[int, int]]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    return adj



def local_env_signature(idx: int, symbols: List[str], adj: List[List[int]]) -> Tuple[int, Tuple[str, ...]]:
    nbr_syms = sorted(symbols[j] for j in adj[idx])
    return len(adj[idx]), tuple(nbr_syms)



def env_penalty(ref_sig, model_sig) -> float:
    ref_degree, ref_nbrs = ref_sig
    model_degree, model_nbrs = model_sig
    penalty = 6.0 * abs(ref_degree - model_degree)
    if ref_nbrs != model_nbrs:
        ref_counts = defaultdict(int)
        model_counts = defaultdict(int)
        for sym in ref_nbrs:
            ref_counts[sym] += 1
        for sym in model_nbrs:
            model_counts[sym] += 1
        all_syms = set(ref_counts) | set(model_counts)
        penalty += 4.0 * sum(abs(ref_counts[s] - model_counts[s]) for s in all_syms)
    return float(penalty)



def sorted_distance_signature(coords: np.ndarray, idx: int, take: int | None = None) -> np.ndarray:
    d = np.linalg.norm(coords - coords[idx], axis=1)
    d = np.sort(np.delete(d, idx))
    if take is not None:
        d = d[:take]
    return d



def signature_penalty(ref_coords, model_coords, ref_idx, model_idx, ref_symbols=None, model_symbols=None) -> float:
    ref_all = sorted_distance_signature(ref_coords, ref_idx, take=8)
    model_all = sorted_distance_signature(model_coords, model_idx, take=8)
    n = min(len(ref_all), len(model_all))
    total = float(np.mean(np.abs(ref_all[:n] - model_all[:n]))) if n > 0 else 0.0
    if ref_symbols is not None and model_symbols is not None:
        ref_same = np.sort([
            float(np.linalg.norm(ref_coords[k] - ref_coords[ref_idx]))
            for k, sym in enumerate(ref_symbols)
            if k != ref_idx and sym == ref_symbols[ref_idx]
        ])[:6]
        model_same = np.sort([
            float(np.linalg.norm(model_coords[k] - model_coords[model_idx]))
            for k, sym in enumerate(model_symbols)
            if k != model_idx and sym == model_symbols[model_idx]
        ])[:6]
        m = min(len(ref_same), len(model_same))
        if m > 0:
            total += 1.5 * float(np.mean(np.abs(ref_same[:m] - model_same[:m])))
    return total



def build_rdkit_mol_from_inferred_bonds(symbols: List[str], coords: np.ndarray):
    if Chem is None:
        return None
    rw = Chem.RWMol()
    for sym in symbols:
        rw.AddAtom(Chem.Atom(sym))
    for i, j in infer_bonds(symbols, coords):
        rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol



def rdkit_substructure_permutations(ref_symbols, ref_coords, model_symbols, model_coords):
    if Chem is None:
        return []
    ref_mol = build_rdkit_mol_from_inferred_bonds(ref_symbols, ref_coords)
    model_mol = build_rdkit_mol_from_inferred_bonds(model_symbols, model_coords)
    if ref_mol is None or model_mol is None:
        return []
    try:
        matches = list(model_mol.GetSubstructMatches(ref_mol, uniquify=False))
    except Exception:
        matches = []
    perms = []
    for match in matches:
        if len(match) != len(ref_symbols):
            continue
        perm = np.asarray(match, dtype=int)
        if [model_symbols[i] for i in perm] == ref_symbols:
            perms.append(perm)
    return perms



def rdkit_atom_mapping(ref_symbols, ref_coords, model_symbols, model_coords):
    perms = rdkit_substructure_permutations(ref_symbols, ref_coords, model_symbols, model_coords)
    if not perms:
        raise ValueError("RDKit could not produce a full atom mapping")

    best_perm = None
    best_cost = None
    ref_centered = ref_coords - ref_coords.mean(axis=0)
    for perm in perms:
        reordered = model_coords[perm]
        reordered_centered = reordered - reordered.mean(axis=0)
        cost = float(np.mean(np.linalg.norm(reordered_centered - ref_centered, axis=1)))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm

    if best_perm is None:
        raise ValueError("RDKit returned no valid atom permutation")
    return best_perm, "rdkit"



def hungarian_atom_mapping(ref_symbols, ref_coords, model_symbols, model_coords):
    if linear_sum_assignment is None:
        raise ValueError("SciPy linear_sum_assignment is not available")

    ref_groups, model_groups = validate_element_counts(ref_symbols, model_symbols)
    ref_adj = adjacency_from_bonds(len(ref_symbols), infer_bonds(ref_symbols, ref_coords))
    model_adj = adjacency_from_bonds(len(model_symbols), infer_bonds(model_symbols, model_coords))
    ref_sigs = [local_env_signature(i, ref_symbols, ref_adj) for i in range(len(ref_symbols))]
    model_sigs = [local_env_signature(i, model_symbols, model_adj) for i in range(len(model_symbols))]

    perm = np.full(len(ref_symbols), -1, dtype=int)
    ref_centered = ref_coords - ref_coords.mean(axis=0)
    model_centered = model_coords - model_coords.mean(axis=0)

    for sym in sorted(ref_groups):
        ref_idx = ref_groups[sym]
        model_idx = model_groups[sym]
        cost = np.zeros((len(ref_idx), len(model_idx)), dtype=float)
        for i, r_idx in enumerate(ref_idx):
            for j, m_idx in enumerate(model_idx):
                cost[i, j] = (
                    signature_penalty(ref_centered, model_centered, r_idx, m_idx, ref_symbols, model_symbols)
                    + env_penalty(ref_sigs[r_idx], model_sigs[m_idx])
                )
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            perm[ref_idx[i]] = model_idx[j]

    if np.any(perm < 0):
        raise ValueError("Hungarian mapping failed to assign all atoms")
    if [model_symbols[i] for i in perm.tolist()] != ref_symbols:
        raise ValueError("Hungarian permutation did not preserve element order")
    return perm, "hungarian"



def align_with_library_kabsch(mobile: np.ndarray, target: np.ndarray, allow_reflection: bool = True):
    if Rotation is None:
        raise ValueError("SciPy Rotation.align_vectors is not available")

    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if mobile.shape != target.shape:
        raise ValueError("Shape mismatch between mobile and target coordinates")

    mobile_centroid = mobile.mean(axis=0)
    target_centroid = target.mean(axis=0)
    mobile_centered = mobile - mobile_centroid
    target_centered = target - target_centroid

    if len(mobile) == 1:
        aligned = mobile - mobile_centroid + target_centroid
        return aligned, np.eye(3), target_centroid - mobile_centroid, False

    reflection_options = [np.array([1.0, 1.0, 1.0])]
    if allow_reflection:
        reflection_options = [np.array(v, dtype=float) for v in itertools.product([-1.0, 1.0], repeat=3)]

    best = None
    for refl in reflection_options:
        reflected = mobile_centered * refl
        try:
            rotation_obj, _ = Rotation.align_vectors(target_centered, reflected)
        except Exception:
            continue
        rotation = rotation_obj.as_matrix()
        aligned_centered = rotation_obj.apply(reflected)
        aligned = aligned_centered + target_centroid
        current_rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
        used_reflection = bool(np.any(refl < 0))
        candidate = (aligned, rotation, target_centroid - mobile_centroid, used_reflection, current_rmsd)
        if best is None or current_rmsd < best[-1]:
            best = candidate

    if best is None:
        raise ValueError("Library Kabsch alignment failed")
    return best[0], best[1], best[2], best[3]



def compute_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((np.asarray(a) - np.asarray(b)) ** 2, axis=1))))




def process_one_pair(task):
    molecule, model, ref_path, model_path, aligned_root = task
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
    }

    try:
        ref_failed, ref_note = is_failure_marker(ref_path)
        if ref_failed:
            record.update({
                "status": "missing_pubchem",
                "note": ref_note,
                "score": 0.0,
            })
            return record

        model_failed, model_note = is_failure_marker(model_path)
        if model_failed:
            try:
                ref_symbols, _ = read_xyz(ref_path)
                record["atom_count"] = len(ref_symbols)
            except Exception:
                pass
            record.update({
                "status": "not_found",
                "note": model_note,
                "score": 0.0,
            })
            return record

        ref_symbols, ref_coords = read_xyz(ref_path)
        model_symbols, model_coords = read_xyz(model_path)
        record["atom_count"] = len(ref_symbols)
        validate_element_counts(ref_symbols, model_symbols)

        try:
            perm, mapping_source = rdkit_atom_mapping(ref_symbols, ref_coords, model_symbols, model_coords)
        except Exception:
            perm, mapping_source = hungarian_atom_mapping(ref_symbols, ref_coords, model_symbols, model_coords)

        reordered_symbols = [model_symbols[idx] for idx in perm.tolist()]
        if reordered_symbols != ref_symbols:
            raise ValueError("Mapped atom order does not match PubChem symbol order")

        reordered_coords = model_coords[perm]
        aligned_coords, _, _, used_reflection = align_with_library_kabsch(
            reordered_coords,
            ref_coords,
            allow_reflection=USE_REFLECTIONS,
        )
        kabsch_rmsd = compute_rmsd(aligned_coords, ref_coords)
        score = rmsd_to_score(kabsch_rmsd)

        pubchem_out = aligned_root / "pubchem_reference" / f"{molecule}_pubchem_reference.xyz"
        if not pubchem_out.exists():
            pubchem_out.parent.mkdir(parents=True, exist_ok=True)
            pubchem_out.write_text(
                xyz_to_text(ref_symbols, ref_coords, comment=f"{molecule} pubchem reference"),
                encoding="utf-8",
            )

        aligned_out = aligned_root / model / f"{molecule}_{model}_aligned.xyz"
        aligned_out.parent.mkdir(parents=True, exist_ok=True)
        aligned_out.write_text(
            xyz_to_text(reordered_symbols, aligned_coords, comment=f"{molecule} {model} aligned to pubchem"),
            encoding="utf-8",
        )

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
        })
        return record
    except Exception as exc:
        record.update({"status": "failed", "note": str(exc), "score": 0.0})
        return record


def launch_viewer():
    viewer_path = BASE_DIR / VIEWER_FILENAME
    if not viewer_path.exists():
        return
    try:
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(viewer_path)])
        print(f"Launched viewer: python -m streamlit run {viewer_path.name}")
    except Exception as exc:
        print(f"Could not launch viewer automatically: {exc}")


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
            plt.figure(figsize=(8, 5))
            plt.hist(vals, bins=bins, edgecolor="black")
            plt.xlim(0, HIST_XMAX)
            plt.xlabel("RMSD")
            plt.ylabel("Count")
            plt.title(f"RMSD Distribution - {model}{suffix}")
            plt.tight_layout()
            savefig_all(out_dir, f"hist_rmsd_{model}.png")
            plt.close()

            zoom_vals = vals[vals <= ZOOM_XMAX]
            if len(zoom_vals) > 0:
                plt.figure(figsize=(8, 5))
                plt.hist(zoom_vals, bins=zoom_bins, edgecolor="black")
                plt.xlim(0, ZOOM_XMAX)
                plt.xlabel("RMSD")
                plt.ylabel("Count")
                plt.title(f"RMSD Distribution (0-{ZOOM_XMAX}) - {model}{suffix}")
                plt.tight_layout()
                savefig_all(out_dir, f"hist_rmsd_zoom_{model}.png")
                plt.close()

        score_vals = df.loc[df["model"] == model, "score"].dropna().values
        if len(score_vals) > 0:
            plt.figure(figsize=(8, 5))
            plt.hist(score_vals, bins=score_bins, edgecolor="black")
            plt.xlim(0, 1.0)
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.title(f"Score Distribution - {model}{suffix}")
            plt.tight_layout()
            savefig_all(out_dir, f"hist_score_{model}.png")
            plt.close()

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
    plt.title(f"Atom Count vs RMSD with Linear Regression{suffix}")
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
    if not threshold_all_runs_df.empty:
        threshold_all_runs_df.to_csv(OUT_DIR / "per_run_threshold_summary.csv", index=False)

    run_averaged_summary = make_run_level_summary_outputs(run_summary_df, OUT_DIR)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="per_molecule_all_runs", index=False)
        overall_summary.to_excel(writer, sheet_name="overall_summary", index=False)
        run_summary_df.to_excel(writer, sheet_name="per_run_summary", index=False)
        mapping_df.to_excel(writer, sheet_name="atom_mapping", index=False)
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
