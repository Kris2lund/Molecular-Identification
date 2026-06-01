from pathlib import Path
from typing import Dict

import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components

if st.button("Reload latest files"):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.rerun()

BASE_DIR = Path(__file__).resolve().parent

MODEL_COLORS = {
    "gemini": "red",
    "openai": "green",
    "claude": "orange",
}


def get_latest_run_dir() -> Path | None:
    runs_dir = BASE_DIR / "runs"
    if not runs_dir.exists():
        return None

    run_dirs = []
    for p in runs_dir.iterdir():
        if p.is_dir() and p.name.startswith("run_"):
            try:
                run_no = int(p.name.split("_")[1])
                run_dirs.append((run_no, p))
            except Exception:
                continue

    if not run_dirs:
        return None

    run_dirs.sort(key=lambda x: x[0])
    return run_dirs[-1][1]


LATEST_RUN_DIR = get_latest_run_dir()

if LATEST_RUN_DIR is not None:
    RESULTS_DIR = LATEST_RUN_DIR / "results"
    OUT_DIR = LATEST_RUN_DIR / "combined_rmsd_analysis"
else:
    RESULTS_DIR = BASE_DIR / "results"
    OUT_DIR = BASE_DIR / "combined_rmsd_analysis"

ALIGNED_ROOT = RESULTS_DIR / "aligned_xyz"

STRUCT_GEMINI = ALIGNED_ROOT / "gemini"
STRUCT_OPENAI = ALIGNED_ROOT / "openai"
STRUCT_CLAUDE = ALIGNED_ROOT / "claude"
STRUCT_PUBCHEM = ALIGNED_ROOT / "pubchem_reference"

RESULTS_CSV = RESULTS_DIR / "rmsd_results_parallel.csv"
XLSX_PATH = RESULTS_DIR / "all_models_vs_pubchem.xlsx"


def scan_common_molecules() -> Dict[str, Dict[str, Path]]:
    molecules: Dict[str, Dict[str, Path]] = {}

    for f in STRUCT_PUBCHEM.glob("*.xyz"):
        name = f.stem
        if name.endswith("_pubchem_reference"):
            base = name[:-len("_pubchem_reference")]
            molecules.setdefault(base, {})["pubchem"] = f
        elif name.endswith("_pubchem"):
            base = name[:-len("_pubchem")]
            molecules.setdefault(base, {})["pubchem"] = f

    for model_name, model_dir in [
        ("gemini", STRUCT_GEMINI),
        ("openai", STRUCT_OPENAI),
        ("claude", STRUCT_CLAUDE),
    ]:
        for f in model_dir.glob("*.xyz"):
            name = f.stem
            for suffix in [f"_{model_name}_aligned", f"_{model_name}_fitted_to_pubchem"]:
                if name.endswith(suffix):
                    base = name[:-len(suffix)]
                    molecules.setdefault(base, {})[model_name] = f
                    break

    filtered = {}
    for name, files in molecules.items():
        if "pubchem" in files and any(model in files for model in ["gemini", "openai", "claude"]):
            filtered[name] = files

    return dict(sorted(filtered.items()))


def read_xyz(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8")


def xyz_atom_count(xyz_text: str) -> int:
    lines = [line.strip() for line in xyz_text.splitlines() if line.strip()]
    if not lines:
        return 0
    try:
        return int(lines[0])
    except Exception:
        return max(0, len(lines) - 2)


def load_scores() -> pd.DataFrame:
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame()


def get_model_metrics(df_scores: pd.DataFrame, molecule: str, model: str):
    if df_scores.empty or "molecule" not in df_scores.columns or "model" not in df_scores.columns:
        return None, None, None

    row = df_scores[(df_scores["molecule"] == molecule) & (df_scores["model"] == model)]
    if row.empty:
        return None, None, None

    row = row.iloc[0]
    rmsd = row["rmsd"] if pd.notna(row.get("rmsd")) and row.get("rmsd") != "" else None
    score = row["score"] if pd.notna(row.get("score")) else None
    status = row["status"] if pd.notna(row.get("status")) else None
    return rmsd, score, status


def make_single_viewer(xyz_text: str, style: str = "stick", spin: bool = False):
    view = py3Dmol.view(width=500, height=500)
    view.addModel(xyz_text, "xyz")
    if style == "sphere":
        view.setStyle({"sphere": {"scale": 0.3}})
    elif style == "line":
        view.setStyle({"line": {}})
    else:
        view.setStyle({"stick": {}})
    view.addPropertyLabels(
        "index",
        "",
        {
            "fontSize": 10,
            "fontColor": "black",
            "showBackground": True,
            "backgroundColor": "white",
            "backgroundOpacity": 0.6,
        },
    )
    view.zoomTo()
    if spin:
        view.spin(True)
    return view._make_html()


def make_overlay_viewer(pubchem_xyz: str, model_xyz: str, model_name: str, spin: bool = False):
    model_color = MODEL_COLORS.get(model_name, "red")
    view = py3Dmol.view(width=900, height=650)
    view.addModel(pubchem_xyz, "xyz")
    view.setStyle({"model": 0}, {"stick": {"color": "blue", "radius": 0.18}, "sphere": {"color": "blue", "scale": 0.22}})
    view.addModel(model_xyz, "xyz")
    view.setStyle({"model": 1}, {"stick": {"color": model_color, "radius": 0.12}, "sphere": {"color": model_color, "scale": 0.16}})
    view.addPropertyLabels(
        "index",
        "",
        {
            "fontSize": 10,
            "fontColor": "black",
            "showBackground": True,
            "backgroundColor": "white",
            "backgroundOpacity": 0.6,
        },
    )
    view.zoomTo()
    if spin:
        view.spin(True)
    return view._make_html()


st.set_page_config(page_title="Aligned Molecule Overlay Viewer", layout="wide")
st.title("LLM vs PubChem Molecule Viewer")

if LATEST_RUN_DIR is not None:
    st.caption(f"Viewer is reading the latest run folder: {LATEST_RUN_DIR.name}")
else:
    st.caption("Viewer is reading root results because no run folders were found.")

molecule_map = scan_common_molecules()
df_scores = load_scores()

if not molecule_map:
    st.error("Ingen aligned filer blev fundet i den nyeste run-mappe.")
    st.write(f"Checked aligned root: `{ALIGNED_ROOT}`")
    st.write(f"Checked results CSV: `{RESULTS_CSV}`")
    st.stop()

molecule_names = list(molecule_map.keys())
selected_molecule = st.selectbox("Vælg et molekyle", molecule_names)
available_files = molecule_map[selected_molecule]

available_models = [m for m in ["gemini", "openai", "claude"] if m in available_files]
model_choice = st.selectbox("Vælg model til sammenligning med PubChem", available_models)
style_choice = st.selectbox("Vælg enkelt-visning", ["stick", "sphere", "line"])
spin_choice = st.checkbox("Auto-rotation", value=False)

pubchem_xyz = read_xyz(available_files["pubchem"])
model_xyz = read_xyz(available_files[model_choice])
rmsd_val, score_val, status_val = get_model_metrics(df_scores, selected_molecule, model_choice)
pubchem_atom_count = xyz_atom_count(pubchem_xyz)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", model_choice.capitalize())
c2.metric("Atoms", str(pubchem_atom_count))
c3.metric("RMSD", f"{float(rmsd_val):.4f}" if rmsd_val is not None else "N/A")
c4.metric("Score", f"{float(score_val):.4f}" if score_val is not None else (status_val or "N/A"))

st.subheader(f"Overlay: PubChem + {model_choice.capitalize()} for {selected_molecule}")
st.caption("PubChem = blå, model = farvet overlay")
components.html(make_overlay_viewer(pubchem_xyz, model_xyz, model_choice, spin_choice), height=670)

st.divider()
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"{model_choice.capitalize()} aligned")
    components.html(make_single_viewer(model_xyz, style_choice, spin_choice), height=520)
with col2:
    st.subheader("PubChem reference")
    components.html(make_single_viewer(pubchem_xyz, style_choice, spin_choice), height=520)

if not df_scores.empty:
    rows = df_scores[df_scores["molecule"] == selected_molecule].copy()
    if not rows.empty:
        st.divider()
        st.subheader("Alle modeller for dette molekyle")
        visible_cols = [col for col in ["run", "model", "rmsd", "score", "atom_count", "status", "note", "mapping_source"] if col in rows.columns]
        st.dataframe(rows[visible_cols], use_container_width=True)

if XLSX_PATH.exists():
    try:
        perf_df = pd.read_excel(XLSX_PATH, sheet_name="overall_summary")
    except Exception:
        perf_df = pd.DataFrame()
    if not perf_df.empty:
        st.divider()
        st.subheader("Excel performance summary")
        st.dataframe(perf_df, use_container_width=True)

with st.expander("Vis xyz-filer"):
    st.markdown(f"**{model_choice.capitalize()} aligned xyz**")
    st.code(model_xyz, language="text")
    st.markdown("**PubChem reference xyz**")
    st.code(pubchem_xyz, language="text")
