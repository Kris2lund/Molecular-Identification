import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re

# ==========================================================
# SETTINGS
# ==========================================================
BASE_DIR = Path("DFT/output")
REFERENCE_MODEL = "pubchem_reference"

MODELS = [
    "claude",
    "deepseek",
    "gemini",
    "llama",
    "mistral",
    "openai",
]

METRICS = [
    "final_energy_eV",
    "homo_lumo_gap_eV",
    "wall_time_s",
    "relaxation_steps",
    "max_force_eV_per_A",
]

OUT_DIR = Path("DFT/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["claude", "gemini", "openai", "deepseek", "llama", "mistral"]

MODEL_LABELS = {
    "claude": "Claude",
    "gemini": "Gemini",
    "openai": "OpenAI",
    "deepseek": "DeepSeek",
    "llama": "Llama",
    "mistral": "Mistral",
}

MODEL_COLORS = {
    "claude": "tab:blue",
    "gemini": "tab:orange",
    "openai": "tab:green",
    "deepseek": "tab:red",
    "llama": "tab:purple",
    "mistral": "tab:brown",
}

# 10 meV = 0.010 eV
SUCCESS_THRESHOLD_EV = 0.010
SUCCESS_THRESHOLD_LABEL = "10 meV threshold"


# ==========================================================
# HELPERS
# ==========================================================
def get_atom_count_from_folder(model_name, molecule_name):
    folder = BASE_DIR / model_name / molecule_name

    if not folder.exists():
        return np.nan

    possible_logs = (
        list(folder.glob("*.txt"))
        + list(folder.glob("*.log"))
        + list(folder.glob("*.out"))
    )

    for log_file in possible_logs:
        try:
            text = log_file.read_text(errors="ignore")

            for line in text.splitlines():
                lower = line.lower()

                if "number of atoms" in lower:
                    match = re.search(r"(\d+)", line)
                    if match:
                        return int(match.group(1))

                if "atoms" in lower:
                    numbers = re.findall(r"\d+", line)
                    if numbers:
                        return int(numbers[0])

        except Exception:
            continue

    return np.nan


def read_results(model_name):
    path = BASE_DIR / model_name / "results.csv"

    if not path.exists():
        print(f"Missing file: {path}")
        return None

    df = pd.read_csv(path)
    df["source_model"] = model_name

    df["atom_count_total"] = df["molecule_name"].apply(
        lambda name: get_atom_count_from_folder(model_name, name)
    )

    return df


def style_plot(title, xlabel=None, ylabel=None):
    plt.title(title, fontsize=14, fontweight="bold")
    if xlabel:
        plt.xlabel(xlabel, fontsize=12)
    if ylabel:
        plt.ylabel(ylabel, fontsize=12)

    plt.grid(True, alpha=0.25)
    plt.tick_params(axis="both", labelsize=10)
    plt.tight_layout()


def save_plot(filename):
    out_path = PLOT_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def safe_r2(x, y):
    if len(x) < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1] ** 2

def lighten_color(color, amount=0.55):
    c = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    return c + (white - c) * amount

# ==========================================================
# READ PUBCHEM REFERENCE
# ==========================================================
pubchem = read_results(REFERENCE_MODEL)

if pubchem is None:
    raise FileNotFoundError("Could not find PubChem reference results.csv")

pubchem = pubchem[["molecule_name"] + METRICS].copy()
pubchem = pubchem.rename(columns={m: f"{m}_pubchem" for m in METRICS})


# ==========================================================
# COMPARE ALL MODELS TO PUBCHEM
# ==========================================================
all_comparisons = []

for model in MODELS:
    df = read_results(model)

    if df is None:
        continue

    keep_cols = ["molecule_name", "atom_count_total"] + METRICS + ["failure_category"]
    df = df[keep_cols].copy()

    merged = df.merge(pubchem, on="molecule_name", how="inner")

    for metric in METRICS:
        merged[f"{metric}_diff"] = merged[metric] - merged[f"{metric}_pubchem"]
        merged[f"{metric}_abs_diff"] = merged[f"{metric}_diff"].abs()

    merged["source_model"] = model

    all_comparisons.append(merged)

    out_path = OUT_DIR / f"comparison_{model}_vs_pubchem.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if not all_comparisons:
    raise ValueError("No valid model comparisons found.")

combined = pd.concat(all_comparisons, ignore_index=True)
combined.to_csv(OUT_DIR / "all_models_vs_pubchem.csv", index=False)


# ==========================================================
# SUMMARY TABLE
# ==========================================================
summary_rows = []

for model, group in combined.groupby("source_model"):
    row = {
        "source_model": model,
        "n_compared": len(group),
    }

    for metric in METRICS:
        row[f"mean_abs_diff_{metric}"] = group[f"{metric}_abs_diff"].mean()
        row[f"median_abs_diff_{metric}"] = group[f"{metric}_abs_diff"].median()
        row[f"mean_diff_{metric}"] = group[f"{metric}_diff"].mean()

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "summary_vs_pubchem.csv", index=False)

print("\nDone.")
print(summary)


# ==========================================================
# 1. SUCCESS RATE PER MODEL
# ==========================================================
success_rows = []

for model in MODEL_ORDER:
    data = combined[combined["source_model"] == model]

    if data.empty:
        continue

    success_rate = (data["final_energy_eV_abs_diff"] < SUCCESS_THRESHOLD_EV).mean() * 100

    success_rows.append({
        "model": model,
        "label": MODEL_LABELS[model],
        "success_rate": success_rate
    })

success_df = pd.DataFrame(success_rows)

plt.figure(figsize=(9, 5.8))

bars = plt.bar(
    success_df["label"],
    success_df["success_rate"],
    color=[MODEL_COLORS[m] for m in success_df["model"]],
    alpha=0.85,
    edgecolor="black",
    linewidth=0.8
)

for bar, row in zip(bars, success_df.itertuples()):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        row.success_rate + 1.5,
        f"{row.success_rate:.1f}%",
        ha="center",
        fontweight="bold",
        fontsize=10
    )

plt.ylim(0, 110)

style_plot(
    "DFT Success Rate by Model",
    "Model",
    "Success rate [%]"
)

save_plot("01_success_rate_by_model.png")

# ==========================================================
# 1B. COVERAGE-AWARE HOMO-LUMO GAP SUCCESS RATE PER MODEL
# Missing molecules count as unsuccessful
# ==========================================================
TOTAL_MOLECULES = 100

homo_coverage_rows = []

for model in MODEL_ORDER:
    data = combined[combined["source_model"] == model]

    if data.empty:
        continue

    n_compared = len(data)
    n_success = (data["homo_lumo_gap_eV_abs_diff"] < SUCCESS_THRESHOLD_EV).sum()
    n_over_threshold = n_compared - n_success
    n_missing = TOTAL_MOLECULES - n_compared

    success_rate_all = n_success / TOTAL_MOLECULES * 100

    homo_coverage_rows.append({
        "model": model,
        "label": MODEL_LABELS[model],
        "n_success": n_success,
        "n_over_threshold": n_over_threshold,
        "n_missing": n_missing,
        "success_rate_all": success_rate_all
    })

homo_coverage_df = pd.DataFrame(homo_coverage_rows)

plt.figure(figsize=(9, 5.8))

x = np.arange(len(homo_coverage_df))

for i, row in homo_coverage_df.iterrows():
    model = row["model"]
    base_color = MODEL_COLORS[model]
    light_color = lighten_color(base_color, amount=0.55)

    plt.bar(
        i,
        row["n_success"],
        color=base_color,
        edgecolor="black",
        linewidth=0.8
    )

    plt.bar(
        i,
        row["n_over_threshold"],
        bottom=row["n_success"],
        color=light_color,
        edgecolor="black",
        linewidth=0.8
    )

    plt.bar(
        i,
        row["n_missing"],
        bottom=row["n_success"] + row["n_over_threshold"],
        color="lightgray",
        edgecolor="black",
        linewidth=0.8
    )

    plt.text(
        i,
        row["n_success"] + 2,
        f"{row['success_rate_all']:.1f}%",
        ha="center",
        fontweight="bold",
        fontsize=10
    )

plt.xticks(x, homo_coverage_df["label"])
plt.ylim(0, TOTAL_MOLECULES + 10)

from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="black", edgecolor="black", label="Found and below threshold"),
    Patch(facecolor=lighten_color("black", amount=0.55), edgecolor="black", label="Found but above threshold"),
    Patch(facecolor="lightgray", edgecolor="black", label="Missing / not compared"),
]

plt.legend(handles=legend_elements, loc="upper right")

style_plot(
    "Coverage-Aware HOMO-LUMO Gap Success Rate by Model",
    "Model",
    "Number of molecules"
)

save_plot("01b_coverage_aware_homo_lumo_success_rate.png")
# ==========================================================
# 1C. COVERAGE-AWARE SUCCESS RATE PER MODEL
# Missing molecules count as unsuccessful
# ==========================================================
TOTAL_MOLECULES = 100

coverage_rows = []

for model in MODEL_ORDER:
    data = combined[combined["source_model"] == model]

    if data.empty:
        continue

    n_compared = len(data)
    n_success = (data["final_energy_eV_abs_diff"] < SUCCESS_THRESHOLD_EV).sum()
    n_over_threshold = n_compared - n_success
    n_missing = TOTAL_MOLECULES - n_compared

    success_rate_all = n_success / TOTAL_MOLECULES * 100

    coverage_rows.append({
        "model": model,
        "label": MODEL_LABELS[model],
        "n_success": n_success,
        "n_over_threshold": n_over_threshold,
        "n_missing": n_missing,
        "success_rate_all": success_rate_all
    })

coverage_df = pd.DataFrame(coverage_rows)

plt.figure(figsize=(9, 5.8))

x = np.arange(len(coverage_df))

for i, row in coverage_df.iterrows():
    model = row["model"]
    base_color = MODEL_COLORS[model]
    light_color = lighten_color(base_color, amount=0.55)

    plt.bar(
        i,
        row["n_success"],
        color=base_color,
        edgecolor="black",
        linewidth=0.8
    )

    plt.bar(
        i,
        row["n_over_threshold"],
        bottom=row["n_success"],
        color=light_color,
        edgecolor="black",
        linewidth=0.8
    )

    plt.bar(
        i,
        row["n_missing"],
        bottom=row["n_success"] + row["n_over_threshold"],
        color="lightgray",
        edgecolor="black",
        linewidth=0.8
    )

    plt.text(
        i,
        row["n_success"] + 2,
        f"{row['success_rate_all']:.1f}%",
        ha="center",
        fontweight="bold",
        fontsize=10
    )

plt.xticks(x, coverage_df["label"])
plt.ylim(0, TOTAL_MOLECULES + 10)

# Custom legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="black", edgecolor="black", label="Found and below threshold"),
    Patch(facecolor=lighten_color("black", amount=0.55), edgecolor="black", label="Found but above threshold"),
    Patch(facecolor="lightgray", edgecolor="black", label="Missing / not compared"),
]

plt.legend(handles=legend_elements, loc="upper right")

style_plot(
    "Coverage-Aware DFT Success Rate by Model",
    "Model",
    "Number of molecules"
)

save_plot("01c_coverage_aware_success_rate.png")
# ==========================================================
# 2. AVERAGE DIFFERENCE WITH ERROR BARS
# ==========================================================
def plot_average_difference(metric, ylabel, title, filename):
    rows = []

    for model in MODEL_ORDER:
        data = combined[combined["source_model"] == model]

        if data.empty:
            continue

        values = data[f"{metric}_abs_diff"].dropna()

        rows.append({
            "model": model,
            "label": MODEL_LABELS[model],
            "mean": values.mean(),
            "sem": values.sem()
        })

    df_plot = pd.DataFrame(rows)

    plt.figure(figsize=(9, 5.8))

    for i, row in df_plot.iterrows():
        color = MODEL_COLORS[row["model"]]

        plt.errorbar(
            i,
            row["mean"],
            yerr=row["sem"],
            fmt="o",
            markersize=10,
            capsize=6,
            linewidth=2,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.8
        )

        offset = 0.04 * df_plot["mean"].max() if df_plot["mean"].max() > 0 else 0.05

        plt.text(
            i,
            row["mean"] + row["sem"] + offset,
            f"{row['mean']:.3f} ± {row['sem']:.3f}",
            color=color,
            fontsize=9,
            fontweight="bold",
            ha="center"
        )

    plt.xticks(range(len(df_plot)), df_plot["label"])
    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    style_plot(title, "Model", ylabel)
    save_plot(filename)


plot_average_difference(
    "final_energy_eV",
    "Average energy difference vs PubChem [eV]",
    "Average Energy Difference by Model",
    "02_average_energy_difference.png"
)

plot_average_difference(
    "homo_lumo_gap_eV",
    "Average HOMO-LUMO gap difference vs PubChem [eV]",
    "Average HOMO-LUMO Gap Difference by Model",
    "03_average_homo_lumo_difference.png"
)

plot_average_difference(
    "relaxation_steps",
    "Average relaxation step difference vs PubChem",
    "Average Relaxation Step Difference by Model",
    "04_average_relaxation_step_difference.png"
)


# ==========================================================
# 3. ENERGY DIFFERENCE BOXPLOT
# ==========================================================
plt.figure(figsize=(9, 5.8))

box_data = []
labels = []
colors = []

for model in MODEL_ORDER:
    model_data = combined[combined["source_model"] == model]["final_energy_eV_abs_diff"].dropna()

    if model_data.empty:
        continue

    box_data.append(model_data)
    labels.append(MODEL_LABELS[model])
    colors.append(MODEL_COLORS[model])

box = plt.boxplot(
    box_data,
    labels=labels,
    patch_artist=True,
    showmeans=True,
    meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=6),
    medianprops=dict(color="black", linewidth=1.5),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2)
)

for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.50)

plt.axhline(
    SUCCESS_THRESHOLD_EV,
    color="black",
    linestyle=":",
    linewidth=2,
    label=SUCCESS_THRESHOLD_LABEL
)

plt.legend(loc="upper right")

style_plot(
    "Energy Difference Boxplot by Model",
    "Model",
    "Energy difference vs PubChem [eV]"
)

save_plot("05_energy_difference_boxplot.png")


# ==========================================================
# 4. ENERGY DIFFERENCE VIOLINPLOT
# ==========================================================
plt.figure(figsize=(9, 5.8))

parts = plt.violinplot(box_data, showmeans=True, showmedians=True)

for body, color in zip(parts["bodies"], colors):
    body.set_facecolor(color)
    body.set_edgecolor("black")
    body.set_alpha(0.50)
    body.set_linewidth(0.8)

for key in ["cbars", "cmins", "cmaxes", "cmeans", "cmedians"]:
    if key in parts:
        parts[key].set_color("black")
        parts[key].set_linewidth(1.2)

plt.xticks(range(1, len(labels) + 1), labels)

plt.axhline(
    SUCCESS_THRESHOLD_EV,
    color="black",
    linestyle=":",
    linewidth=2,
    label=SUCCESS_THRESHOLD_LABEL
)

plt.legend(loc="upper right")

style_plot(
    "Energy Difference Distribution by Model",
    "Model",
    "Energy difference vs PubChem [eV]"
)

save_plot("06_energy_difference_violinplot.png")


# ==========================================================
# 5. ENERGY DIFFERENCE VS ATOM COUNT
# ==========================================================
plt.figure(figsize=(9, 5.8))

for model in MODEL_ORDER:
    data = combined[combined["source_model"] == model].dropna(
        subset=["atom_count_total", "final_energy_eV_abs_diff"]
    )

    if data.empty:
        continue

    x = data["atom_count_total"]
    y = data["final_energy_eV_abs_diff"]

    plt.scatter(
        x,
        y,
        color=MODEL_COLORS[model],
        label="_nolegend_",
        alpha=0.55,
        s=55,
        edgecolor="black",
        linewidth=0.4
    )

    if len(data) > 1:
        coeffs = np.polyfit(x, y, 1)
        fit = np.poly1d(coeffs)

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = fit(x_line)

        r2 = safe_r2(x, y)

        if np.isnan(r2):
            label = f"{MODEL_LABELS[model]} fit"
        else:
            label = f"{MODEL_LABELS[model]} fit, $R^2$={r2:.2f}"

        plt.plot(
            x_line,
            y_line,
            color=MODEL_COLORS[model],
            linestyle="--",
            linewidth=2.4,
            label=label
        )

plt.axhline(
    SUCCESS_THRESHOLD_EV,
    color="black",
    linestyle=":",
    linewidth=2,
    label=SUCCESS_THRESHOLD_LABEL
)

plt.legend(title="Linear fits", loc="upper left")

style_plot(
    "Energy Difference vs Atom Count",
    "Atom count",
    "Energy difference vs PubChem [eV]"
)

save_plot("07_energy_difference_vs_atom_count.png")


# ==========================================================
# 6. FINAL ENERGY VS RELAXATION STEPS
# ==========================================================
plt.figure(figsize=(9, 5.8))

for model in MODEL_ORDER:
    data = combined[combined["source_model"] == model].dropna(
        subset=["relaxation_steps", "final_energy_eV"]
    )

    if data.empty:
        continue

    plt.scatter(
        data["relaxation_steps"],
        data["final_energy_eV"],
        color=MODEL_COLORS[model],
        label=MODEL_LABELS[model],
        alpha=0.60,
        s=55,
        edgecolor="black",
        linewidth=0.4
    )

plt.legend(title="Model", loc="best")

style_plot(
    "Final Energy vs Relaxation Steps",
    "Relaxation steps",
    "Final energy [eV]"
)

save_plot("08_final_energy_vs_relaxation_steps.png")


# ==========================================================
# 7. RELAXATION STEPS VS ATOM COUNT
# ==========================================================
plt.figure(figsize=(9, 5.8))

for model in MODEL_ORDER:
    data = combined[combined["source_model"] == model].dropna(
        subset=["atom_count_total", "relaxation_steps"]
    )

    if data.empty:
        continue

    x = data["atom_count_total"]
    y = data["relaxation_steps"]

    plt.scatter(
        x,
        y,
        color=MODEL_COLORS[model],
        label="_nolegend_",
        alpha=0.55,
        s=55,
        edgecolor="black",
        linewidth=0.4
    )

    if len(data) > 1:
        coeffs = np.polyfit(x, y, 1)
        fit = np.poly1d(coeffs)

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = fit(x_line)

        r2 = safe_r2(x, y)

        if np.isnan(r2):
            label = f"{MODEL_LABELS[model]} fit"
        else:
            label = f"{MODEL_LABELS[model]} fit, $R^2$={r2:.2f}"

        plt.plot(
            x_line,
            y_line,
            color=MODEL_COLORS[model],
            linestyle="--",
            linewidth=2.4,
            label=label
        )

plt.legend(title="Linear fits", loc="upper left")

style_plot(
    "Relaxation Steps vs Atom Count",
    "Atom count",
    "Relaxation steps"
)

save_plot("09_relaxation_steps_vs_atom_count.png")
# ==========================================================
# 10. RMSD VS ENERGY DIFFERENCE
# ==========================================================

RMSD_CSV = Path("results/rmsd_results_parallel.csv")

RMSD_THRESHOLD_A = 1.0   # RMSD threshold [Å]

if not RMSD_CSV.exists():
    raise FileNotFoundError(f"Could not find RMSD file: {RMSD_CSV}")

# Read RMSD file
# engine="python" is useful because your CSV has long text fields with messages
rmsd_df = pd.read_csv(RMSD_CSV, engine="python")

# Keep only successful RMSD calculations
# This removes not_found rows where RMSD is incorrectly 0
rmsd_df = rmsd_df[rmsd_df["status"] == "ok"].copy()

# Keep only the columns needed
rmsd_df = rmsd_df[["molecule", "model", "rmsd"]].copy()

# Rename columns so they match your DFT comparison dataframe
rmsd_df = rmsd_df.rename(columns={
    "molecule": "molecule_name",
    "model": "source_model"
})

# Make sure names match
rmsd_df["source_model"] = rmsd_df["source_model"].astype(str).str.lower().str.strip()
rmsd_df["molecule_name"] = rmsd_df["molecule_name"].astype(str).str.strip()

# If there are multiple RMSD values for same model/molecule, take the average
rmsd_df = (
    rmsd_df
    .dropna(subset=["rmsd"])
    .groupby(["source_model", "molecule_name"], as_index=False)["rmsd"]
    .mean()
)

# Merge RMSD with DFT energy difference
rmsd_energy = combined.merge(
    rmsd_df,
    on=["source_model", "molecule_name"],
    how="inner"
)

rmsd_energy.to_csv(
    OUT_DIR / "rmsd_vs_energy_difference.csv",
    index=False
)

print(f"Saved: {OUT_DIR / 'rmsd_vs_energy_difference.csv'}")
print(f"Number of matched RMSD + DFT molecules: {len(rmsd_energy)}")


# ==========================================================
# PLOT: RMSD VS ENERGY DIFFERENCE
# ==========================================================
plt.figure(figsize=(9, 5.8))

for model in MODEL_ORDER:
    data = rmsd_energy[
        rmsd_energy["source_model"] == model
    ].dropna(subset=["rmsd", "final_energy_eV_abs_diff"])

    if data.empty:
        continue

    x = data["rmsd"]
    y = data["final_energy_eV_abs_diff"]

    plt.scatter(
        x,
        y,
        color=MODEL_COLORS[model],
        label="_nolegend_",
        alpha=0.60,
        s=55,
        edgecolor="black",
        linewidth=0.4
    )

    if len(data) > 1:
        coeffs = np.polyfit(x, y, 1)
        fit = np.poly1d(coeffs)

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = fit(x_line)

        r2 = safe_r2(x, y)

        if np.isnan(r2):
            label = f"{MODEL_LABELS[model]} fit"
        else:
            label = f"{MODEL_LABELS[model]} fit, $R^2$={r2:.2f}"

        plt.plot(
            x_line,
            y_line,
            color=MODEL_COLORS[model],
            linestyle="--",
            linewidth=2.4,
            label=label
        )

# Energy threshold: 10 meV
plt.axhline(
    SUCCESS_THRESHOLD_EV,
    color="black",
    linestyle=":",
    linewidth=2,
    label=SUCCESS_THRESHOLD_LABEL
)

# RMSD threshold: 1 Å
plt.axvline(
    RMSD_THRESHOLD_A,
    color="gray",
    linestyle=":",
    linewidth=2,
    label="1 Å RMSD threshold"
)

plt.legend(title="Linear fits", loc="upper left")

style_plot(
    "RMSD vs Energy Difference",
    "RMSD vs PubChem [Å]",
    "Energy difference vs PubChem [eV]"
)

save_plot("10_rmsd_vs_energy_difference.png")