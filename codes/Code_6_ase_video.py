from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.visualize import view
from ase.data import covalent_radii


# ============================================================
# SETTINGS
# ============================================================

model = "gemini"  # Change this to the model you want to compare with PubChem
molecule = "leucine"  # Change this to the molecule you want to analyze

llm_traj_path = Path(f"DFT/output/{model}/{molecule}/relaxation.traj")
pubchem_traj_path = Path(f"DFT/output/pubchem_reference/{molecule}/relaxation.traj")

output_traj = Path(f"overlay_PubChem_background_{model}_animation_{molecule}.traj")
output_png = Path(f"overlay_PubChem_background_{model}_final_{molecule}_with_bonds.png")

ALIGN_LLM_TO_PUBCHEM = True
OPEN_ASE_VIEWER = True

BOND_SCALE = 1.25
MIN_BOND_DISTANCE = 0.3


# ============================================================
# FUNCTIONS
# ============================================================

def center_positions(atoms):
    atoms = atoms.copy()
    atoms.positions -= atoms.get_center_of_mass()
    return atoms


def kabsch_align(moving_atoms, reference_atoms):
    """
    Align moving_atoms onto reference_atoms using Kabsch alignment.
    Assumes same atom order and same number of atoms.
    """
    moving = center_positions(moving_atoms)
    reference = center_positions(reference_atoms)

    P = moving.get_positions()
    Q = reference.get_positions()

    if len(P) != len(Q):
        raise ValueError(
            f"Cannot align: different number of atoms. "
            f"LLM has {len(P)}, PubChem has {len(Q)}."
        )

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # Prevent reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    moving.positions = P @ R

    return moving, reference


def find_bonds(atoms, scale=1.25, min_distance=0.3):
    """
    Estimate bonds using covalent radii.
    A bond is drawn if distance < scale * (r_i + r_j).
    """
    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()

    bonds = []

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            distance = np.linalg.norm(positions[i] - positions[j])

            r_i = covalent_radii[atomic_numbers[i]]
            r_j = covalent_radii[atomic_numbers[j]]

            cutoff = scale * (r_i + r_j)

            if min_distance < distance < cutoff:
                bonds.append((i, j))

    return bonds


def plot_atoms_and_bonds(ax, atoms, atom_color, bond_color, label, alpha_atoms=1.0, alpha_bonds=1.0):
    """
    Plot atoms and estimated bonds in a 3D matplotlib plot.
    """
    positions = atoms.get_positions()

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=70,
        color=atom_color,
        alpha=alpha_atoms,
        label=label
    )

    bonds = find_bonds(
        atoms,
        scale=BOND_SCALE,
        min_distance=MIN_BOND_DISTANCE
    )

    for i, j in bonds:
        x = [positions[i, 0], positions[j, 0]]
        y = [positions[i, 1], positions[j, 1]]
        z = [positions[i, 2], positions[j, 2]]

        ax.plot(
            x, y, z,
            color=bond_color,
            alpha=alpha_bonds,
            linewidth=1.8
        )


def set_axes_equal(ax):
    """
    Make x, y, z axes have equal scale.
    This prevents the molecule from looking distorted.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def make_final_plot(pubchem, llm, output_png):
    """
    Make a clear final 3D plot with atoms and bonds.
    PubChem is shown in the background.
    OpenAI/model is shown on top.
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # PubChem in background
    plot_atoms_and_bonds(
        ax,
        pubchem,
        atom_color="gray",
        bond_color="gray",
        label="PubChem reference",
        alpha_atoms=0.35,
        alpha_bonds=0.35
    )

    # LLM/model on top
    plot_atoms_and_bonds(
        ax,
        llm,
        atom_color="red",
        bond_color="red",
        label=f"{model} final structure",
        alpha_atoms=0.95,
        alpha_bonds=0.95
    )

    ax.set_title(f"Final DFT structure overlay with bonds\nPubChem vs {model} — {molecule}")
    ax.set_xlabel("x [Å]")
    ax.set_ylabel("y [Å]")
    ax.set_zlabel("z [Å]")
    ax.legend()

    set_axes_equal(ax)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()


# ============================================================
# CHECK FILES
# ============================================================

if not llm_traj_path.exists():
    raise FileNotFoundError(f"Could not find LLM trajectory:\n{llm_traj_path}")

if not pubchem_traj_path.exists():
    raise FileNotFoundError(f"Could not find PubChem trajectory:\n{pubchem_traj_path}")


# ============================================================
# READ TRAJECTORIES
# ============================================================

print("Reading trajectories...")
print(f"LLM trajectory:     {llm_traj_path}")
print(f"PubChem trajectory: {pubchem_traj_path}")

llm_frames = read(llm_traj_path, index=":")
pubchem_last = read(pubchem_traj_path, index=-1)

print(f"Number of {model} frames: {len(llm_frames)}")
print(f"Number of PubChem atoms: {len(pubchem_last)}")
print(f"Number of {model} atoms: {len(llm_frames[-1])}")


# ============================================================
# CREATE OVERLAY TRAJECTORY
# ============================================================

combined_frames = []

for i, frame in enumerate(llm_frames):
    llm = frame.copy()
    ref = pubchem_last.copy()

    if ALIGN_LLM_TO_PUBCHEM:
        llm, ref = kabsch_align(llm, ref)
    else:
        llm = center_positions(llm)
        ref = center_positions(ref)

    combined = ref + llm

    # Tags:
    # tag 1 = PubChem atoms
    # tag 2 = OpenAI/model atoms
    combined.set_tags(
        [1] * len(ref) +
        [2] * len(llm)
    )

    combined.info["comment"] = (
        f"Frame {i}: PubChem reference = first {len(ref)} atoms, "
        f"{model} = last {len(llm)} atoms. "
        f"Tags: PubChem=1, {model}=2."
    )

    combined_frames.append(combined)


# ============================================================
# SAVE OVERLAY TRAJECTORY
# ============================================================

write(output_traj, combined_frames)

print("\nSaved overlay trajectory:")
print(output_traj)

print("\nIn the ASE structure:")
print(f"PubChem atoms = atom indices 0 to {len(pubchem_last) - 1}")
print(f"{model} atoms = atom indices {len(pubchem_last)} to {len(combined_frames[-1]) - 1}")
print("PubChem = tag 1")
print(f"{model} = tag 2")


# ============================================================
# MAKE CLEAR FINAL PNG PLOT WITH BONDS
# ============================================================

final_combined = combined_frames[-1]

n_pubchem = len(pubchem_last)

pubchem_final = final_combined[:n_pubchem]
llm_final = final_combined[n_pubchem:]

make_final_plot(pubchem_final, llm_final, output_png)

print("\nSaved final comparison figure with bonds:")
print(output_png)


# ============================================================
# OPEN ASE VIEWER
# ============================================================

if OPEN_ASE_VIEWER:
    print("\nOpening ASE viewer...")
    print("In ASE:")
    print("PubChem is the first molecule.")
    print(f"{model} is the second molecule on top.")
    print("Note: ASE may guess bonds visually, but the clearest bonded figure is the saved PNG.")
    view(combined_frames)