# Failed GPAW Runs Report

This report summarizes the failed runs from the `deepseek`, `llama`, and `mistral` submissions on this cluster.

Reason extraction:
- Failure reason was taken from each run's `gpaw_stdout.log`.
- The `SCF runs completed before failure` count was taken from `gpaw.log` by counting completed GPAW SCF sections, i.e. occurrences of `Converged after N iterations.`
- The geometry defect column was determined by comparing all pairwise interatomic distances in the LLM-generated `.xyz` file against the corresponding `pubchem_reference` `.xyz` file. All pubchem reference runs converged successfully.

## Summary

- Total failed runs: `32`
- All `32` failures ended with `gpaw.KohnShamConvergenceError: Did not converge!  See text output for help.`
- All `32` failures are attributable to **errors in the LLM-generated molecular geometry**. The AI models placed atoms at unphysical positions (atom clashes, compressed bonds, wrong connectivity), making the Kohn–Sham SCF equations unsolvable regardless of the number of iterations.

## Deepseek

| Molecule | Reason for failure | SCF runs completed before failure | Geometry defect in LLM-generated XYZ (vs pubchem) |
| --- | --- | ---: | --- |
| `1-octanol` | `gpaw.KohnShamConvergenceError` | 0 | C–H at 0.59 Å (expected ~1.09 Å): one H atom placed almost on top of a C atom |
| `1_3-butadiene` | `gpaw.KohnShamConvergenceError` | 1 | Three C–C pairs at 0.89 Å (expected 1.34–1.54 Å): multiple carbon atoms nearly overlapping |
| `butyric_acid` | `gpaw.KohnShamConvergenceError` | 0 | All C–H bonds compressed to ~1.00–1.04 Å (pubchem: 1.094 Å): uniformly scaled-down structure |
| `caproic_acid` | `gpaw.KohnShamConvergenceError` | 0 | H–H at 0.55 Å and O–H at 0.59 Å: severe multi-atom clash in the acid/chain region |
| `chloroethane` | `gpaw.KohnShamConvergenceError` | 1 | Cl–C at 1.27 Å (expected ~1.77 Å): Cl atom placed far too close to C |
| `diethyl_ether` | `gpaw.KohnShamConvergenceError` | 0 | Four C–H bonds at 0.88 Å (expected ~1.09 Å): entire methyl group compressed |
| `ethanol` | `gpaw.KohnShamConvergenceError` | 0 | C–H bonds compressed to ~1.02–1.03 Å (pubchem: 1.094 Å); non-bonded H–H at 1.00 Å (pubchem: ~1.79 Å) |
| `ethyl_acetate` | `gpaw.KohnShamConvergenceError` | 4 | Two C–H pairs at 0.00 Å: two distinct atoms placed at exactly the same coordinates (degenerate positions) |
| `furan` | `gpaw.KohnShamConvergenceError` | 0 | C–H at 0.93 Å (expected ~1.08 Å), aromatic C–O at 1.24 Å (pubchem: 1.36 Å): ring bonds compressed throughout |
| `hexane` | `gpaw.KohnShamConvergenceError` | 0 | H–H at 0.51 Å (expected non-bonded ≥2.0 Å): two H atoms on adjacent chain carbons essentially overlapping |
| `isopropanol` | `gpaw.KohnShamConvergenceError` | 0 | Five close contacts: C–H at 0.70–0.87 Å, O–H at 0.76 Å — widespread atom clashes throughout the molecule |
| `morpholine` | `gpaw.KohnShamConvergenceError` | 8 | All C–H and N–H bonds at a uniform 1.052 Å (pubchem: 1.019–1.095 Å): uniformly compressed, likely a coordinate-unit error |
| `naphthalene` | `gpaw.KohnShamConvergenceError` | 0 | Two H–H non-bonded pairs at 1.28 Å (expected ≥2.4 Å for ortho aromatic H): H atoms forced into each other's space; all C–C at exactly 1.42 Å (over-idealized ring) |
| `propionic_acid` | `gpaw.KohnShamConvergenceError` | 1 | O–H at 0.54 Å and O–C at 1.11 Å (expected ~1.43 Å): carboxyl group severely compressed |
| `styrene` | `gpaw.KohnShamConvergenceError` | 0 | C–H at 0.61 Å, C–C at 1.22 Å (expected 1.34–1.54 Å): clashes in vinyl/aromatic junction |
| `undecane` | `gpaw.KohnShamConvergenceError` | 0 | H–H at 0.46 Å, multiple C–C at 1.18 Å (expected 1.54 Å): worst-case failure — severely folded chain with many simultaneous clashes |
| `valeric_acid` | `gpaw.KohnShamConvergenceError` | 0 | O–H at 0.85 Å (expected ~0.98 Å), C–H at 0.89 Å: both the hydroxyl and adjacent C–H bonds compressed |

## Llama

| Molecule | Reason for failure | SCF runs completed before failure | Geometry defect in LLM-generated XYZ (vs pubchem) |
| --- | --- | ---: | --- |
| `1-butanol` | `gpaw.KohnShamConvergenceError` | 0 | Two O–H at 0.75 Å: only one O–H should exist; O is sandwiched between two H atoms at half the expected bond length |
| `1-pentanol` | `gpaw.KohnShamConvergenceError` | 0 | Two O–H at 0.99 Å (impossible for single –OH group) and C–H compressed to 1.06 Å: O is flanked by two H atoms |
| `benzoic_acid` | `gpaw.KohnShamConvergenceError` | 0 | O–O at 0.80 Å (expected ≥2.2 Å non-bonded): the two carboxylic acid oxygens are nearly overlapping; C–H at 0.70 Å |
| `caproic_acid` | `gpaw.KohnShamConvergenceError` | 4 | O–H at 0.30 Å: extreme clash — H placed almost at the oxygen nucleus |
| `chloromethane` | `gpaw.KohnShamConvergenceError` | 0 | Cl–H at 1.58 Å (expected non-bonded ~2.4 Å): Cl atom displaced toward the methyl H atoms instead of sitting on the C–Cl axis |
| `cyclopentane` | `gpaw.KohnShamConvergenceError` | 0 | C–H at 1.64 Å (expected ~1.09 Å): all 10 H atoms misplaced far from their C atoms; ring C–C connectivity also distorted (1.44 and 1.72 Å) |
| `diethyl_ether` | `gpaw.KohnShamConvergenceError` | 0 | O–C at 0.54 Å and H–H at 0.54 Å: ether oxygen collapsed onto a carbon; simultaneous H–H clash |
| `methanol` | `gpaw.KohnShamConvergenceError` | 0 | OH hydrogen detached from O: the O–H bond is broken in the input geometry (O–H >> 1.5 Å); C–H bonds also compressed to ~1.00 Å |
| `propene` | `gpaw.KohnShamConvergenceError` | 0 | C=C at 1.20 Å (expected ~1.34 Å for a double bond, shorter than even a C≡C triple bond): double bond severely over-compressed |
| `pyridine` | `gpaw.KohnShamConvergenceError` | 1 | N–C at 0.98–0.99 Å (expected ~1.34 Å), C–C at 1.20 Å, C–H at 0.86 Å: entire aromatic ring severely compressed |
| `salicylic_acid` | `gpaw.KohnShamConvergenceError` | 0 | O–H at 0.67 Å, O–C at 0.92 Å, C–H at 0.70 Å: multiple severe clashes in both the hydroxyl and carboxyl groups |
| `urea` | `gpaw.KohnShamConvergenceError` | 0 | All 4 H atoms placed ~0.87 Å from C (not from N): in (NH₂)₂CO no C–H bonds exist, yet all H atoms are nearest-neighbor to C; N–H bonds are absent |

## Mistral

| Molecule | Reason for failure | SCF runs completed before failure | Geometry defect in LLM-generated XYZ (vs pubchem) |
| --- | --- | ---: | --- |
| `1-octanol` | `gpaw.KohnShamConvergenceError` | 0 | Three C–C bonds at 1.37 Å (expected 1.54 Å for sp3 C–C), and C–O at 1.37 Å (expected ~1.43 Å): several backbone bonds compressed, indicating a locally distorted carbon chain |
| `naphthalene` | `gpaw.KohnShamConvergenceError` | 0 | Multiple C–C at 0.71 Å and C–H at 0.71 Å: severe ring-atom clashes; distances less than half the expected values |
| `tetrahydrofuran` | `gpaw.KohnShamConvergenceError` | 2 | O placed equidistant (1.40 Å) from all four ring carbons: O is at the geometric center of the ring rather than incorporated into it; the 5-membered ring topology is completely wrong |
