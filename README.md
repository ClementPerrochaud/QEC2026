Tools for modelling **quantum error correction (QEC) on a photonic quantum computer** — from the abstract decoding performance of stabilizer codes down to the real-world power and energy cost of running them on cryogenic photonic hardware.

## Files

### `QEC.py` — QEC core library
The engine. Represents Pauli operators as base-4 integers and provides:
- **Stabilizer codes** (`QEC_code`): syndromes, logical operators, and a catalogue of built-in codes (5-qubit `LMPZ`, Steane-like `SQSC`, Reed–Muller, and arbitrary-distance `rotated_surface_code(d)`).
- **Noise model**: single-qubit Pauli error probabilities `(pI, pX, pY, pZ)` derived from a Lindblad decay/dephasing model (`get_IXYZ_error_proba`).
- **Decoders**: exact/approximate maximum-likelihood (`MLDecoder`) and minimum-weight perfect matching (`MWPMDecoder`).
- **Threshold analysis**: fits Monte-Carlo curves to the QEC threshold-theorem power law `pL = A·(p/p_th)^(B(d+1)/2)` (`fit_threshold_mc`).

### `run.py` — simulations & plots
Drives `QEC.py` to measure code performance and produce figures:
- **Brute force** (`single_/multiple_brute_force`): exhaustively enumerates errors to get exact logical-error probabilities, entropies, and correction efficiency; saves to `saves_bf/`.
- **Monte Carlo** (`monte_carlo`, `monte_carlo_rsc`): samples random errors, decodes them, and estimates logical-error rates for surface codes of increasing distance; saves to `saves_mc/`.
- **Plotting** (`bf_plotplot`, `plot_ml`, `plot_threshold_fit`, `plot_rsc_compare`, …): logical-error-vs-physical-error curves, efficiencies, and threshold crossings.

### `mnr.py` + `mnr_files/` — hardware energy model
Estimates the **power and energy cost** of implementing a surface code on the SPOQC photonic architecture:
- `params.py` — physical constants and hardware defaults (cryostat temperature, cooling power, source brightness, component losses, …).
- `functions.py` — quantum-dot source physics, cryogenic power models, component counts per code distance, and `energy_ecc(d, T)` (energy per error-correction cycle). It links the QEC threshold fit (`logical_error`) to the temperature-dependent photon noise.
- `mnr.py` — computes power/energy for a chosen code distance and renders heatmaps of a logical-error "metric" and its energy cost over the (code distance `d`, cryostat temperature `T`) plane (saved to `mnr_files/`).

## Data
`saves_bf/` and `saves_mc/` hold saved `.npy` simulation results and per-run diagnostic PNGs consumed by the plotting functions.

## Requirements
Python with `numpy`, `scipy`, `matplotlib`, and `networkx`.

README written by my good friend Claude :)
