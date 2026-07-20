# QEC — surface-code error-correction sandbox

Two files:

- **`QEC.py`** — the library: quantum-error-correction primitives, the brute-force
  exact-fidelity pipeline, Monte-Carlo decoders, and the data/fit helpers. Import-only.
- **`run.py`** — the runnable side: the experiment drivers
  and all the plots. Does `from QEC import *` and calls the library.

Everything is stored as `numpy` arrays of **base-4 integer "tags"**: one integer encodes a
full n-qubit Pauli, two bits per qubit (`I=0, X=1, Z=2, Y=3`).

---

## `QEC.py`

**Codes**
- `QEC_code(name, stabilizers)` — builds generators + logical operators (`.X`, `.Z`) from a
  list of Pauli strings. Prebuilt: `code_LMPZ`, `code_SQSC`, `code_NQSC`, `code_RM15`, `code_SC13`, …
- `rotated_surface_code(d)` — distance-`d` rotated surface code (`n = d²` data qubits).

**Tags & Paulis**
- `get_tags(base, size, wmax, wmin)` — enumerate all Paulis up to weight `wmax` (the
  brute-force truncation knob).
- `get_IXYZ_error_proba(t, g, gc)` — single-qubit `(pI, pX, pY, pZ)` from the Lindblad rates.
- `symp`, `logical`, `is_stabilizer`, `get_weight`, `letterify`, `binrepr` — Pauli algebra / display.

**Brute-force exact pipeline** (given `error_tags`)
`get_proba` → `get_syndrome` → `find_syndrome_errors` → `apply_correction` (max-likelihood
per syndrome) → `get_new_proba` → `get_fidelity` / `get_logical_proba`. Plus
`get_syndrome_proba`, `get_equivalence_proba`, `entropy_form` for the entropy/efficiency diagnostics.

**Monte-Carlo decoders**
- `random_tag(n, p_unit)` — sample a random error.
- `MWPMDecoder(code, noise)` — minimum-weight-perfect-matching decoder (used by the MC runs).
- `MLDecoder` — maximum-likelihood decoder.

**Thresholds, fits & I/O helpers**
- `compute_threshold(x1, p1, x2, p2)` — crossing of two logical-error curves; the two curves
  **need not share an x-axis** (resampled by log-log interpolation over their overlap).
- `fit_threshold_mc(...)` — joint regression of the MC curves to the threshold-theorem model
  `pL = A·(p/p_th)^(B(d+1)/2)` (fit in log space = geometric loss).
- `load_mc_curve(d)` → `(p, pL, N)` from `saves_mc/data_{d}.npy`.
- `load_bf_curve(folder, suffix)` → `(p, pL, eff)` from `saves_bf/{folder}/data{suffix}.npy`.

---

## `run.py`

Experiment drivers and figures:

- `single_brute_force()` — one exact run for one code/time;
  (errors→syndromes breakdown via `plot_errors_to_syndromes`).
- `multiple_brute_force()` — sweep over noise strength; writes `saves_bf/{code}/…`.
- `monte_carlo()` / `monte_carlo_rsc(d)` — MWPM Monte-Carlo sweep; writes `saves_mc/data_{d}.npy`.
- `plot_ml`, `bf_plotplot`, `bf_compare_maxw` — assorted result plots.
- `plot_threshold_fit(...)` — overlays the `fit_threshold_mc` model on the MC data.
- `plot_rsc_compare()` — the main figure: MC vs exact/approx logical-error curves + efficiency.

Pick a function in the `if __name__ == "__main__":` block and run:

```bash
python run.py
```

---

## Data layout

- `saves_bf/{code}/data{suffix}.npy` — brute-force runs
  `[_, S_measure, _, S_equivalence, error_probas, logical_probas]`.
- `saves_mc/data_{d}.npy` — Monte-Carlo runs `[error_rates, finals]`, where `finals[i]` is the
  `[I, X, Y, Z]` logical-outcome counts at that error rate.

## Notes

- Tags use `int64` for `n ≤ 31` qubits and fall back to Python big-ints (`object` arrays) beyond
  that — so `d ≥ 7` is memory-bound; keep `wmax` low.
- `requires`: `numpy`, `scipy`, `matplotlib`, `networkx`.
- My good friend Claude wrote this README for me :)
