# Minimal Python Project

This is a minimal Python project scaffold.

## Structure

- `requirements.txt`
- `src/`
- `train_psc.py`

## Run

```bash
python train_psc.py
```

## Simplified PSC Assumptions

- `PSCModule` is implemented as a two-stage HQS-unrolling skeleton.
- Each stage has learnable `t`, `rho`, and `mu` parameters.
- The current implementation uses the fixed paper parameters from the supplementary material: `fc = 9.599e9`, `B = 0.591e9`, `phi_syn_deg = 2.3`, `c = 3e8`, `xinterval = 0.202`, `yinterval = 0.202`, `H = W = P = Q = 80`.
- `src/psc_dictionary.py` exposes a paper-parameter dictionary construction interface and builds `Psi` with shape `[P*Q, H*W]`.
- The current HQS update is still an engineering approximation rather than a full complex-valued PSC solver.
- The module keeps the same input/output interface so the training script can run end to end.

## Input Approximation

- The current input image is not real complex SAR data.
- `src/dataset.py` reads image amplitude from `data/`, converts it to grayscale, center-crops it to `80x80`, and normalizes it to `[0, 1]`.
- The complex input is constructed as `amplitude * exp(1j * phase)` with random phase sampled from `Uniform(0, 2*pi)`.
- This is an engineering approximation used to keep the pipeline runnable before real SAR complex measurements are connected.

## Debug And Full Dictionary Modes

- The dictionary builder supports a debug mode to avoid allocating the full `6400 x 6400` matrix during early development.
- Debug mode uses a reduced `32 x 32` signal and measurement grid while preserving the same paper-parameter interface.
- Full mode uses `H = W = P = Q = 80`, which produces the full `6400 x 6400` complex dictionary.
- To switch from debug mode to full mode, construct `PSCModule(dictionary_debug=False)` or call `build_psc_dictionary(debug=False)`.

## Current Debug Validation

- The current debug dictionary size is `32 x 32`, so `Psi` has shape `[1024, 1024]`.
- During training, the script saves input amplitude, reconstruction amplitude, `o` heatmap, and `p` heatmap for each epoch in `outputs/`.
- The current reconstruction can look visually dark even when structure is present, because the raw amplitude may occupy a compressed numeric range; `recon_norm.png` is saved to diagnose this display-scale effect.
- The current raw reconstruction is mainly dark because of dynamic-range compression rather than immediate reconstruction collapse.
- After min-max normalization, `recon_norm.png` usually reveals that structural content is still present.
- The current optimization focus has shifted from "can the model reconstruct anything at all" to "can the latent coefficients evolve into truly sparse scatterer points".
- The current `o` output should not yet be interpreted as a true sparse-scatterer map; in many runs it reflects overall amplitude shrinkage more than isolated physically meaningful bright points.
- At this stage, `p` comes from a simplified HQS-style approximation, so any sparsity-like pattern should be treated only as a debugging signal rather than a paper-level reconstruction result.
