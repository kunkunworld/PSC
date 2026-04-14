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
- The current implementation is still an engineering approximation rather than a full paper-faithful HQS solver.
- This version is the stable baseline for follow-up experiments and comparisons.
- The module keeps the same input/output interface so the training script can run end to end.

## Input Approximation

- The current input image is not real complex SAR data.
- `src/dataset.py` reads image amplitude from `data/`, converts it to grayscale, center-crops it to `80x80`, and normalizes it to `[0, 1]`.
- The complex input is constructed as `amplitude * exp(1j * phase)` with random phase sampled from `Uniform(0, 2*pi)`.
- This is an engineering approximation used to keep the pipeline runnable before real SAR complex measurements are connected.
- In other words, the current input is an `amplitude + random phase` approximation of complex SAR rather than true measured complex SAR.

## Debug And Full Dictionary Modes

- The dictionary builder supports a debug mode to avoid allocating the full `6400 x 6400` matrix during early development.
- Debug mode uses a reduced `32 x 32` signal and measurement grid while preserving the same paper-parameter interface.
- Full mode uses `H = W = P = Q = 80`, which produces the full `6400 x 6400` complex dictionary.
- To switch from debug mode to full mode, construct `PSCModule(dictionary_debug=False)` or call `build_psc_dictionary(debug=False)`.
- The current baseline keeps the dictionary in debug mode for reproducibility and faster diagnosis.

## Current Debug Validation

- The current debug dictionary size is `32 x 32`, so `Psi` has shape `[1024, 1024]`.
- During training, the stable baseline saves `input.png`, `recon_raw.png`, `recon_norm.png`, `o_norm.png`, and `o_hist.png` to `outputs/`.
- Random atom visualizations are saved to `outputs/psi_atoms/`.
- `analyze_psc.py` provides a separate, non-training diagnostic path that saves atom visualizations again, computes sample inter-atom correlation statistics, and writes `outputs/summary.txt`.
- The current version has already connected the fixed paper parameters and the physical dictionary interface, but it is still a baseline engineering implementation rather than the complete paper HQS pipeline.

## Dictionary Analysis

- `analyze_dictionary.py` is the dedicated dictionary-quality analysis entry point; it does not change the training flow.
- In the current debug `32 x 32` run, `Psi` has shape `(1024, 1024)`, dtype `torch.complex64`, and device `cpu`.
- Column L2 norms are extremely stable after normalization: `min=1.000000`, `max=1.000000`, `mean=1.000000`, `std=0.000000`.
- Random-pair atom correlation statistics in the current debug analysis are low to moderate: `min=0.000299`, `max=0.122156`, `mean=0.009823`.
- The saved atom visualizations show non-local diagonal stripe / phase-texture responses, so the current dictionary does not behave like a near-identity pixel basis.
- The current debug analysis does not indicate severe column-scale instability or extreme dictionary redundancy.

## Scaling Results

- Debug dictionary construction succeeds for all tested sizes: `32 x 32`, `40 x 40`, `48 x 48`, and `64 x 64`.
- Measured dictionary tensor sizes are approximately:
  - `32 x 32`: `8.00 MiB`
  - `40 x 40`: `19.53 MiB`
  - `48 x 48`: `40.50 MiB`
  - `64 x 64`: `128.00 MiB`
- Measured construction times stay short up to `64 x 64` in the current CPU debug setting.
- A full `80 x 80` dictionary would have shape `(6400, 6400)` and an estimated final tensor size of about `312.50 MiB`.
- Full `80 x 80` looks plausible for the final tensor itself, but temporary phase / atom buffers during construction are the more important bottleneck.

## Current Conclusion

- The current dictionary quality looks healthy enough to keep using as the stable debug baseline.
- The most likely next bottleneck is not the conceptual physics form of `Psi`, but the numerical implementation cost of constructing and using larger dictionaries efficiently.
- Based on the current results, the safer next step is to optimize the numerical implementation and memory behavior before switching the main workflow to full `80 x 80`.
