# PSC Engineering Reproduction

## Overview

This repository is an engineering reproduction of a PSC-style SAR imaging pipeline under a controlled debug setting.

The current codebase includes:

- image-to-complex input preprocessing from grayscale amplitude plus random phase
- a paper-parameter physical dictionary interface
- an operator-based PSC dictionary implementation with explicit-matrix analysis tools
- a lightweight PSC module placeholder with a stable training baseline
- dictionary analysis scripts for atom visualization, scaling checks, and feasibility testing

This repository is intended for research iteration and engineering diagnosis rather than for claiming a complete paper-faithful implementation.

## Status

Current repository status:

- fixed paper parameters are connected through `src/config.py`
- the physical dictionary interface is implemented and analyzed in debug mode
- the physical dictionary has been migrated from an explicit matrix pathway in training to an operator-based `Psi / Psi^H` implementation
- training runs stably in the current baseline configuration
- reconstruction is usable as a baseline, but the method is still an engineering approximation
- the current input is not true complex SAR measurement data; it is built from amplitude plus random phase

In short, this is a stable experimental baseline for further PSC reproduction work.

## Usage

Install dependencies:

```bash
D:\anaconda\envs\for_codex\python.exe -m pip install -r requirements.txt
```

Run the stable baseline training script:

```bash
D:\anaconda\envs\for_codex\python.exe train_psc.py
```

Run dictionary analysis:

```bash
D:\anaconda\envs\for_codex\python.exe analyze_dictionary.py
```

Optional auxiliary analysis:

```bash
D:\anaconda\envs\for_codex\python.exe analyze_psc.py
```

Representative outputs are stored under `outputs/` during experiments, and selected examples are copied to `assets/` for documentation.

## Notes

- This project is an engineering reproduction, not a full paper reproduction.
- The current PSC update and reconstruction flow still use engineering approximations.
- The current dictionary normally runs in debug mode, which keeps the system practical for diagnosis and iteration.
- The training path now uses operator-form `Psi` application instead of storing the full explicit matrix, which is intended to make full-scale PSC more practical later.
- The operator now caches geometry and sampling terms at initialization, including coordinate grids, frequency samples, `cos(phi)`, `sin(phi)`, wavenumbers, phase projections, and chunk index metadata.
- The latest optimization round additionally caches chunk-level atom blocks, which substantially reduces repeated forward / adjoint cost in full `80x80` mode.
- After this optimization round, full `80x80` operator timing improved markedly in the current CPU tests:
  - `100x forward` average: from about `0.226s` to about `0.013s`
  - `100x adjoint` average: from about `0.072s` to about `0.014s`
  - `50x forward+adjoint` pair average: from about `0.144s` to about `0.026s`
- The current main remaining bottleneck is no longer repeated phase construction on every call, but the memory cost of caching chunk atom blocks and the remaining dense complex matrix multiplications.
- To move toward full `80x80`, the next likely bottleneck is numerical efficiency and memory behavior rather than repository structure.
- The current stable baseline is useful for comparing future changes in reconstruction quality, dictionary scaling, and sparse-scatter behavior.
- A full `80x80` baseline smoke test has already completed successfully with batch size `1`, a tiny subset, and short training.
- In that smoke test, the full baseline finished stably without NaN or memory exceptions, and the short-run loss decreased slightly.
- Based on the current smoke-test result and the latest operator speedup, the codebase is now much closer to being ready for longer full baseline training attempts.
