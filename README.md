# PSC Engineering Reproduction

## Overview

This repository is an engineering reproduction of a PSC-style SAR imaging pipeline under a controlled debug setting.

The current codebase includes:

- image-to-complex input preprocessing from grayscale amplitude plus random phase
- a paper-parameter physical dictionary interface
- a lightweight PSC module placeholder with a stable training baseline
- dictionary analysis scripts for atom visualization, scaling checks, and feasibility testing

This repository is intended for research iteration and engineering diagnosis rather than for claiming a complete paper-faithful implementation.

## Status

Current repository status:

- fixed paper parameters are connected through `src/config.py`
- the physical dictionary interface is implemented and analyzed in debug mode
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
- To move toward full `80x80`, the next likely bottleneck is numerical efficiency and memory behavior rather than repository structure.
- The current stable baseline is useful for comparing future changes in reconstruction quality, dictionary scaling, and sparse-scatter behavior.
