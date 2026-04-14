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
- The current implementation uses a simplified real-valued approximation to mimic the update structure.
- It does not implement full complex-valued optics or exact PSC physics.
- The module keeps the same input/output interface so the training script can run end to end.
