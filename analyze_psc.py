from pathlib import Path

import torch

from src.psc_dictionary import build_psc_dictionary, save_atom_visualizations, summarize_psi


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    psi = build_psc_dictionary(debug=True, debug_size=32)
    psi_summary = summarize_psi(psi)
    atom_dir = output_dir / "psi_atoms"
    atom_indices = save_atom_visualizations(
        psi,
        measure_p=32,
        measure_q=32,
        output_dir=atom_dir,
        debug=True,
        debug_size=32,
        num_atoms=5,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    sample_indices = torch.randperm(psi.shape[1], generator=generator)[:32]
    sampled_atoms = psi[:, sample_indices]
    correlation = sampled_atoms.conj().transpose(0, 1) @ sampled_atoms
    correlation_abs = torch.abs(correlation)
    off_diag_mask = ~torch.eye(correlation_abs.shape[0], dtype=torch.bool)
    off_diag_values = correlation_abs[off_diag_mask]

    summary_lines = [
        "PSC Debug Dictionary Summary",
        f"Psi shape: {psi_summary['shape']}",
        f"Psi dtype: {psi_summary['dtype']}",
        f"Psi device: {psi_summary['device']}",
        (
            "Psi column norms: "
            f"min={psi_summary['norm_min']:.6f}, "
            f"max={psi_summary['norm_max']:.6f}, "
            f"mean={psi_summary['norm_mean']:.6f}"
        ),
        f"Saved atom indices: {atom_indices}",
        (
            "Sampled atom correlation |<psi_i, psi_j>|: "
            f"mean={off_diag_values.mean().item():.6f}, "
            f"max={off_diag_values.max().item():.6f}, "
            f"median={off_diag_values.median().item():.6f}"
        ),
        "Interpretation: the debug dictionary is column-normalized and non-local; "
        "off-diagonal correlation quantifies how coherent different atoms remain.",
    ]

    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
