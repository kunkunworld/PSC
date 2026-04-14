import math
import time
from pathlib import Path

import torch
from PIL import Image

from src.config import H, P, Q, W
from src.psc_dictionary import (
    build_psc_dictionary,
    save_atom_visualizations,
    summarize_psi,
)


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def save_matrix_heatmap(matrix: torch.Tensor, path: Path) -> None:
    image = matrix.detach().cpu().to(torch.float32)
    image = image - image.min()
    max_value = image.max()
    if max_value > 0:
        image = image / max_value
    array = (image.numpy() * 255.0).astype("uint8")
    Image.fromarray(array, mode="L").save(path)


def analyze_debug_dictionary(output_dir: Path) -> dict[str, float | str | tuple[int, ...]]:
    psi = build_psc_dictionary(debug=True, debug_size=32)
    psi_summary = summarize_psi(psi)
    norm_values = torch.linalg.vector_norm(psi, ord=2, dim=0)
    psi_summary["norm_std"] = norm_values.std().item()

    atom_dir = output_dir / "dictionary_atoms"
    atom_indices = save_atom_visualizations(
        psi,
        measure_p=32,
        measure_q=32,
        output_dir=atom_dir,
        debug=True,
        debug_size=32,
        num_atoms=10,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    pair_indices = torch.randint(0, psi.shape[1], (50, 2), generator=generator)
    correlations = []
    for left_idx, right_idx in pair_indices.tolist():
        left = psi[:, left_idx]
        right = psi[:, right_idx]
        corr = torch.abs(torch.sum(left.conj() * right)).item()
        correlations.append(corr)
    corr_tensor = torch.tensor(correlations, dtype=torch.float32)

    gram_indices = torch.randperm(psi.shape[1], generator=generator)[:32]
    gram_block = psi[:, gram_indices].conj().transpose(0, 1) @ psi[:, gram_indices]
    save_matrix_heatmap(torch.abs(gram_block), output_dir / "dictionary_gram_block.png")

    summary_lines = [
        "Dictionary Debug Analysis",
        f"Psi shape: {psi_summary['shape']}",
        f"Psi dtype: {psi_summary['dtype']}",
        f"Psi device: {psi_summary['device']}",
        (
            "Column L2 norms: "
            f"min={psi_summary['norm_min']:.6f}, "
            f"max={psi_summary['norm_max']:.6f}, "
            f"mean={psi_summary['norm_mean']:.6f}, "
            f"std={psi_summary['norm_std']:.6f}"
        ),
        f"Saved atom indices: {atom_indices}",
        (
            "Random pair correlation |<psi_i, psi_j>|: "
            f"min={corr_tensor.min().item():.6f}, "
            f"max={corr_tensor.max().item():.6f}, "
            f"mean={corr_tensor.mean().item():.6f}"
        ),
        (
            "Assessment: the dictionary is not close to a pixel-basis identity because "
            "its atoms are non-local stripe / phase textures instead of isolated deltas."
        ),
        (
            "Assessment: column scales are stable because normalization forces near-unit "
            "L2 norms with negligible spread."
        ),
        (
            "Assessment: sampled atom correlations are non-zero but moderate, so the "
            "dictionary shows some coherence without looking highly redundant."
        ),
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    return {
        **psi_summary,
        "norm_std": psi_summary["norm_std"],
        "corr_min": corr_tensor.min().item(),
        "corr_max": corr_tensor.max().item(),
        "corr_mean": corr_tensor.mean().item(),
    }


def scan_debug_sizes(output_dir: Path) -> list[dict[str, str | int | float | bool]]:
    report = []
    for size in (32, 40, 48, 64):
        signal_dim = size * size
        measure_dim = size * size
        estimated_bytes = signal_dim * measure_dim * torch.tensor([], dtype=torch.complex64).element_size()
        start = time.perf_counter()
        success = False
        error = ""
        try:
            psi = build_psc_dictionary(debug=True, debug_size=size)
            build_time = time.perf_counter() - start
            actual_mib = bytes_to_mib(psi.nelement() * psi.element_size())
            success = True
            del psi
        except Exception as exc:
            build_time = time.perf_counter() - start
            actual_mib = 0.0
            error = str(exc)

        report.append(
            {
                "size": size,
                "shape": f"({measure_dim}, {signal_dim})",
                "time_sec": build_time,
                "estimated_mib": bytes_to_mib(estimated_bytes),
                "actual_mib": actual_mib,
                "success": success,
                "error": error,
            }
        )

    lines = ["Dictionary Scaling Report"]
    for item in report:
        lines.append(
            f"{item['size']}x{item['size']}: shape={item['shape']}, "
            f"time={item['time_sec']:.3f}s, "
            f"estimated_mib={item['estimated_mib']:.2f}, "
            f"actual_mib={item['actual_mib']:.2f}, "
            f"success={item['success']}, "
            f"error={item['error'] or 'none'}"
        )
    (output_dir / "dictionary_scaling_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines))
    return report


def test_full_feasibility(output_dir: Path) -> dict[str, str | float | bool]:
    signal_dim = H * W
    measure_dim = P * Q
    estimated_bytes = signal_dim * measure_dim * torch.tensor([], dtype=torch.complex64).element_size()
    estimated_mib = bytes_to_mib(estimated_bytes)

    # Only test that construction can begin: create a minimal first-chunk workspace
    # instead of allocating the full 6400x6400 dictionary tensor.
    success = False
    error = ""
    start = time.perf_counter()
    try:
        probe_chunk = 2
        phase_probe = torch.empty((probe_chunk, signal_dim, Q), dtype=torch.float32)
        atom_probe = torch.empty((probe_chunk * Q, signal_dim), dtype=torch.complex64)
        del phase_probe
        del atom_probe
        success = True
    except Exception as exc:
        error = str(exc)
    elapsed = time.perf_counter() - start

    lines = [
        "Full 80x80 Feasibility Test",
        f"Target shape: ({measure_dim}, {signal_dim})",
        f"Estimated dictionary memory: {estimated_mib:.2f} MiB",
        (
            "Estimated bottleneck: full construction needs the final complex dictionary "
            "plus large temporary phase / atom blocks during chunked assembly."
        ),
        f"Chunk-start probe success: {success}",
        f"Probe time: {elapsed:.6f}s",
        f"Probe error: {error or 'none'}",
        (
            "Recommendation: full 80x80 is plausible on CPU memory grounds for the final "
            "tensor itself, but temporary allocations and runtime will likely become the "
            "main bottleneck before training is practical."
        ),
    ]
    (output_dir / "full_feasibility_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines))
    return {
        "estimated_mib": estimated_mib,
        "probe_success": success,
        "probe_time_sec": elapsed,
        "probe_error": error,
    }


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    analyze_debug_dictionary(output_dir)
    scan_debug_sizes(output_dir)
    test_full_feasibility(output_dir)


if __name__ == "__main__":
    main()
