import time
from pathlib import Path

import torch
from PIL import Image

from src.config import H, P, Q, W
from src.psc_dictionary import build_psc_operator


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def parse_previous_metric(report_path: Path, key: str) -> float | None:
    if not report_path.exists():
        return None
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(key):
            try:
                value = line.split(":", 1)[1].strip().rstrip("s")
                return float(value)
            except Exception:
                return None
    return None


def save_complex_amplitude_raw(tensor: torch.Tensor, path: Path) -> None:
    amplitude = torch.abs(tensor.detach().cpu()).reshape(H, W).clamp(0.0, 1.0)
    image = (amplitude.numpy() * 255.0).astype("uint8")
    Image.fromarray(image, mode="L").save(path)


def save_complex_amplitude_norm(tensor: torch.Tensor, path: Path) -> None:
    amplitude = torch.abs(tensor.detach().cpu()).reshape(H, W)
    amplitude = amplitude - amplitude.min()
    max_value = amplitude.max()
    if max_value > 0:
        amplitude = amplitude / max_value
    image = (amplitude.numpy() * 255.0).astype("uint8")
    Image.fromarray(image, mode="L").save(path)


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    previous_report = output_dir / "operator_optimization_report.txt"
    prev_forward_avg = parse_previous_metric(previous_report, "100x forward average time")
    prev_adjoint_avg = parse_previous_metric(previous_report, "100x adjoint average time")
    prev_pair_avg = parse_previous_metric(previous_report, "50x forward+adjoint average pair time")

    operator = build_psc_operator(debug=False, run_validation=False)
    signal_dim = H * W
    measure_dim = P * Q

    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026)

    input_real = torch.rand((1, signal_dim), generator=generator, dtype=torch.float32)
    input_imag = torch.rand((1, signal_dim), generator=generator, dtype=torch.float32)
    x = torch.complex(input_real, input_imag).to(torch.complex64)

    y_real = torch.rand((1, measure_dim), generator=generator, dtype=torch.float32)
    y_imag = torch.rand((1, measure_dim), generator=generator, dtype=torch.float32)
    y = torch.complex(y_real, y_imag).to(torch.complex64)

    forward_start = time.perf_counter()
    forward_out = operator.psi_forward(x)
    forward_time = time.perf_counter() - forward_start

    adjoint_start = time.perf_counter()
    adjoint_out = operator.psi_adjoint(y)
    adjoint_time = time.perf_counter() - adjoint_start

    forward_loop_start = time.perf_counter()
    for _ in range(100):
        operator.psi_forward(x)
    forward_loop_time = time.perf_counter() - forward_loop_start

    adjoint_loop_start = time.perf_counter()
    for _ in range(100):
        operator.psi_adjoint(y)
    adjoint_loop_time = time.perf_counter() - adjoint_loop_start

    mixed_loop_start = time.perf_counter()
    for _ in range(50):
        tmp = operator.psi_forward(x)
        operator.psi_adjoint(tmp)
    mixed_loop_time = time.perf_counter() - mixed_loop_start

    coeffs_real = torch.rand((1, signal_dim), generator=generator, dtype=torch.float32)
    coeffs_imag = torch.rand((1, signal_dim), generator=generator, dtype=torch.float32)
    coeffs = torch.complex(coeffs_real, coeffs_imag).to(torch.complex64)
    recon = operator.psi_forward(coeffs)

    save_complex_amplitude_raw(recon[0], output_dir / "full_recon_raw.png")
    save_complex_amplitude_norm(recon[0], output_dir / "full_recon_norm.png")

    chunk_atoms = min(operator.chunk_size, signal_dim)
    phase_tensor_bytes = P * chunk_atoms * Q * torch.tensor([], dtype=torch.float32).element_size()
    atom_block_bytes = chunk_atoms * measure_dim * torch.tensor([], dtype=torch.complex64).element_size()
    input_bytes = x.nelement() * x.element_size()
    output_bytes = forward_out.nelement() * forward_out.element_size()

    recon_has_nan = torch.isnan(recon.real).any().item() or torch.isnan(recon.imag).any().item()
    recon_is_empty = recon.numel() == 0

    lines = [
        "Full 80x80 Operator Test",
        f"Input x shape: {tuple(x.shape)}, dtype: {x.dtype}",
        f"Forward output shape: {tuple(forward_out.shape)}, dtype: {forward_out.dtype}",
        f"Adjoint input y shape: {tuple(y.shape)}, dtype: {y.dtype}",
        f"Adjoint output shape: {tuple(adjoint_out.shape)}, dtype: {adjoint_out.dtype}",
        f"Single forward time: {forward_time:.6f}s",
        f"Single adjoint time: {adjoint_time:.6f}s",
        f"100x forward total time: {forward_loop_time:.6f}s",
        f"100x forward average time: {forward_loop_time / 100.0:.6f}s",
        f"100x adjoint total time: {adjoint_loop_time:.6f}s",
        f"100x adjoint average time: {adjoint_loop_time / 100.0:.6f}s",
        f"50x forward+adjoint total time: {mixed_loop_time:.6f}s",
        f"50x forward+adjoint average pair time: {mixed_loop_time / 50.0:.6f}s",
        (
            f"Previous 100x forward average time: {prev_forward_avg:.6f}s"
            if prev_forward_avg is not None
            else "Previous 100x forward average time: n/a"
        ),
        (
            f"Previous 100x adjoint average time: {prev_adjoint_avg:.6f}s"
            if prev_adjoint_avg is not None
            else "Previous 100x adjoint average time: n/a"
        ),
        (
            f"Previous 50x forward+adjoint average pair time: {prev_pair_avg:.6f}s"
            if prev_pair_avg is not None
            else "Previous 50x forward+adjoint average pair time: n/a"
        ),
        f"Estimated input tensor memory: {bytes_to_mib(input_bytes):.4f} MiB",
        f"Estimated output tensor memory: {bytes_to_mib(output_bytes):.4f} MiB",
        f"Estimated phase chunk memory: {bytes_to_mib(phase_tensor_bytes):.4f} MiB",
        f"Estimated atom block memory: {bytes_to_mib(atom_block_bytes):.4f} MiB",
        f"Cached operator items: {', '.join(operator.cached_items)}",
        f"Reconstruction empty tensor: {recon_is_empty}",
        f"Reconstruction contains NaN: {recon_has_nan}",
        (
            "Full 80x80 runnable: "
            f"{(not recon_is_empty) and (not recon_has_nan)}"
        ),
        (
            "Main bottleneck: chunk-wise complex exponential generation inside each "
            "operator call, even after caching geometry and sampling terms."
        ),
        (
            "Recommended next step: operator performance optimization before full-size training."
        ),
    ]

    report_path = output_dir / "full_operator_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    optimization_report_path = output_dir / "operator_optimization_report.txt"
    optimization_report_path.write_text("\n".join(lines), encoding="utf-8")
    profile_report_path = output_dir / "operator_profile_report.txt"
    profile_report_path.write_text(
        "\n".join(operator.get_profile_report_lines()), encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
