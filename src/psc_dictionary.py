import math
from pathlib import Path

import torch
from PIL import Image

from src.config import B, H, P, Q, W, c, fc, phi_syn_deg, xinterval, yinterval


def _linspace_extent(num_points: int, spacing: float, device: torch.device) -> torch.Tensor:
    half_extent = spacing * (num_points - 1) / 2.0
    return torch.linspace(-half_extent, half_extent, steps=num_points, device=device)


def _normalize_to_uint8(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().to(torch.float32)
    image = image - image.min()
    max_value = image.max()
    if max_value > 0:
        image = image / max_value
    array = (image.numpy() * 255.0).astype("uint8")
    return Image.fromarray(array, mode="L")


def save_atom_visualizations(
    psi: torch.Tensor,
    measure_p: int,
    measure_q: int,
    output_dir: str | Path = Path("outputs") / "psi_atoms",
    debug: bool = True,
    debug_size: int = 32,
    num_atoms: int = 5,
) -> list[int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    atom_indices = torch.randperm(psi.shape[1], generator=generator)[:num_atoms].tolist()
    mode_name = f"debug_{debug_size}" if debug else "full_80"

    for atom_idx in atom_indices:
        atom = psi[:, atom_idx].view(measure_p, measure_q)
        real_image = _normalize_to_uint8(atom.real)
        imag_image = _normalize_to_uint8(atom.imag)
        phase_image = _normalize_to_uint8(torch.angle(atom))
        real_image.save(output_dir / f"{mode_name}_atom_{atom_idx}_real.png")
        imag_image.save(output_dir / f"{mode_name}_atom_{atom_idx}_imag.png")
        phase_image.save(output_dir / f"{mode_name}_atom_{atom_idx}_phase.png")

    return atom_indices


def summarize_psi(psi: torch.Tensor) -> dict[str, float | str | tuple[int, ...]]:
    norm_values = torch.linalg.vector_norm(psi, ord=2, dim=0)
    return {
        "shape": tuple(psi.shape),
        "dtype": str(psi.dtype),
        "device": str(psi.device),
        "norm_min": norm_values.min().item(),
        "norm_max": norm_values.max().item(),
        "norm_mean": norm_values.mean().item(),
    }


def build_psc_dictionary(
    debug: bool = True,
    debug_size: int = 32,
    device: torch.device | str | None = None,
    chunk_size: int = 128,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    signal_h = debug_size if debug else H
    signal_w = debug_size if debug else W
    measure_p = debug_size if debug else P
    measure_q = debug_size if debug else Q

    scatter_x = _linspace_extent(signal_w, xinterval, device)
    scatter_y = _linspace_extent(signal_h, yinterval, device)
    yy, xx = torch.meshgrid(scatter_y, scatter_x, indexing="ij")
    scatter_positions = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)

    phi_syn = math.radians(phi_syn_deg)
    phi = torch.linspace(
        -phi_syn / 2.0,
        phi_syn / 2.0,
        steps=measure_q,
        device=device,
        dtype=torch.float32,
    )
    frequencies = torch.linspace(
        fc - B / 2.0,
        fc + B / 2.0,
        steps=measure_p,
        device=device,
        dtype=torch.float32,
    )

    phase_projection = (
        scatter_positions[:, 0].unsqueeze(1) * torch.cos(phi).unsqueeze(0)
        + scatter_positions[:, 1].unsqueeze(1) * torch.sin(phi).unsqueeze(0)
    )

    signal_dim = signal_h * signal_w
    measure_dim = measure_p * measure_q
    psi = torch.empty((measure_dim, signal_dim), dtype=torch.complex64, device=device)

    write_row = 0
    for start in range(0, measure_p, chunk_size):
        end = min(start + chunk_size, measure_p)
        frequency_block = frequencies[start:end]
        phase = (
            -(4.0 * math.pi / c)
            * frequency_block.view(-1, 1, 1)
            * phase_projection.view(1, signal_dim, measure_q)
        )
        atom_block = torch.polar(torch.ones_like(phase), phase)
        atom_block = atom_block.permute(0, 2, 1).reshape(-1, signal_dim)
        psi[write_row : write_row + atom_block.shape[0]] = atom_block
        write_row += atom_block.shape[0]

    column_norms = torch.linalg.vector_norm(psi, ord=2, dim=0, keepdim=True).clamp_min(1e-12)
    psi = psi / column_norms

    psi_summary = summarize_psi(psi)
    print(f"Psi shape: {psi.shape}, dtype: {psi.dtype}, device: {psi.device}")
    print(
        "Psi column norms: "
        f"min={psi_summary['norm_min']:.6f}, "
        f"max={psi_summary['norm_max']:.6f}, "
        f"mean={psi_summary['norm_mean']:.6f}"
    )

    save_atom_visualizations(
        psi,
        measure_p,
        measure_q,
        output_dir=Path("outputs") / "psi_atoms",
        debug=debug,
        debug_size=debug_size,
    )
    return psi
