import math

import torch

from src.config import B, H, P, Q, W, c, fc, phi_syn_deg, xinterval, yinterval


def _linspace_extent(num_points: int, spacing: float, device: torch.device) -> torch.Tensor:
    half_extent = spacing * (num_points - 1) / 2.0
    return torch.linspace(-half_extent, half_extent, steps=num_points, device=device)


def build_psc_dictionary(
    debug: bool = True,
    debug_size: int = 32,
    device: torch.device | str | None = None,
    chunk_size: int = 8,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    signal_h = debug_size if debug else H
    signal_w = debug_size if debug else W
    measure_p = debug_size if debug else P
    measure_q = debug_size if debug else Q

    x_coords = _linspace_extent(signal_w, xinterval, device)
    y_coords = _linspace_extent(signal_h, yinterval, device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    spatial_grid = xx.reshape(-1)
    range_grid = yy.reshape(-1)

    phi_syn = math.radians(phi_syn_deg)
    view_angles = torch.linspace(
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
    wavenumbers = 2.0 * math.pi * frequencies / c

    steering = (
        torch.cos(view_angles).unsqueeze(1) * range_grid.unsqueeze(0)
        + torch.sin(view_angles).unsqueeze(1) * spatial_grid.unsqueeze(0)
    )

    signal_dim = signal_h * signal_w
    measure_dim = measure_p * measure_q
    psi = torch.empty((measure_dim, signal_dim), dtype=torch.complex64, device=device)

    scale = 1.0 / math.sqrt(signal_dim)
    write_row = 0
    for start in range(0, measure_p, chunk_size):
        end = min(start + chunk_size, measure_p)
        phase = wavenumbers[start:end].view(-1, 1, 1) * steering.view(1, measure_q, -1)
        block = torch.polar(torch.ones_like(phase), -phase) * scale
        block_rows = (end - start) * measure_q
        psi[write_row : write_row + block_rows] = block.reshape(block_rows, signal_dim)
        write_row += block_rows

    print(f"Psi shape: {psi.shape}, dtype: {psi.dtype}, device: {psi.device}")
    return psi
