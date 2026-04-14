import math
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.config import B, H, P, Q, W, c, fc, phi_syn_deg, xinterval, yinterval

_VALIDATION_DONE = False


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


class PSCOperator:
    def __init__(
        self,
        debug: bool = True,
        debug_size: int = 32,
        device: torch.device | str | None = None,
        chunk_size: int = 128,
    ) -> None:
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.signal_h = debug_size if debug else H
        self.signal_w = debug_size if debug else W
        self.measure_p = debug_size if debug else P
        self.measure_q = debug_size if debug else Q
        self.signal_dim = self.signal_h * self.signal_w
        self.measure_dim = self.measure_p * self.measure_q
        self.chunk_size = chunk_size
        self.dtype = torch.complex64
        self.debug = debug
        self.debug_size = debug_size
        self.cached_items = []
        self.profile_stats = {
            "forward_calls": 0,
            "adjoint_calls": 0,
            "forward_total": 0.0,
            "adjoint_total": 0.0,
            "chunk_loop_total": 0.0,
            "atom_fetch_total": 0.0,
            "atom_generate_total": 0.0,
            "aggregate_total": 0.0,
        }

        self.scatter_x = _linspace_extent(self.signal_w, xinterval, self.device)
        self.scatter_y = _linspace_extent(self.signal_h, yinterval, self.device)
        yy, xx = torch.meshgrid(self.scatter_y, self.scatter_x, indexing="ij")
        self.scatter_positions = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
        self.cached_items.extend(["scatter_x", "scatter_y", "scatter_positions"])

        phi_syn = math.radians(phi_syn_deg)
        self.phi = torch.linspace(
            -phi_syn / 2.0,
            phi_syn / 2.0,
            steps=self.measure_q,
            device=self.device,
            dtype=torch.float32,
        )
        self.frequencies = torch.linspace(
            fc - B / 2.0,
            fc + B / 2.0,
            steps=self.measure_p,
            device=self.device,
            dtype=torch.float32,
        )
        self.cos_phi = torch.cos(self.phi)
        self.sin_phi = torch.sin(self.phi)
        self.wavenumbers = (-(4.0 * math.pi / c) * self.frequencies).view(-1, 1, 1)
        self.norm_scale = 1.0 / math.sqrt(self.measure_dim)
        self.cached_items.extend(["phi", "frequencies", "cos_phi", "sin_phi", "wavenumbers"])
        self._phase_projection = (
            self.scatter_positions[:, 0].unsqueeze(1) * self.cos_phi.unsqueeze(0)
            + self.scatter_positions[:, 1].unsqueeze(1) * self.sin_phi.unsqueeze(0)
        )
        self.cached_items.append("phase_projection")
        self._chunk_indices = []
        self._phase_projection_chunks = []
        self._atom_block_cache = []
        for start in range(0, self.signal_dim, self.chunk_size):
            end = min(start + self.chunk_size, self.signal_dim)
            atom_indices = torch.arange(start, end, device=self.device, dtype=torch.long)
            self._chunk_indices.append(atom_indices)
            phase_chunk = self._phase_projection.index_select(0, atom_indices).contiguous()
            self._phase_projection_chunks.append(phase_chunk)
            atom_block = self._atom_block_from_projection(phase_chunk)
            self._atom_block_cache.append(atom_block.contiguous())
        self.cached_items.extend(["chunk_indices", "phase_projection_chunks", "atom_block_cache"])

    def _atom_block_from_projection(self, phase_projection: torch.Tensor) -> torch.Tensor:
        phase = self.wavenumbers * phase_projection.view(1, phase_projection.shape[0], self.measure_q)
        atom_block = torch.polar(torch.ones_like(phase), phase)
        atom_block = atom_block.permute(1, 0, 2).reshape(phase_projection.shape[0], self.measure_dim)
        return atom_block.to(torch.complex64) * self.norm_scale

    def _atom_block(self, atom_indices: torch.Tensor) -> torch.Tensor:
        atom_indices = atom_indices.to(device=self.device, dtype=torch.long)
        phase_projection = self._phase_projection.index_select(0, atom_indices)
        return self._atom_block_from_projection(phase_projection)

    def atom_response(self, atom_index: int) -> torch.Tensor:
        atom = self._atom_block(torch.tensor([atom_index], device=self.device))[0]
        return atom.view(self.measure_p, self.measure_q)

    def _get_cached_atom_block(self, chunk_idx: int) -> torch.Tensor:
        return self._atom_block_cache[chunk_idx]

    def psi_forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
        x = x.to(device=self.device, dtype=torch.complex64)

        call_start = time.perf_counter()
        y = torch.zeros((x.shape[0], self.measure_dim), dtype=torch.complex64, device=self.device)
        for chunk_idx, chunk_indices in enumerate(self._chunk_indices):
            chunk_start = time.perf_counter()
            fetch_start = time.perf_counter()
            atom_block = self._get_cached_atom_block(chunk_idx)
            self.profile_stats["atom_fetch_total"] += time.perf_counter() - fetch_start
            start = chunk_indices[0].item()
            end = chunk_indices[-1].item() + 1
            aggregate_start = time.perf_counter()
            y = y + x[:, start:end] @ atom_block
            self.profile_stats["aggregate_total"] += time.perf_counter() - aggregate_start
            self.profile_stats["chunk_loop_total"] += time.perf_counter() - chunk_start
        call_elapsed = time.perf_counter() - call_start
        self.profile_stats["forward_calls"] += 1
        self.profile_stats["forward_total"] += call_elapsed
        return y.squeeze(0) if squeeze else y

    def psi_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        squeeze = y.ndim == 1
        if squeeze:
            y = y.unsqueeze(0)
        y = y.to(device=self.device, dtype=torch.complex64)

        call_start = time.perf_counter()
        x = torch.zeros((y.shape[0], self.signal_dim), dtype=torch.complex64, device=self.device)
        for chunk_idx, chunk_indices in enumerate(self._chunk_indices):
            chunk_start = time.perf_counter()
            fetch_start = time.perf_counter()
            atom_block = self._get_cached_atom_block(chunk_idx)
            self.profile_stats["atom_fetch_total"] += time.perf_counter() - fetch_start
            start = chunk_indices[0].item()
            end = chunk_indices[-1].item() + 1
            aggregate_start = time.perf_counter()
            x[:, start:end] = y @ atom_block.conj().transpose(0, 1)
            self.profile_stats["aggregate_total"] += time.perf_counter() - aggregate_start
            self.profile_stats["chunk_loop_total"] += time.perf_counter() - chunk_start
        call_elapsed = time.perf_counter() - call_start
        self.profile_stats["adjoint_calls"] += 1
        self.profile_stats["adjoint_total"] += call_elapsed
        return x.squeeze(0) if squeeze else x

    def get_profile_report_lines(self) -> list[str]:
        forward_avg = (
            self.profile_stats["forward_total"] / self.profile_stats["forward_calls"]
            if self.profile_stats["forward_calls"] > 0
            else 0.0
        )
        adjoint_avg = (
            self.profile_stats["adjoint_total"] / self.profile_stats["adjoint_calls"]
            if self.profile_stats["adjoint_calls"] > 0
            else 0.0
        )
        return [
            "Operator Profile Report",
            f"forward_calls={self.profile_stats['forward_calls']}",
            f"adjoint_calls={self.profile_stats['adjoint_calls']}",
            f"forward_total={self.profile_stats['forward_total']:.6f}s",
            f"adjoint_total={self.profile_stats['adjoint_total']:.6f}s",
            f"forward_avg={forward_avg:.6f}s",
            f"adjoint_avg={adjoint_avg:.6f}s",
            f"chunk_loop_total={self.profile_stats['chunk_loop_total']:.6f}s",
            f"atom_fetch_total={self.profile_stats['atom_fetch_total']:.6f}s",
            f"atom_generate_total={self.profile_stats['atom_generate_total']:.6f}s",
            f"aggregate_total={self.profile_stats['aggregate_total']:.6f}s",
            f"cached_items={', '.join(self.cached_items)}",
        ]


def summarize_psi(psi: Any) -> dict[str, float | str | tuple[int, ...]]:
    if isinstance(psi, PSCOperator):
        return {
            "shape": (psi.measure_dim, psi.signal_dim),
            "dtype": str(psi.dtype),
            "device": str(psi.device),
            "norm_min": 1.0,
            "norm_max": 1.0,
            "norm_mean": 1.0,
            "cached_items": ", ".join(psi.cached_items),
        }

    norm_values = torch.linalg.vector_norm(psi, ord=2, dim=0)
    return {
        "shape": tuple(psi.shape),
        "dtype": str(psi.dtype),
        "device": str(psi.device),
        "norm_min": norm_values.min().item(),
        "norm_max": norm_values.max().item(),
        "norm_mean": norm_values.mean().item(),
    }


def save_atom_visualizations(
    psi: Any,
    measure_p: int,
    measure_q: int,
    output_dir: str | Path = Path("outputs") / "psi_atoms",
    debug: bool = True,
    debug_size: int = 32,
    num_atoms: int = 5,
) -> list[int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(psi, PSCOperator):
        signal_dim = psi.signal_dim
    else:
        signal_dim = psi.shape[1]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    atom_indices = torch.randperm(signal_dim, generator=generator)[:num_atoms].tolist()
    mode_name = f"debug_{debug_size}" if debug else "full_80"

    for atom_idx in atom_indices:
        if isinstance(psi, PSCOperator):
            atom = psi.atom_response(atom_idx)
        else:
            atom = psi[:, atom_idx].view(measure_p, measure_q)
        real_image = _normalize_to_uint8(atom.real)
        imag_image = _normalize_to_uint8(atom.imag)
        phase_image = _normalize_to_uint8(torch.angle(atom))
        real_image.save(output_dir / f"{mode_name}_atom_{atom_idx}_real.png")
        imag_image.save(output_dir / f"{mode_name}_atom_{atom_idx}_imag.png")
        phase_image.save(output_dir / f"{mode_name}_atom_{atom_idx}_phase.png")

    return atom_indices


def build_psc_dictionary(
    debug: bool = True,
    debug_size: int = 32,
    device: torch.device | str | None = None,
    chunk_size: int = 128,
) -> torch.Tensor:
    operator = build_psc_operator(
        debug=debug,
        debug_size=debug_size,
        device=device,
        chunk_size=chunk_size,
        run_validation=False,
    )
    psi = torch.empty(
        (operator.measure_dim, operator.signal_dim),
        dtype=torch.complex64,
        device=operator.device,
    )
    for start in range(0, operator.signal_dim, chunk_size):
        end = min(start + chunk_size, operator.signal_dim)
        atom_indices = torch.arange(start, end, device=operator.device)
        psi[:, start:end] = operator._atom_block(atom_indices).transpose(0, 1)

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
        operator.measure_p,
        operator.measure_q,
        output_dir=Path("outputs") / "psi_atoms",
        debug=debug,
        debug_size=debug_size,
    )
    return psi


def validate_operator_consistency(size: int = 16, chunk_size: int = 32) -> None:
    operator = PSCOperator(debug=True, debug_size=size, chunk_size=chunk_size)
    psi_small = build_psc_dictionary(
        debug=True,
        debug_size=size,
        device=operator.device,
        chunk_size=chunk_size,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    x = torch.randn((1, size * size), generator=generator, dtype=torch.float32)
    x = torch.complex(x, torch.randn((1, size * size), generator=generator, dtype=torch.float32))
    y = torch.randn((1, size * size), generator=generator, dtype=torch.float32)
    y = torch.complex(y, torch.randn((1, size * size), generator=generator, dtype=torch.float32))

    explicit_forward = x.to(torch.complex64) @ psi_small.transpose(0, 1)
    operator_forward = operator.psi_forward(x)
    explicit_adjoint = y.to(torch.complex64) @ psi_small.conj()
    operator_adjoint = operator.psi_adjoint(y)

    forward_error = torch.max(torch.abs(explicit_forward - operator_forward)).item()
    adjoint_error = torch.max(torch.abs(explicit_adjoint - operator_adjoint)).item()
    print(
        f"Operator consistency (size={size}): "
        f"forward_error={forward_error:.6e}, "
        f"adjoint_error={adjoint_error:.6e}"
    )


def build_psc_operator(
    debug: bool = True,
    debug_size: int = 32,
    device: torch.device | str | None = None,
    chunk_size: int = 128,
    run_validation: bool = True,
) -> PSCOperator:
    global _VALIDATION_DONE
    operator = PSCOperator(
        debug=debug,
        debug_size=debug_size,
        device=device,
        chunk_size=chunk_size,
    )
    psi_summary = summarize_psi(operator)
    print(
        f"Psi operator shape: {psi_summary['shape']}, "
        f"dtype: {psi_summary['dtype']}, device: {psi_summary['device']}"
    )
    print(
        "Psi operator column norms: "
        f"min={psi_summary['norm_min']:.6f}, "
        f"max={psi_summary['norm_max']:.6f}, "
        f"mean={psi_summary['norm_mean']:.6f}"
    )
    save_atom_visualizations(
        operator,
        operator.measure_p,
        operator.measure_q,
        output_dir=Path("outputs") / "psi_atoms",
        debug=debug,
        debug_size=debug_size,
    )
    if run_validation and not _VALIDATION_DONE:
        validate_operator_consistency()
        _VALIDATION_DONE = True
    return operator


def psi_forward(x: torch.Tensor, debug: bool = True, debug_size: int = 32) -> torch.Tensor:
    operator = build_psc_operator(debug=debug, debug_size=debug_size)
    return operator.psi_forward(x)


def psi_adjoint(y: torch.Tensor, debug: bool = True, debug_size: int = 32) -> torch.Tensor:
    operator = build_psc_operator(debug=debug, debug_size=debug_size)
    return operator.psi_adjoint(y)
