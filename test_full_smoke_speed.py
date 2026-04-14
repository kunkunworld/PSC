import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from src.dataset import ComplexImageDataset
from src.psc_module import PSCModule


def complex_mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(x.real, y.real) + nn.functional.mse_loss(
        x.imag, y.imag
    )


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


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    previous_report = output_dir / "full_smoke_report.txt"
    prev_forward = parse_previous_metric(previous_report, "Average forward time")
    prev_backward = parse_previous_metric(previous_report, "Average backward time")
    prev_total = parse_previous_metric(previous_report, "Average step time")

    dataset = ComplexImageDataset(data_dir="data")
    subset = Subset(dataset, [0, 1])
    dataloader = DataLoader(subset, batch_size=1, shuffle=False)
    model = PSCModule(dictionary_debug=False)
    optimizer = Adam(model.parameters(), lr=1e-3)

    forward_times = []
    backward_times = []
    total_times = []
    losses = []
    nan_detected = False

    for step, batch in enumerate(dataloader, start=1):
        step_start = time.perf_counter()
        forward_start = time.perf_counter()
        recon, o, p = model(batch)
        forward_elapsed = time.perf_counter() - forward_start

        loss = (
            complex_mse_loss(batch, recon)
            + 0.1 * nn.functional.mse_loss(o, p)
            + 0.01 * torch.mean(torch.abs(p))
        )

        if (
            torch.isnan(loss).item()
            or torch.isnan(recon.real).any().item()
            or torch.isnan(recon.imag).any().item()
        ):
            nan_detected = True
            break

        backward_start = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_elapsed = time.perf_counter() - backward_start
        total_elapsed = time.perf_counter() - step_start

        forward_times.append(forward_elapsed)
        backward_times.append(backward_elapsed)
        total_times.append(total_elapsed)
        losses.append(loss.item())

        print(
            f"smoke speed step {step}: "
            f"loss={loss.item():.6f} "
            f"forward={forward_elapsed:.6f}s "
            f"backward={backward_elapsed:.6f}s "
            f"total={total_elapsed:.6f}s"
        )

    avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0.0
    avg_backward = sum(backward_times) / len(backward_times) if backward_times else 0.0
    avg_total = sum(total_times) / len(total_times) if total_times else 0.0

    lines = [
        "Full Smoke Speed Compare",
        f"NaN detected: {nan_detected}",
        f"Current average forward time: {avg_forward:.6f}s",
        f"Current average backward time: {avg_backward:.6f}s",
        f"Current average total time: {avg_total:.6f}s",
        (
            f"Previous average forward time: {prev_forward:.6f}s"
            if prev_forward is not None
            else "Previous average forward time: n/a"
        ),
        (
            f"Previous average backward time: {prev_backward:.6f}s"
            if prev_backward is not None
            else "Previous average backward time: n/a"
        ),
        (
            f"Previous average total time: {prev_total:.6f}s"
            if prev_total is not None
            else "Previous average total time: n/a"
        ),
        f"Initial loss: {losses[0]:.6f}" if losses else "Initial loss: n/a",
        f"Final loss: {losses[-1]:.6f}" if losses else "Final loss: n/a",
    ]

    report_path = output_dir / "full_smoke_speed_compare.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
