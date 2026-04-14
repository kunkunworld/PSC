import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from src.dataset import ComplexImageDataset
from src.psc_module import PSCModule


def complex_mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(x.real, y.real) + nn.functional.mse_loss(
        x.imag, y.imag
    )


def save_amplitude_image(tensor: torch.Tensor, path: Path) -> None:
    amplitude = torch.abs(tensor.detach().cpu()).squeeze(0).clamp(0.0, 1.0)
    image = (amplitude.numpy() * 255.0).astype("uint8")
    Image.fromarray(image, mode="L").save(path)


def save_minmax_amplitude_image(tensor: torch.Tensor, path: Path) -> None:
    amplitude = torch.abs(tensor.detach().cpu()).squeeze(0)
    amplitude = amplitude - amplitude.min()
    max_value = amplitude.max()
    if max_value > 0:
        amplitude = amplitude / max_value
    image = (amplitude.numpy() * 255.0).astype("uint8")
    Image.fromarray(image, mode="L").save(path)


def save_heatmap_image(tensor: torch.Tensor, path: Path) -> None:
    side = int(tensor.numel() ** 0.5)
    heatmap = torch.abs(tensor.detach().cpu()).reshape(side, side)
    heatmap = heatmap - heatmap.min()
    max_value = heatmap.max()
    if max_value > 0:
        heatmap = heatmap / max_value
    image = (heatmap.numpy() * 255.0).astype("uint8")
    Image.fromarray(image, mode="L").save(path)


def save_histogram_image(
    tensor: torch.Tensor, path: Path, bins: int = 64, width: int = 256, height: int = 256
) -> None:
    values = torch.abs(tensor.detach().cpu()).flatten()
    max_value = values.max().item()
    hist = torch.histc(values, bins=bins, min=0.0, max=max_value if max_value > 0 else 1.0)
    hist = hist / hist.max().clamp(min=1.0)

    image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(image)
    bar_width = max(1, width // bins)
    for idx in range(bins):
        bar_height = int(hist[idx].item() * (height - 1))
        left = idx * bar_width
        right = min(width - 1, left + bar_width - 1)
        top = height - max(bar_height, 1)
        draw.rectangle((left, top, right, height - 1), fill=255)
    image.save(path)


def main() -> None:
    output_root = Path("outputs")
    output_root.mkdir(exist_ok=True)
    smoke_dir = output_root / "full_smoke"
    smoke_dir.mkdir(exist_ok=True)

    dataset = ComplexImageDataset(data_dir="data")
    subset = Subset(dataset, [0, 1])
    dataloader = DataLoader(subset, batch_size=1, shuffle=False)

    model = PSCModule(dictionary_debug=False)
    optimizer = Adam(model.parameters(), lr=1e-3)

    num_epochs = 2
    forward_times = []
    backward_times = []
    losses = []
    nan_detected = False
    completed_steps = 0

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader, start=1):
            forward_start = time.perf_counter()
            recon, o, p = model(batch)
            forward_elapsed = time.perf_counter() - forward_start

            loss = (
                complex_mse_loss(batch, recon)
                + 0.1 * nn.functional.mse_loss(o, p)
                + 0.01 * torch.mean(torch.abs(p))
            )

            has_nan = (
                torch.isnan(loss).item()
                or torch.isnan(recon.real).any().item()
                or torch.isnan(recon.imag).any().item()
            )
            if has_nan:
                nan_detected = True
                print(f"epoch {epoch + 1} step {step} encountered NaN")
                break

            backward_start = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            backward_elapsed = time.perf_counter() - backward_start

            forward_times.append(forward_elapsed)
            backward_times.append(backward_elapsed)
            losses.append(loss.item())
            completed_steps += 1

            print(
                f"epoch {epoch + 1} step {step} "
                f"loss={loss.item():.6f} "
                f"forward={forward_elapsed:.6f}s "
                f"backward={backward_elapsed:.6f}s "
                f"nan={has_nan}"
            )

            if step == 1:
                save_amplitude_image(batch[0], smoke_dir / f"epoch_{epoch + 1}_input.png")
                save_amplitude_image(recon[0], smoke_dir / f"epoch_{epoch + 1}_recon_raw.png")
                save_minmax_amplitude_image(
                    recon[0], smoke_dir / f"epoch_{epoch + 1}_recon_norm.png"
                )
                save_heatmap_image(o[0], smoke_dir / f"epoch_{epoch + 1}_o_norm.png")
                save_histogram_image(o[0], smoke_dir / f"epoch_{epoch + 1}_o_hist.png")

        if nan_detected:
            break

    avg_step_time = 0.0
    if forward_times and backward_times:
        avg_step_time = sum(forward_times[i] + backward_times[i] for i in range(len(losses))) / len(losses)

    loss_trend = "stable"
    if len(losses) >= 2:
        if losses[-1] < losses[0]:
            loss_trend = "decreasing"
        elif losses[-1] > losses[0] * 1.1:
            loss_trend = "increasing"

    report_lines = [
        "Full 80x80 Baseline Smoke Test",
        f"Completed steps: {completed_steps}",
        f"NaN detected: {nan_detected}",
        f"Average forward time: {sum(forward_times) / len(forward_times):.6f}s" if forward_times else "Average forward time: n/a",
        f"Average backward time: {sum(backward_times) / len(backward_times):.6f}s" if backward_times else "Average backward time: n/a",
        f"Average step time: {avg_step_time:.6f}s" if losses else "Average step time: n/a",
        f"Loss trend: {loss_trend}",
        f"Initial loss: {losses[0]:.6f}" if losses else "Initial loss: n/a",
        f"Final loss: {losses[-1]:.6f}" if losses else "Final loss: n/a",
        (
            "Full 80x80 training stable: "
            f"{(completed_steps == num_epochs * len(dataloader)) and not nan_detected}"
        ),
        (
            "Memory / runtime note: no explicit memory exception occurred during the smoke test; "
            "runtime cost is dominated by repeated full-size operator calls."
        ),
        (
            "Reconstruction note: recon_norm images are saved for visual inspection of whether "
            "the main input structure remains visible under full-mode baseline training."
        ),
        (
            "Suggested next step: "
            "b. operator final-round performance optimization before expanding full baseline training."
        ),
    ]

    report_path = output_root / "full_smoke_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
