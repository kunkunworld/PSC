from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dataset import ComplexImageDataset
from src.psc_module import PSCModule


def complex_mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(x.real, y.real) + nn.functional.mse_loss(
        x.imag, y.imag
    )


def save_amplitude_image(tensor: torch.Tensor, path: str) -> None:
    amplitude = torch.abs(tensor.detach().cpu()).squeeze(0).clamp(0.0, 1.0)
    image = (amplitude.numpy() * 255.0).astype("uint8")
    Image.fromarray(image, mode="L").save(path)


def save_minmax_amplitude_image(tensor: torch.Tensor, path: str) -> None:
    amplitude = torch.abs(tensor.detach().cpu()).squeeze(0)
    amplitude = amplitude - amplitude.min()
    max_value = amplitude.max()
    if max_value > 0:
        amplitude = amplitude / max_value
    image = (amplitude.numpy() * 255.0).astype("uint8")
    Image.fromarray(image, mode="L").save(path)


def infer_square_side(tensor: torch.Tensor) -> int:
    numel = tensor.numel()
    side = int(numel**0.5)
    if side * side != numel:
        raise ValueError(f"Cannot reshape tensor with {numel} elements into square heatmap.")
    return side


def save_heatmap_image(tensor: torch.Tensor, side: int, path: str) -> None:
    heatmap = torch.abs(tensor.detach().cpu().reshape(side, side))
    heatmap = heatmap - heatmap.min()
    max_value = heatmap.max()
    if max_value > 0:
        heatmap = heatmap / max_value
    image = (heatmap.numpy() * 255.0).astype("uint8")
    Image.fromarray(image, mode="L").save(path)


def save_histogram_image(
    tensor: torch.Tensor, path: str, bins: int = 64, width: int = 256, height: int = 256
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
    dataset = ComplexImageDataset(data_dir="data")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = PSCModule()
    optimizer = Adam(model.parameters(), lr=1e-3)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    for epoch in range(2):
        epoch_input_min_total = 0.0
        epoch_input_max_total = 0.0
        epoch_input_mean_total = 0.0
        epoch_recon_min_total = 0.0
        epoch_recon_max_total = 0.0
        epoch_recon_mean_total = 0.0
        epoch_o_min_total = 0.0
        epoch_o_max_total = 0.0
        epoch_o_mean_total = 0.0
        epoch_o_nonzero_ratio_total = 0.0
        last_o_abs_top20 = None
        num_steps = 0

        for step, batch in enumerate(dataloader, start=1):
            if epoch == 0 and step == 1:
                print(f"input shape: {batch.shape}")
                print(f"dtype: {batch.dtype}")
                print(f"is complex: {batch.is_complex()}")

            recon, o, p = model(batch)

            if epoch == 0 and step == 1:
                print(f"recon shape: {recon.shape}")
                print(f"o shape: {o.shape}")
                print(f"p shape: {p.shape}")

            loss = (
                complex_mse_loss(batch, recon)
                + 0.1 * nn.functional.mse_loss(o, p)
                + 0.01 * torch.mean(torch.abs(p))
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"epoch {epoch + 1} step {step} loss: {loss.item():.6f}")
            num_steps += 1

            input_amplitude = torch.abs(batch)
            recon_amplitude = torch.abs(recon)
            o_abs = torch.abs(o)
            epoch_input_min_total += input_amplitude.min().item()
            epoch_input_max_total += input_amplitude.max().item()
            epoch_input_mean_total += input_amplitude.mean().item()
            epoch_recon_min_total += recon_amplitude.min().item()
            epoch_recon_max_total += recon_amplitude.max().item()
            epoch_recon_mean_total += recon_amplitude.mean().item()
            epoch_o_min_total += o_abs.min().item()
            epoch_o_max_total += o_abs.max().item()
            epoch_o_mean_total += o_abs.mean().item()
            epoch_o_nonzero_ratio_total += (o_abs > 1e-3).float().mean().item()
            last_o_abs_top20 = torch.topk(o_abs.flatten(), k=20).values.detach().cpu()

            if step == 1:
                save_amplitude_image(batch[0], str(output_dir / "input.png"))
                save_amplitude_image(recon[0], str(output_dir / "recon_raw.png"))
                save_minmax_amplitude_image(recon[0], str(output_dir / "recon_norm.png"))
                save_heatmap_image(
                    o[0], infer_square_side(o[0]), str(output_dir / "o_norm.png")
                )
                save_histogram_image(o[0], str(output_dir / "o_hist.png"))

        psi_diag = model.get_psi_diagnostics()
        print(
            f"epoch {epoch + 1} input amplitude stats: "
            f"min={epoch_input_min_total / num_steps:.6f}, "
            f"max={epoch_input_max_total / num_steps:.6f}, "
            f"mean={epoch_input_mean_total / num_steps:.6f}"
        )
        print(
            f"epoch {epoch + 1} recon amplitude stats: "
            f"min={epoch_recon_min_total / num_steps:.6f}, "
            f"max={epoch_recon_max_total / num_steps:.6f}, "
            f"mean={epoch_recon_mean_total / num_steps:.6f}"
        )
        print(
            f"epoch {epoch + 1} o stats: "
            f"min(abs)={epoch_o_min_total / num_steps:.6f}, "
            f"max(abs)={epoch_o_max_total / num_steps:.6f}, "
            f"mean(abs)={epoch_o_mean_total / num_steps:.6f}"
        )
        print(
            f"epoch {epoch + 1} o nonzero ratio (>1e-3): "
            f"{epoch_o_nonzero_ratio_total / num_steps:.6f}"
        )
        print(
            f"epoch {epoch + 1} o abs top20: "
            f"{[round(value, 6) for value in last_o_abs_top20.tolist()]}"
        )
        print(
            f"epoch {epoch + 1} Psi: "
            f"shape={psi_diag['shape']}, dtype={psi_diag['dtype']}, device={psi_diag['device']}"
        )
        print(
            f"epoch {epoch + 1} Psi column norms: "
            f"min={psi_diag['norm_min']:.6f}, "
            f"max={psi_diag['norm_max']:.6f}, "
            f"mean={psi_diag['norm_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
