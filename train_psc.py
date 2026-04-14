from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from src.dataset import ComplexImageDataset
from src.psc_module import PSCModule

SPARSE_L1_WEIGHT = 50.0
SOFT_THRESHOLD_TAU = 0.01


def complex_mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(x.real, y.real) + nn.functional.mse_loss(
        x.imag, y.imag
    )


def soft_threshold(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)


def save_amplitude_image(tensor: torch.Tensor, path: str) -> None:
    amplitude = torch.abs(tensor.detach().cpu()).squeeze(0)
    amplitude = amplitude.clamp(0.0, 1.0)
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


def save_heatmap_image(tensor: torch.Tensor, side: int, path: str) -> None:
    heatmap = tensor.detach().cpu().reshape(side, side)
    heatmap = torch.abs(heatmap)
    heatmap = heatmap - heatmap.min()
    max_value = heatmap.max()
    if max_value > 0:
        heatmap = heatmap / max_value
    image = (heatmap.numpy() * 255.0).astype("uint8")
    Image.fromarray(image, mode="L").save(path)


def save_raw_heatmap_image(tensor: torch.Tensor, side: int, path: str) -> None:
    heatmap = torch.abs(tensor.detach().cpu().reshape(side, side))
    heatmap = heatmap.clamp(min=0.0, max=1.0)
    image = (heatmap.numpy() * 255.0).astype("uint8")
    Image.fromarray(image, mode="L").save(path)


def infer_square_side(tensor: torch.Tensor) -> int:
    numel = tensor.numel()
    side = int(numel**0.5)
    if side * side != numel:
        raise ValueError(f"Cannot reshape tensor with {numel} elements into square heatmap.")
    return side


def save_histogram_image(
    tensor: torch.Tensor, path: str, bins: int = 64, width: int = 256, height: int = 256
) -> None:
    values = torch.abs(tensor.detach().cpu()).flatten()
    max_value = values.max().item()
    hist_range = (0.0, max_value if max_value > 0 else 1.0)
    hist = torch.histc(values, bins=bins, min=hist_range[0], max=hist_range[1])
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
        epoch_recon_loss_total = 0.0
        epoch_input_min_total = 0.0
        epoch_input_max_total = 0.0
        epoch_input_mean_total = 0.0
        epoch_recon_min_total = 0.0
        epoch_recon_max_total = 0.0
        epoch_recon_mean_total = 0.0
        epoch_o_min_total = 0.0
        epoch_o_mean_total = 0.0
        epoch_o_max_total = 0.0
        epoch_o_nonzero_ratio_total = 0.0
        epoch_p_min_total = 0.0
        epoch_p_mean_total = 0.0
        epoch_p_max_total = 0.0
        epoch_p_l1_total = 0.0
        epoch_p_nonzero_ratio_total = 0.0
        last_o_abs_top20 = None
        num_steps = 0

        for step, batch in enumerate(dataloader, start=1):
            if epoch == 0 and step == 1:
                print(f"input shape: {batch.shape}")
                print(f"dtype: {batch.dtype}")
                print(f"is complex: {batch.is_complex()}")

            recon, o, p = model(batch)
            p = soft_threshold(p, SOFT_THRESHOLD_TAU)

            if epoch == 0 and step == 1:
                print(f"recon shape: {recon.shape}")
                print(f"o shape: {o.shape}")
                print(f"p shape: {p.shape}")

            recon_loss = complex_mse_loss(batch, recon)
            loss = (
                recon_loss
                + 0.1 * nn.functional.mse_loss(o, p)
                + SPARSE_L1_WEIGHT * torch.mean(torch.abs(p))
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"epoch {epoch + 1} step {step} loss: {loss.item():.6f}")
            num_steps += 1
            epoch_recon_loss_total += recon_loss.item()
            input_amplitude = torch.abs(batch)
            recon_amplitude = torch.abs(recon)
            o_abs = torch.abs(o)
            p_abs = torch.abs(p)
            epoch_input_min_total += input_amplitude.min().item()
            epoch_input_max_total += input_amplitude.max().item()
            epoch_input_mean_total += input_amplitude.mean().item()
            epoch_recon_min_total += recon_amplitude.min().item()
            epoch_recon_max_total += recon_amplitude.max().item()
            epoch_recon_mean_total += recon_amplitude.mean().item()
            epoch_o_min_total += o.min().item()
            epoch_o_mean_total += torch.abs(o).mean().item()
            epoch_o_max_total += torch.abs(o).max().item()
            epoch_o_nonzero_ratio_total += (torch.abs(o) > 1e-3).float().mean().item()
            epoch_p_min_total += p.min().item()
            epoch_p_mean_total += p.mean().item()
            epoch_p_max_total += p.max().item()
            epoch_p_l1_total += torch.sum(p_abs).item()
            epoch_p_nonzero_ratio_total += (p_abs > 1e-2).float().mean().item()
            last_o_abs_top20 = torch.topk(o_abs.flatten(), k=20).values.detach().cpu()

            if step == 1:
                save_amplitude_image(
                    batch[0], str(output_dir / f"epoch_{epoch + 1}_input.png")
                )
                save_amplitude_image(
                    recon[0], str(output_dir / f"epoch_{epoch + 1}_recon_raw.png")
                )
                save_minmax_amplitude_image(
                    recon[0], str(output_dir / f"epoch_{epoch + 1}_recon_norm.png")
                )
                save_raw_heatmap_image(
                    o[0],
                    infer_square_side(o[0]),
                    str(output_dir / f"epoch_{epoch + 1}_o_raw.png"),
                )
                save_heatmap_image(
                    o[0],
                    infer_square_side(o[0]),
                    str(output_dir / f"epoch_{epoch + 1}_o_norm.png"),
                )
                save_histogram_image(
                    o[0], str(output_dir / f"epoch_{epoch + 1}_o_hist.png")
                )
                save_raw_heatmap_image(
                    p[0],
                    infer_square_side(p[0]),
                    str(output_dir / f"epoch_{epoch + 1}_p_raw.png"),
                )
                save_heatmap_image(
                    p[0],
                    infer_square_side(p[0]),
                    str(output_dir / f"epoch_{epoch + 1}_p_norm.png"),
                )

        print(f"epoch {epoch + 1} recon loss: {epoch_recon_loss_total / num_steps:.6f}")
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
            f"epoch {epoch + 1} o stats: min={epoch_o_min_total / num_steps:.6f}, "
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
            f"epoch {epoch + 1} p stats: min={epoch_p_min_total / num_steps:.6f}, "
            f"max={epoch_p_max_total / num_steps:.6f}, "
            f"mean={epoch_p_mean_total / num_steps:.6f}"
        )
        print(f"epoch {epoch + 1} p l1 norm: {epoch_p_l1_total / num_steps:.6f}")
        print(
            f"epoch {epoch + 1} p nonzero ratio (>1e-2): "
            f"{epoch_p_nonzero_ratio_total / num_steps:.6f}"
        )


if __name__ == "__main__":
    main()
