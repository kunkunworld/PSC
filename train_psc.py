from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from PIL import Image

from src.dataset import ComplexImageDataset
from src.psc_module import PSCModule


def complex_mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(x.real, y.real) + nn.functional.mse_loss(
        x.imag, y.imag
    )


def save_amplitude_image(tensor: torch.Tensor, path: str) -> None:
    amplitude = torch.abs(tensor.detach().cpu()).squeeze(0)
    amplitude = amplitude.clamp(0.0, 1.0)
    image = (amplitude.numpy() * 255.0).astype("uint8")
    Image.fromarray(image, mode="L").save(path)


def main() -> None:
    dataset = ComplexImageDataset(data_dir="data")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = PSCModule()
    optimizer = Adam(model.parameters(), lr=1e-3)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    for epoch in range(2):
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

            if step == 1:
                save_amplitude_image(
                    batch[0], str(output_dir / f"epoch_{epoch + 1}_input.png")
                )
                save_amplitude_image(
                    recon[0], str(output_dir / f"epoch_{epoch + 1}_recon.png")
                )


if __name__ == "__main__":
    main()
