from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ComplexImageDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        image_size: int = 80,
        use_center_crop: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.use_center_crop = use_center_crop
        self.image_paths = self._collect_image_paths()
        self._print_count = 0

        if not self.image_paths:
            raise ValueError(f"No png/jpg images found in {self.data_dir}")

    def _collect_image_paths(self) -> List[Path]:
        patterns: Sequence[str] = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG")
        image_paths: List[Path] = []
        for pattern in patterns:
            image_paths.extend(self.data_dir.glob(pattern))
        return sorted(set(image_paths))

    def __len__(self) -> int:
        return len(self.image_paths)

    def _resize_or_crop(self, image: Image.Image) -> Image.Image:
        if self.use_center_crop:
            width, height = image.size
            crop_size = min(width, height)
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            image = image.crop((left, top, left + crop_size, top + crop_size))
        return image.resize((self.image_size, self.image_size), Image.BILINEAR)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("L")
            image = self._resize_or_crop(image)
            amplitude = np.asarray(image, dtype=np.float32) / 255.0

        phase = np.random.uniform(
            low=0.0,
            high=2.0 * np.pi,
            size=(self.image_size, self.image_size),
        ).astype(np.float32)

        complex_img = amplitude * np.exp(1j * phase)
        tensor = torch.from_numpy(complex_img.astype(np.complex64)).unsqueeze(0)

        if self._print_count < 3:
            print(f"dataset item shape={tensor.shape}, dtype={tensor.dtype}")
            self._print_count += 1

        return tensor
