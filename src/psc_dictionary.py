import torch


def build_debug_dictionary(size: int = 32) -> torch.Tensor:
    dim = size * size
    psi = torch.eye(dim, dtype=torch.float32)
    print(f"debug Psi shape: {psi.shape}, dtype: {psi.dtype}")
    return psi
