import torch
import torch.nn.functional as F
from torch import nn

from src.psc_dictionary import build_debug_dictionary


class HQSStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.rho = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.mu = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(
        self, z: torch.Tensor, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_z = z.flatten(start_dim=1)
        flat_obs = observation.flatten(start_dim=1)

        data_term = flat_z - flat_obs
        prior_term = flat_z

        # Simplified real-valued HQS-like update for structure only.
        o = flat_z - self.t * data_term
        p = o / (1.0 + self.rho.abs()) - self.mu * torch.tanh(o)
        updated = p.view_as(z)
        return updated, o, p


class PSCModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._print_count = 0
        self.stages = nn.ModuleList([HQSStage(), HQSStage()])
        self.dictionary_size = 32
        self.register_buffer("psi", build_debug_dictionary(size=self.dictionary_size))

    def _dictionary_reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        x_real = x.real.to(torch.float32)
        downsampled = F.interpolate(
            x_real,
            size=(self.dictionary_size, self.dictionary_size),
            mode="bilinear",
            align_corners=False,
        )
        coeffs = downsampled.flatten(start_dim=1)
        recon_flat = coeffs @ self.psi.transpose(0, 1)
        recon_small = recon_flat.view(
            x.shape[0], 1, self.dictionary_size, self.dictionary_size
        )
        recon_real = F.interpolate(
            recon_small,
            size=(x.shape[-2], x.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        return torch.complex(recon_real, x.imag.to(torch.float32))

    def forward(self, x: torch.Tensor):
        real_observation = x.real.to(torch.float32)
        z = real_observation
        o = None
        p = None

        for stage in self.stages:
            z, o, p = stage(z, real_observation)

        hqs_recon = torch.complex(z, x.imag.to(torch.float32))
        dict_recon = self._dictionary_reconstruct(x)
        recon = 0.5 * hqs_recon + 0.5 * dict_recon

        if self._print_count < 2:
            print(
                f"PSCModule forward: input={x.shape}, recon={recon.shape}, "
                f"o={o.shape}, p={p.shape}"
            )
            self._print_count += 1

        return recon.to(torch.complex64), o, p
