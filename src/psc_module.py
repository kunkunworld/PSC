import torch
import torch.nn.functional as F
from torch import nn

from src.config import H, W
from src.psc_dictionary import build_psc_dictionary, summarize_psi


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
    def __init__(self, dictionary_debug: bool = True, debug_size: int = 32) -> None:
        super().__init__()
        self._print_count = 0
        self.stages = nn.ModuleList([HQSStage(), HQSStage()])
        self.dictionary_debug = dictionary_debug
        self.dictionary_size = debug_size if dictionary_debug else H
        self.register_buffer(
            "psi",
            build_psc_dictionary(debug=dictionary_debug, debug_size=debug_size),
        )

    def _dictionary_reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        x_real = x.real.to(torch.float32)
        x_imag = x.imag.to(torch.float32)

        real_small = F.interpolate(
            x_real,
            size=(self.dictionary_size, self.dictionary_size),
            mode="bilinear",
            align_corners=False,
        )
        imag_small = F.interpolate(
            x_imag,
            size=(self.dictionary_size, self.dictionary_size),
            mode="bilinear",
            align_corners=False,
        )
        coeffs = torch.complex(real_small, imag_small).flatten(start_dim=1)
        recon_flat = coeffs @ self.psi
        recon_small = recon_flat.view(
            x.shape[0], 1, self.dictionary_size, self.dictionary_size
        )
        recon_real = F.interpolate(
            recon_small.real,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        recon_imag = F.interpolate(
            recon_small.imag,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        return torch.complex(recon_real, recon_imag)

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

    def get_psi_diagnostics(self) -> dict[str, float | str | tuple[int, ...]]:
        return summarize_psi(self.psi)
