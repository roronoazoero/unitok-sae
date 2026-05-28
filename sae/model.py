import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """TopK SAE with exact sparsity control via top-k activation function."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 8192,
        k: int = 64,
        dead_threshold_steps: int = 10_000_000,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k

        # Learned mean-centering bias (subtracted before encode, added after decode)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        # Dead-feature tracking buffers
        self.register_buffer("steps_since_activation", torch.zeros(hidden_dim))
        self.register_buffer("activation_count", torch.zeros(hidden_dim))
        self.dead_threshold = dead_threshold_steps

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.contiguous().clone()
            self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        """Unit-normalize decoder weight columns."""
        norms = self.decoder.weight.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.data /= norms

    @torch.no_grad()
    def remove_parallel_grad(self):
        """Project out grad component parallel to decoder columns (preserves unit norm)."""
        if self.decoder.weight.grad is None:
            return
        w = self.decoder.weight.data
        g = self.decoder.weight.grad
        parallel = (g * w).sum(dim=0, keepdim=True) * w
        self.decoder.weight.grad = g - parallel

    def encode(self, x: torch.Tensor):
        """
        Args:
            x: (..., input_dim)
        Returns:
            latent: (..., hidden_dim) sparse with exactly k nonzeros
            topk_indices: (..., k)
            topk_values: (..., k)
        """
        x_centered = x - self.pre_bias
        pre_acts = self.encoder(x_centered)
        topk_values, topk_indices = torch.topk(pre_acts, k=self.k, dim=-1)
        topk_values = F.relu(topk_values)
        latent = torch.zeros_like(pre_acts)
        latent.scatter_(-1, topk_indices, topk_values)
        return latent, topk_indices, topk_values

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent) + self.pre_bias

    def forward(self, x: torch.Tensor) -> dict:
        x_centered = x - self.pre_bias
        pre_acts = self.encoder(x_centered)
        latent, topk_indices, topk_values = self._encode_from_preacts(pre_acts) # pseudo-code for optimization, to be implemented later
        reconstruction = self.decode(latent)
        recon_loss = F.mse_loss(reconstruction, x)

        # Auxiliary loss: have dead features predict the residual (revives them)
        with torch.no_grad():
            dead_mask = self.steps_since_activation > self.dead_threshold
        aux_loss = torch.tensor(0.0, device=x.device)
        n_dead = int(dead_mask.sum().item())
        if self.training and n_dead > 0:
            residual = (x - reconstruction).detach()
            x_centered = x - self.pre_bias
            pre_acts = self.encoder(x_centered)
            k_aux = min(2 * self.k, n_dead)
            dead_pre_acts = pre_acts.masked_fill(~dead_mask, float("-inf"))
            aux_vals, aux_idx = torch.topk(dead_pre_acts, k=k_aux, dim=-1)
            aux_vals = F.relu(aux_vals)
            aux_latent = torch.zeros_like(pre_acts)
            aux_latent.scatter_(-1, aux_idx, aux_vals)
            aux_recon = self.decoder(aux_latent)
            aux_loss = F.mse_loss(aux_recon, residual)

        if self.training:
            self._update_dead_tracking(topk_indices)

        return {
            "reconstruction": reconstruction,
            "latent": latent,
            "topk_indices": topk_indices,
            "topk_values": topk_values,
            "recon_loss": recon_loss,
            "aux_loss": aux_loss,
            "n_dead": n_dead,
        }

    @torch.no_grad()
    def _update_dead_tracking(self, indices: torch.Tensor):
        self.steps_since_activation += 1
        unique_idx = torch.unique(indices.reshape(-1))
        self.steps_since_activation[unique_idx] = 0
        self.activation_count[unique_idx] += 1

    def get_dead_mask(self) -> torch.Tensor:
        return self.steps_since_activation > self.dead_threshold
