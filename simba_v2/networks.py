import torch
import torch.nn as nn


from simba_v2.layers import (
    HyperCategoricalValue,
    HyperEmbedder,
    HyperLERPBlock,
    HyperNormalTanhPolicy,
)


class SimbaV2Actor(nn.Module):
    def __init__(self,
                 num_blocks: int,
                 hidden_dim: int,
                 obs_dim: int,
                 action_dim: int,
                 scaler_init: float,
                 scaler_scale: float,
                 alpha_init: float,
                 alpha_scale: float,
                 c_shift: float):
        super().__init__()
        self.embedder = HyperEmbedder(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift
        )
        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )
                for _ in range(num_blocks)
            ]
        )
        self.predictor = HyperNormalTanhPolicy(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(self, observations: torch.tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedder(observations)
        x = self.encoder(x)
        return self.predictor(x)

    def normalize_weights(self):
        self.embedder.normalize_weights()
        for block in self.encoder:
            block.normalize_weights()
        self.predictor.normalize_weights()


class SimbaV2Critic(nn.Module):
    def __init__(self, num_blocks: int,
                 obs_dim: int, action_dim: int,
                 hidden_dim: int,
                 scaler_init: float,
                 scaler_scale: float,
                 alpha_init: float,
                 alpha_scale: float,
                 c_shift: float,
                 num_bins: int,
                 min_v: float,
                 max_v: float):
        super().__init__()
        self.embedder = HyperEmbedder(
            input_dim=obs_dim + action_dim,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
        )
        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )
                for _ in range(num_blocks)
            ]
        )

        self.predictor = HyperCategoricalValue(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(self, observations: torch.tensor, actions: torch.tensor) -> tuple[torch.tensor, dict]:
        x = torch.concatenate((observations, actions), dim=-1)
        x = self.embedder(x)
        x = self.encoder(x)
        return self.predictor(x)
    
    def normalize_weights(self):
        self.embedder.normalize_weights()
        for block in self.encoder:
            block.normalize_weights()
        self.predictor.normalize_weights()
