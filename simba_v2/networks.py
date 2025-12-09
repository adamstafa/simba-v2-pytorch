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
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.scaler_init = scaler_init
        self.scaler_scale = scaler_scale
        self.alpha_init = alpha_init
        self.alpha_scale = alpha_scale
        self.c_shift = c_shift

        self.embedder = HyperEmbedder(
            input_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift
        )
        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    input_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    scaler_init=self.scaler_init,
                    scaler_scale=self.scaler_scale,
                    alpha_init=self.alpha_init,
                    alpha_scale=self.alpha_scale,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.predictor = HyperNormalTanhPolicy(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
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
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.input_dim = self.obs_dim + self.action_dim
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.scaler_init = scaler_init
        self.scaler_scale = scaler_scale
        self.alpha_init = alpha_init
        self.alpha_scale = alpha_scale
        self.c_shift = c_shift
        self.num_bins = num_bins
        self.min_v = min_v
        self.max_v = max_v

        self.embedder = HyperEmbedder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )
        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    input_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    scaler_init=self.scaler_init,
                    scaler_scale=self.scaler_scale,
                    alpha_init=self.alpha_init,
                    alpha_scale=self.alpha_scale,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.predictor = HyperCategoricalValue(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_bins=self.num_bins,
            min_v=self.min_v,
            max_v=self.max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(self, observations: torch.tensor, actions: torch.tensor) -> tuple[torch.tensor, dict]:
        x = torch.concatenate((observations, actions), dim=-1)
        y = self.embedder(x)
        z = self.encoder(y)
        q, info = self.predictor(z)
        return q, info
    
    def normalize_weights(self):
        self.embedder.normalize_weights()
        for block in self.encoder:
            block.normalize_weights()
        self.predictor.normalize_weights()


# class SimbaV2DoubleCritic(nn.Module):
#     """
#     Vectorized Double-Q for Clipped Double Q-learning.
#     https://arxiv.org/pdf/1802.09477v3
#     """

#     num_blocks: int
#     hidden_dim: int
#     scaler_init: float
#     scaler_scale: float
#     alpha_init: float
#     alpha_scale: float
#     c_shift: float
#     num_bins: int
#     min_v: float
#     max_v: float

#     num_qs: int = 2

#     @nn.compact
#     def __call__(
#         self,
#         observations: jnp.ndarray,
#         actions: jnp.ndarray,
#     ) -> jnp.ndarray:
#         VmapCritic = nn.vmap(
#             SimbaV2Critic,
#             variable_axes={"params": 0},
#             split_rngs={"params": True},
#             in_axes=None,
#             out_axes=0,
#             axis_size=self.num_qs,
#         )

#         qs, infos = VmapCritic(
#             num_blocks=self.num_blocks,
#             hidden_dim=self.hidden_dim,
#             scaler_init=self.scaler_init,
#             scaler_scale=self.scaler_scale,
#             alpha_init=self.alpha_init,
#             alpha_scale=self.alpha_scale,
#             c_shift=self.c_shift,
#             num_bins=self.num_bins,
#             min_v=self.min_v,
#             max_v=self.max_v,
#         )(observations, actions)

#         return qs, infos
