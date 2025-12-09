import math
import torch
import torch.nn as nn


def l2normalize(x: torch.tensor, dim: int, eps: float = 1e-8) -> torch.tensor:
    l2norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    return x / l2norm.clamp_min(eps)


class Scaler(nn.Module):
    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.init = init
        self.scale = scale

        self.scaler = nn.Parameter(torch.ones(dim) * self.scale)
        self.forward_scaler = self.init / self.scale

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.scaler * self.forward_scaler * x


class HyperDense(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(
            self.input_dim,
            self.output_dim,
            bias=False)

        nn.init.orthogonal_(self.linear.weight, gain=1.0)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.linear(x)
    
    def normalize_weights(self):
        eps = 1e-8
        weight_norm = torch.norm(self.linear.weight.data, p=2, dim=1, keepdim=True)
        self.linear.weight.data /= weight_norm.clamp(eps)


class HyperMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 scaler_init: float, scaler_scale: float, eps: float = 1e-8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.scaler_init = scaler_init
        self.scaler_scale = scaler_scale
        self.eps = eps

        self.w1 = HyperDense(self.input_dim, self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim,
                             self.scaler_init,
                             self.scaler_scale)
        self.w2 = HyperDense(self.hidden_dim, self.output_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.w1(x)
        x = self.scaler(x)
        x = torch.relu(x) + self.eps
        x = self.w2(x)
        x = l2normalize(x, dim=-1)
        return x

    def normalize_weights(self):
        self.w1.normalize_weights()
        self.w2.normalize_weights()

class HyperEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, scaler_init: float,
                 scaler_scale: float, c_shift: float):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scaler_init = scaler_init
        self.scaler_scale = scaler_scale
        self.c_shift = c_shift

        self.w = HyperDense(self.input_dim + 1,
                            self.hidden_dim)  # +1 for the shift
        self.scaler = Scaler(self.hidden_dim,
                             self.scaler_init,
                             self.scaler_scale)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # TODO: maybe pass the device in the constructor?
        new_axis = torch.ones((x.shape[:-1] + (1,)), device=x.device) * self.c_shift
        x = torch.concatenate([x, new_axis], axis=-1)
        x = l2normalize(x, dim=-1)
        x = self.w(x)
        x = self.scaler(x)
        x = l2normalize(x, dim=-1)

        return x

    def normalize_weights(self):
        self.w.normalize_weights()


class HyperLERPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, scaler_init: float,
                 scaler_scale: float, alpha_init: float, alpha_scale: float,
                 expansion: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scaler_init = scaler_init
        self.scaler_scale = scaler_scale
        self.alpha_init = alpha_init
        self.alpha_scale = alpha_scale
        self.expansion = expansion

        self.mlp = HyperMLP(
            input_dim=self.input_dim,
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim * self.expansion,
            scaler_init=self.scaler_init / math.sqrt(self.expansion),
            scaler_scale=self.scaler_scale / math.sqrt(self.expansion)
        )
        self.alpha_scaler = Scaler(
            self.hidden_dim,
            init=self.alpha_init,
            scale=self.alpha_scale,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        residual = x
        x = self.mlp(x)
        x = residual + self.alpha_scaler(x - residual)
        x = l2normalize(x, dim=-1)

        return x

    def normalize_weights(self):
        self.mlp.normalize_weights()


class HyperNormalTanhPolicy(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 scaler_init: float,
                 scaler_scale: float,
                 log_std_min: float = -10.0,
                 log_std_max: float = 2.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.scaler_init = scaler_init
        self.scaler_scale = scaler_scale
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.mean_w1 = HyperDense(self.input_dim, self.hidden_dim)
        self.mean_scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)
        self.mean_w2 = HyperDense(self.hidden_dim, self.action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(self.action_dim))

        self.std_w1 = HyperDense(self.input_dim, self.hidden_dim)
        self.std_scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)
        self.std_w2 = HyperDense(self.hidden_dim, self.action_dim)
        self.std_bias = nn.Parameter(torch.zeros(self.action_dim))

    def forward(self, x: torch.tensor, temperature: float = 1.0) -> torch.distributions.Distribution:
        mean = self.mean_w1(x)
        mean = self.mean_scaler(mean)
        mean = self.mean_w2(mean) + self.mean_bias

        log_std = self.std_w1(x)
        log_std = self.std_scaler(log_std)
        log_std = self.std_w2(log_std) + self.std_bias

        # normalize log-stds for stability
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1 + torch.tanh(log_std)
        )

        # N(mu, exp(log_sigma))
        dist = torch.distributions.MultivariateNormal(
            loc=mean,
            scale_tril=torch.diag_embed(torch.exp(log_std) * temperature),
        )

        # tanh(N(mu, sigma))
        dist = torch.distributions.TransformedDistribution(
            dist,
            transforms=[torch.distributions.transforms.TanhTransform()]
        )

        info = {'log_std': log_std, 'mean': torch.tanh(mean), 'mean_normal': mean}
        return dist, info
    
    def normalize_weights(self):
        self.mean_w1.normalize_weights()
        self.mean_w2.normalize_weights()
        self.std_w1.normalize_weights()
        self.std_w2.normalize_weights()


class HyperCategoricalValue(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_bins: int, min_v: float, max_v: float,
                 scaler_init: float, scaler_scale: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.min_v = min_v
        self.max_v = max_v
        self.scaler_init = scaler_init
        self.scaler_scale = scaler_scale

        self.w1 = HyperDense(input_dim, self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)
        self.w2 = HyperDense(self.hidden_dim, self.num_bins)
        self.bias = nn.Parameter(torch.zeros(self.num_bins))
        self.bin_values = torch.linspace(
            start=self.min_v, end=self.max_v, steps=self.num_bins
        ).reshape(1, -1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        value = self.w1(x)
        value = self.scaler(value)
        value = self.w2(value) + self.bias

        log_prob = self.log_softmax(value)
        value = torch.sum(torch.exp(log_prob) * self.bin_values, dim=-1)

        info = {"log_prob": log_prob}
        return value, info
    
    def normalize_weights(self):
        self.w1.normalize_weights()
        self.w2.normalize_weights()
