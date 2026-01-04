import torch
import torch.nn as nn

class RunningMeanStd(nn.Module):
    def __init__(self, shape):
        super().__init__()

        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("m2", torch.zeros(shape, dtype=torch.float32))  # Second moment
        self.register_buffer("std", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0.0))

    @torch.no_grad()
    def update(self, x):
        batch_size = x.shape[0]
        self.count += batch_size
        self.m2 += (x.square().sum(dim=0) - batch_size * self.m2) / self.count
        self.mean.copy_(self.mean + (x.sum(dim=0) - batch_size * self.mean) / self.count)
        self.std.copy_(torch.sqrt(self.m2 - self.mean.square() + 1e-8))


class ObservationNormalizer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.rms = RunningMeanStd(shape)

    def update(self, obs):
        self.rms.update(obs)

    def forward(self, obs):
        return (obs - self.rms.mean) / self.rms.std

    def apply_(self, obs):
        obs.sub_(self.rms.mean).div_(self.rms.std)


class RewardNormalizer(nn.Module):
    def __init__(self, gamma, max_v, num_envs):
        super().__init__()

        self.register_buffer("eps", torch.tensor(1e-2))
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("max_v", torch.tensor(max_v))
        self.register_buffer("discounts", torch.ones(num_envs, dtype=torch.float32))
        self.register_buffer("ep_returns", torch.zeros(num_envs, dtype=torch.float32))
        self.register_buffer("max_return", torch.tensor(0.0))
        self.register_buffer("reward_scale", torch.tensor(1.0))

    @torch.no_grad()
    def update(self, rewards, dones):
        self.ep_returns += rewards * self.discounts
        self.discounts *= self.gamma
        self.max_return.copy_(torch.max(self.max_return, self.ep_returns.abs().max()))
        self.reward_scale.copy_(self.max_v / self.max_return.clamp_min(self.eps))
        self.discounts[dones] = 1.0
        self.ep_returns[dones] = 0.0

    def forward(self, rewards):
        return rewards * self.reward_scale

    def apply_(self, rewards):
        rewards.mul_(self.reward_scale)
