
import torch
import torch.nn.functional as F

from simba_v2.networks import SimbaV2Actor, SimbaV2Critic

EPS = 1e-8


def l2normalize_layer(tree):
    pass


def l2normalize_network():
    pass


def update_actor(
        actor: SimbaV2Actor,
        critic: SimbaV2Critic,
        batch: dict,  # TODO: specify type
        temperature: float):

    def actor_loss_fn(actor_params) -> tuple[torch.tensor, dict[str, float]]:
        dist, _ = actor(observations=batch["observation"])

        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # qs: (2, n)
        qs, q_infos = critic(
            observations=batch["observation"], actions=actions)
        q = torch.minimum(qs[0], qs[1])
        actor_loss = (log_probs * temperature() - q).mean()

        actor_info = {
            "actor/loss": actor_loss,
            "actor/entropy": -log_probs.mean(),
            "actor/mean_action": torch.mean(actions),
        }
        return actor_loss, actor_info

    # TODO: compute loss
    # TODO: apply optimizer
    # TODO: l2 normalize weights


def categorical_td_loss(
    pred_log_probs: torch.tensor,  # (n, num_bins)
    target_log_probs: torch.tensor,  # (n, num_bins)
    reward: torch.tensor,  # (n,)
    done: torch.tensor,  # (n,)
    actor_entropy: torch.tensor,  # (n,)
    alpha: float,
    gamma: float,
    num_bins: int,
    min_v: float,
    max_v: float,
) -> torch.tensor:
    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)

    # compute target value buckets
    # target_bin_values: (n, num_bins)
    bin_values = torch.linspace(start=min_v, end=max_v, steps=num_bins, device=pred_log_probs.device).reshape(1, -1)
    target_bin_values = reward + gamma * (bin_values - alpha * actor_entropy) * ~done
    target_bin_values = torch.clip(target_bin_values, min_v, max_v)  # (B, num_bins)

    # update indices
    b = (target_bin_values - min_v) / ((max_v - min_v) / (num_bins - 1))
    l = torch.floor(b).long()
    # l = torch.clamp(l, 0, num_bins - 1)  # TODO: investigate why this happens
    l_mask = F.one_hot(l.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
    u = torch.ceil(b).long()
    # u = torch.clamp(u, 0, num_bins - 1)  # TODO: investigate why this happens
    u_mask = F.one_hot(u.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))

    # target label
    _target_probs = torch.exp(target_log_probs).reshape(-1, num_bins)
    m_l = (_target_probs * (u + (l == u) - b)).reshape(-1, num_bins, 1)
    m_u = (_target_probs * (b - l)).reshape((-1, num_bins, 1))

    target_probs = torch.sum(m_l * l_mask + m_u * u_mask, dim=1)

    # cross entropy loss
    loss = -torch.mean(torch.sum(target_probs * pred_log_probs, dim=1))

    return loss

