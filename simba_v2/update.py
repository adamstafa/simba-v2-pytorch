
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
    bin_values = torch.linspace(
        start=min_v, stop=max_v, num=num_bins).reshape(1, -1)
    target_bin_values = reward + gamma * \
        (bin_values - actor_entropy) * (1.0 - done)
    target_bin_values = torch.clip(
        target_bin_values, min_v, max_v)  # (B, num_bins)

    # update indices
    b = (target_bin_values - min_v) / ((max_v - min_v) / (num_bins - 1))
    l = torch.floor(b)
    l_mask = F.one_hot(
        l.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
    u = torch.ceil(b)
    u_mask = F.one_hot(
        u.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))

    # target label
    _target_probs = torch.exp(target_log_probs)
    m_l = (_target_probs * (u + (l == u).astype(torch.float32) - b)).reshape(
        -1, num_bins,
    )
    m_u = (_target_probs * (b - l)).reshape((-1, num_bins, 1))
    target_probs = torch.sum(m_l * l_mask + m_u * u_mask, dim=1)

    # cross entropy loss
    loss = -torch.mean(torch.sum(target_probs * pred_log_probs, dim=1))

    return loss


# def update_critic(
#     actor: SimbaV2Actor,
#     critic: SimbaV2Critic,
#     target_critic: SimbaV2Critic,
#     temperature: float,
#     batch: dict,
#     use_cdq: bool,
#     min_v: float,
#     max_v: float,
#     num_bins: int,
#     gamma: float,
#     n_step: int,
# ) -> dict[str, float]:
#     # compute the target q-value
#     next_dist, _ = actor(observations=batch["next_observation"])
#     next_actions = next_dist.sample()
#     next_actor_log_probs = next_dist.log_prob(next_actions)
#     next_actor_entropy = temperature * next_actor_log_probs

#     if use_cdq:
#         # next_qs: (2, n)
#         # next_q_infos['log_prob]: (2, n, num_bins)
#         # next_q_log_probs: (n, num_bins)
#         next_qs, next_q_infos = target_critic(
#             observations=batch["next_observation"], actions=next_actions
#         )
#         min_indices = next_qs.argmin(dim=0)
#         next_q_log_probs = torch.vmap(
#             lambda log_prob, idx: log_prob[idx], in_axes=(1, 0)
#         )(next_q_infos["log_prob"], min_indices)
#     else:
#         next_q, next_q_info = target_critic(
#             observations=batch["next_observation"],
#             actions=next_actions,
#         )
#         next_q_log_probs = next_q_info["log_prob"]

#     def critic_loss_fn(
#         critic_params: flax.core.FrozenDict[str, Any],
#     ) -> Tuple[jnp.ndarray, Dict[str, float]]:
#         pred_qs, pred_q_infos = critic.apply(
#             variables={"params": critic_params},
#             observations=batch["observation"],
#             actions=batch["action"],
#         )
#         loss_1 = categorical_td_loss(
#             pred_log_probs=pred_q_infos["log_prob"][0],
#             target_log_probs=next_q_log_probs,
#             reward=batch["reward"],
#             done=batch["terminated"],
#             actor_entropy=next_actor_entropy,
#             gamma=gamma**n_step,
#             num_bins=num_bins,
#             min_v=min_v,
#             max_v=max_v,
#         )
#         loss_2 = categorical_td_loss(
#             pred_log_probs=pred_q_infos["log_prob"][1],
#             target_log_probs=next_q_log_probs,
#             reward=batch["reward"],
#             done=batch["terminated"],
#             actor_entropy=next_actor_entropy,
#             gamma=gamma**n_step,
#             num_bins=num_bins,
#             min_v=min_v,
#             max_v=max_v,
#         )
#         critic_loss = (loss_1 + loss_2).mean()

#         critic_info = {
#             "critic/loss": critic_loss,
#             "critic/batch_rew_min": batch["reward"].min(),
#             "critic/batch_rew_mean": batch["reward"].mean(),
#             "critic/batch_rew_max": batch["reward"].max(),
#         }

#         return critic_loss, critic_info

#     critic, info = critic.apply_gradient(critic_loss_fn)
#     critic = l2normalize_network(critic)

#     return critic, info


# def update_target_network(
#     network: Network,
#     target_network: Network,
#     target_tau: bool,
# ) -> Tuple[Network, Dict[str, float]]:
#     new_target_params = jax.tree_map(
#         lambda p, tp: p * target_tau + tp * (1 - target_tau),
#         network.params,
#         target_network.params,
#     )
#     target_network = target_network.replace(params=new_target_params)

#     info = {}

#     return target_network, info


# def update_temperature(
#     temperature: Network, entropy: float, target_entropy: float
# ) -> Tuple[Network, Dict[str, float]]:
#     def temperature_loss_fn(
#         temperature_params: flax.core.FrozenDict[str, Any],
#     ) -> Tuple[jnp.ndarray, Dict[str, float]]:
#         temperature_value = temperature.apply({"params": temperature_params})
#         temperature_loss = temperature_value * (entropy - target_entropy).mean()
#         temperature_info = {
#             "temperature/value": temperature_value,
#             "temperature/loss": temperature_loss,
#         }

#         return temperature_loss, temperature_info

#     temperature, info = temperature.apply_gradient(temperature_loss_fn)

#     return temperature, info
