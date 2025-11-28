# Based on CleanRL implementation of SAC
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py

import os
import random
import time
import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from buffers import ReplayBuffer

from simba_v2.networks import SimbaV2Actor, SimbaV2Critic
from simba_v2.update import categorical_td_loss

torch.set_default_device("cuda:1")
REWARD_SCALE = 0.005

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = "adam-stafa-org"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: True)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = env.single_observation_space.shape[0]
        action_dim = env.single_action_space.shape[0]
        hidden_dim = 64
        num_blocks = 2
        scaler_init = math.sqrt(2 / hidden_dim)
        scaler_scale = math.sqrt(2 / hidden_dim)
        alpha_init = 1 / (num_blocks + 1)
        alpha_scale = 1 / math.sqrt(hidden_dim)
        c_shift = 3.0
        
        self.critic = SimbaV2Critic(
            num_blocks=num_blocks,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
            c_shift=c_shift,
            num_bins=101,
            min_v=-5.0,
            max_v=5.0)

    def forward(self, x, a, return_log_probs=False):
        q, info = self.critic(x, a)

        if return_log_probs:
            return q, info['log_prob']
        return q

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = env.single_observation_space.shape[0]
        action_dim = env.single_action_space.shape[0]

        hidden_dim = 64
        num_blocks = 1
        scaler_init = math.sqrt(2 / hidden_dim)
        scaler_scale = math.sqrt(2 / hidden_dim)
        alpha_init = 1 / (num_blocks + 1)
        alpha_scale = 1 / math.sqrt(hidden_dim)
        c_shift = 3.0

        self.actor = SimbaV2Actor(num_blocks=num_blocks, hidden_dim=hidden_dim, obs_dim=obs_dim, action_dim=action_dim,scaler_init=scaler_init, scaler_scale=scaler_scale, alpha_init=alpha_init, alpha_scale=alpha_scale, c_shift=c_shift)

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        return self.get_action(x)

    def get_action(self, x):
        dist, info = self.actor(x)
        sample = dist.rsample()
        log_prob = dist.log_prob(sample)
        
        return sample * self.action_scale + self.action_bias, log_prob, info # TODO: maybe return mean as third value?


def grad_norm(optimizer):
    total = 0.0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            total += param_norm.item() ** 2
    return math.sqrt(total)


if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:1" if torch.cuda.is_available() and args.cuda else "cpu")
    # device = torch.device("cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        rewards = rewards * REWARD_SCALE

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, action_info = actor.get_action(data.next_observations)
                qf1_next_target, qf1_target_log_prob = qf1_target(data.next_observations, next_state_actions, return_log_probs=True)
                qf2_next_target, qf2_target_log_prob = qf2_target(data.next_observations, next_state_actions, return_log_probs=True)

                qf_stack = torch.stack([qf1_next_target, qf2_next_target], dim=0)
                qf_log_prob_stack = torch.stack([qf1_target_log_prob, qf2_target_log_prob], dim=0)
                min_idx = torch.argmin(qf_stack, dim=0)
                
                target_log_probs = qf_log_prob_stack[min_idx, torch.arange(qf_log_prob_stack.shape[1])]

                rewards = data.rewards.flatten()
                dones = data.dones.flatten()

            qf1_a_values, qf1_log_probs = qf1(data.observations, data.actions, return_log_probs=True)
            qf2_a_values, qf2_log_probs = qf2(data.observations, data.actions, return_log_probs=True)
            qf1_loss = categorical_td_loss(qf1_log_probs, target_log_probs, rewards, dones, next_state_log_pi, args.gamma, 101, -5.0, 5.0)
            qf2_loss = categorical_td_loss(qf1_log_probs, target_log_probs, rewards, dones, next_state_log_pi, args.gamma, 101, -5.0, 5.0)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item() / REWARD_SCALE, global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item() / REWARD_SCALE, global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("gradients/actor_norm", grad_norm(actor_optimizer), global_step)
                writer.add_scalar("gradients/qf_norm", grad_norm(q_optimizer), global_step)
                wandb.log({"backups/action_log_std": action_info['log_std'], "global_step": global_step})
                wandb.log({"backups/action_mean": action_info['mean'], "global_step": global_step})
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    envs.close()
    writer.close()