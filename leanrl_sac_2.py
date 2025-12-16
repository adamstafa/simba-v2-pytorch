# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule, TensorDictModule

# from stable_baselines3.common.buffers import ReplayBuffer
from torchrl.data import LazyTensorStorage, ReplayBuffer

from simba_v2.networks import SimbaV2Actor, SimbaV2Critic
from simba_v2.update import categorical_td_loss


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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 1e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-4
    """the learning rate of the Q network network optimizer"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""

    compile: bool = True
    """whether to use torch.compile."""
    cudagraphs: bool = True
    """whether to use cudagraphs on top of compile."""

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        # env = gym.wrappers.normalize.NormalizeObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, envs, n_act, n_obs):
        super().__init__()
        obs_dim = n_obs
        action_dim = n_act
        hidden_dim = 256
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

    def forward(self, x, a):
        # return q, log_prob
        return self.critic(x, a)
    
    def normalize_weights(self):
        self.critic.normalize_weights()

LOG_STD_MAX = 2
LOG_STD_MIN = -5

REWARD_SCALE = 0.0025


class Actor(nn.Module):
    def __init__(self, envs, n_act, n_obs):
        super().__init__()
        obs_dim = n_obs
        action_dim = n_act

        hidden_dim = 128
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
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        return self.get_action(x)

    def get_action(self, x):
        mean, log_std = self.actor(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def normalize_weights(self):
        self.actor.normalize_weights()

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
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"

    wandb.init(
        project="sac_continuous_action",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.set_default_device(device)  # TODO: remove default_device
    # optional speedup:
    # torch.set_float32_matmul_precision("high")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    n_act = math.prod(envs.single_action_space.shape)
    n_obs = math.prod(envs.single_observation_space.shape)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs, n_act=n_act, n_obs=n_obs)
    qf1 = SoftQNetwork(envs, n_act=n_act, n_obs=n_obs).to(device)
    qf2 = SoftQNetwork(envs, n_act=n_act, n_obs=n_obs).to(device)
    actor.normalize_weights()
    qf1.normalize_weights()
    qf2.normalize_weights()
    qf1_target = SoftQNetwork(envs, n_act=n_act, n_obs=n_obs).to(device)
    qf2_target = SoftQNetwork(envs, n_act=n_act, n_obs=n_obs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, capturable=args.cudagraphs and not args.compile)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    else:
        alpha = torch.as_tensor(args.alpha, device=device) * REWARD_SCALE

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    def update_qnets(data):
        with torch.no_grad():
            observations = data["observations"]
            actions = data["actions"]
            next_observations = data["next_observations"]
            rewards = data["rewards"].flatten()
            dones = data["dones"].flatten()

            next_state_actions, next_state_log_pi, _ = actor.get_action(next_observations)
            qf1_next_target, qf1_target_log_prob = qf1_target(next_observations, next_state_actions)
            qf2_next_target, qf2_target_log_prob = qf2_target(next_observations, next_state_actions)
            qf_stack = torch.stack([qf1_next_target, qf2_next_target], dim=0)
            qf_log_prob_stack = torch.stack([qf1_target_log_prob, qf2_target_log_prob], dim=0)
            min_idx = torch.argmin(qf_stack, dim=0)
            target_log_probs = qf_log_prob_stack[min_idx, torch.arange(qf_log_prob_stack.shape[1])]
            min_qf_next_target = qf_stack[min_idx, torch.arange(qf_stack.shape[1])] - alpha * next_state_log_pi
            next_q_value = rewards + (~ dones) * args.gamma * (min_qf_next_target)

        qf1_a_values, qf1_log_probs = qf1(observations, actions)
        qf2_a_values, qf2_log_probs = qf2(observations, actions)

        # MSE loss
        # qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        # qf2_loss = F.mse_loss(qf2_a_values, next_q_value)

        # Distributional loss
        qf1_loss = categorical_td_loss(qf1_log_probs, target_log_probs, rewards, dones, next_state_log_pi, alpha, args.gamma, 101, -5.0, 5.0)
        qf2_loss = categorical_td_loss(qf2_log_probs, target_log_probs, rewards, dones, next_state_log_pi, alpha, args.gamma, 101, -5.0, 5.0)
        qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach(), qvals=qf1_a_values.detach())

    def update_policy(data):
        pi, log_pi, _ = actor.get_action(data["observations"])
        qf1_pi, _ = qf1(data["observations"], pi)
        qf2_pi, _ = qf2(data["observations"], pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        if args.autotune:
            a_optimizer.zero_grad()
            with torch.no_grad():
                _, log_pi, _ = actor.get_action(data["observations"]) # TODO: already evaluated, use computed result
            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

            alpha_loss.backward()
            a_optimizer.step()

        alpha_loss = torch.tensor(0)
        return TensorDict(alpha=alpha.detach(), actor_loss=actor_loss.detach(), alpha_loss=alpha_loss.detach())
    
    def update_params(data):
        with torch.no_grad():
            # Normalize weights
            # TODO: evaluate the effect of weight normalization
            qf1.normalize_weights()
            qf2.normalize_weights()
            actor.normalize_weights()
            # update the target networks
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
    

    def extend_and_sample(transition):
        rb.extend(transition)
        return rb.sample(args.batch_size)
    
    def policy(obs):
        with torch.no_grad():
            return actor(obs)[0]

    is_extend_compiled = False
    if args.compile:
        mode = None
        update_qnets = torch.compile(update_qnets, mode=mode)
        update_policy = torch.compile(update_policy, mode=mode)
        update_params = torch.compile(update_params, mode=mode)
        policy = torch.compile(policy, mode=mode)

    if args.cudagraphs:
        update_qnets = CudaGraphModule(update_qnets, in_keys=[], out_keys=[])
        update_policy = CudaGraphModule(update_policy, in_keys=[], out_keys=[])
        update_params = CudaGraphModule(update_params, in_keys=[], out_keys=[])
        policy = CudaGraphModule(policy)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs, device=device, dtype=torch.float)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""

    for global_step in pbar:
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = policy(obs).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        rewards *= REWARD_SCALE

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                r = float(info["episode"]["r"])
                max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)
            desc = (
                f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"
            )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float)
        real_next_obs = next_obs.clone()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = torch.as_tensor(infos["final_observation"][idx], device=device, dtype=torch.float)
        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_tensor(actions, device=device, dtype=torch.float),
            rewards=torch.as_tensor(rewards, device=device, dtype=torch.float),
            terminations=terminations,
            dones=terminations,
            batch_size=obs.shape[0],
            device=device,
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        data = extend_and_sample(transition)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            out = update_qnets(data)
            out.update(update_policy(data))
            update_params(data)

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                wandb.log(
                    {
                        "losses/actor_loss": out["actor_loss"].mean(),
                        "losses/critic_loss": out["qf_loss"].mean(),
                        "losses/alpha_loss": out.get("alpha_loss", 0),
                        "gradients/actor_norm": grad_norm(actor_optimizer),
                        "gradients/critic_norm": grad_norm(q_optimizer),
                        "monitoring/alpha": alpha,
                        "monitoring/q_mean": out["qvals"].mean(),
                        "monitoring/q_min": out["qvals"].min(),
                        "monitoring/q_max": out["qvals"].max(),
                        "episode_return": torch.tensor(avg_returns).mean(),
                        "speed": speed,
                    },
                    step=global_step,
                )

    envs.close()