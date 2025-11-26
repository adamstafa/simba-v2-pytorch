from simba_v2.networks import SimbaV2Actor, SimbaV2Critic
import torch

def test_actor():
    obs_dim = 10
    action_dim = 4
    batch = 256
    
    actor = SimbaV2Actor(
        num_blocks=2,
        hidden_dim=64,
        c_shift=3.0,
        obs_dim=obs_dim,
        action_dim=action_dim,
        scaler_init=1,
        scaler_scale=2,
        alpha_init=3,
        alpha_scale=4,
    )

    input = torch.zeros(obs_dim)
    dist, info = actor(observations=input)
    sampled_action = dist.sample()

    print("ACTOR")
    print(dist)
    print(info)
    print(sampled_action)
    print()


    batch_input = torch.zeros((batch, obs_dim))
    dist, info = actor(observations=batch_input)
    sampled_action = dist.sample()

    print("ACTOR BATCH")
    print(dist)
    print(info)
    print(sampled_action)
    print()
    

def test_critic():
    obs_dim = 10
    action_dim = 4
    batch = 256

    critic = SimbaV2Critic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_blocks=2,
        hidden_dim=64,
        num_bins=101,
        min_v=-5,
        max_v=5,
        scaler_init=1,
        scaler_scale=2,
        alpha_init=3,
        alpha_scale=4,
        c_shift=3.0
    )

    obs = torch.zeros(obs_dim)
    act = torch.zeros(action_dim)
    q, info = critic(obs, act)

    print("CRITIC")
    print(q)
    print(info)
    print()
    
    batch_obs = torch.zeros((batch, obs_dim))
    batch_act = torch.zeros((batch, action_dim))
    q, info = critic(batch_obs, batch_act)

    print("CRITIC BATCH")
    print(q)
    print(info)

if __name__ == "__main__":
    test_actor()
    test_critic()