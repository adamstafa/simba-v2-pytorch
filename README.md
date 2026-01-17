# SimbaV2 - PyTorch
Optimized PyTorch implementation of the paper [Hyperspherical Normalization for Scalable Deep Reinforcement Learning
](https://arxiv.org/pdf/2502.15280).

The architecture is inspired by the paper authors' [JAX implmenetation](https://github.com/DAVIAN-Robotics/SimbaV2) and the underlying SAC implementation is adapted from [LeanRL](https://github.com/meta-pytorch/LeanRL/blob/main/leanrl/sac_continuous_action_torchcompile.py).

## Features
- Faithful implementation of the paper
- Heavily optimized using torch.compile and CUDA graphs
- Fast training (1 million timesteps in 1 hour on RTX 4060 Ti)
- Easy to modify 

## Differences to the original JAX version
- The rewards are normalized by the maximum discounted episodic return (same performance as the original version but more interpretable)
