
# Introduction

In order to explore hyperparameter tuning, I decided to cobble together implementations for a couple of hyperparameter tuning methods. In this package, I provide an implementation of PPO and DQN using a vector input and discrete output. I provide implementations of random search as well as Population-Based Training (PBT).

# Usage

My implementation is currently only setup to run on systems with CUDA. However, it should be reasonable to convert for use on CPUs, and if I have time I will go back to make changes.

To use PBT, run `python run_pbt.py`. Random search can be used with `python run_random_search.py`. For command-line options, run with `--help`.

# Details

My PBT implementation works by spawning multiple threads that each manage a single work member. This was done because I was having issues with sharing CUDA data between processes. Every few rounds, it will exploit a better solution if it doesn't meet sufficient performance requirements.

The random search algorithm uses concurrent processes that each manage their own training process. I had originally done this to look at the multiprocessing library but ended up using apply_async.

Some of the hyperparameter bounds are still manually configured. These can be seen in the setup of the ConfigurationGenerators in `pbt/train` and `random_search/train`.

# Papers

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)
- [Population Based Training of Neural Networks](https://arxiv.org/pdf/1711.09846.pdf)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)

