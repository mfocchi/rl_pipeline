# jumpleg_rl
Jumpleg reinforcement learning agents based on TD3 author's Pytorch implementation
https://github.com/sfujim/TD3

# Action range
|Parameter| Value|
|--------|--------|
|max_time | 1|
|min_time | 0.2|
|max_velocity | 4|
|min_velocity | 0.1|
|max_extension | 0.32|
|min_extension | 0.25|
|min_phi | pi/4|
|min_phi_d | pi/6|

# Domain of targetCoM
|Parameter| Value|
|--------|--------|
|exp_rho | [-pi, pi]|
|exp_z | [0.25, 0.5]|
|exp_r | [0, 0.65]|

# RL
|Parameter| Value|
|--------|--------|
|layer_dim | 256 |
|lr | 5e-4|
