from functools import partial

import numpy as np
import gym

import torch as th
import torch.distributions as distr
from torch import nn


class BasicActorCritic(nn.Module):

    def __init__(self, ac_space, pi_module, pi_emb_size,
                 v_module=None, v_emb_size=None):
        super(BasicActorCritic, self).__init__()
        self.pi_module = pi_module
        self.v_module = v_module if v_module is not None else pi_module

        if isinstance(ac_space, gym.spaces.Tuple):
            ac_space = ac_space.spaces[0]

        if isinstance(ac_space, gym.spaces.Box):
            assert len(ac_space.shape) == 1
            n_acs = ac_space.shape[0]
            distr_scale = nn.Parameter(0.5 * th.ones(ac_space.shape[0]))
            self.ac_distr_params = nn.ParameterDict({'scale': distr_scale})
            self.ac_distr = partial(distr.normal.Normal, scale=distr_scale)
        elif isinstance(ac_space, gym.spaces.Discrete):
            n_acs = ac_space.n
            self.ac_distr = partial(distr.categorical.Categorical, None)
        else:
            raise NotImplementedError(f'ac_space {ac_space} must be Discrete or Box')

        self.pi_head = nn.Linear(pi_emb_size, n_acs)
        v_emb_size = v_emb_size or pi_emb_size
        self.v_head = nn.Linear(v_emb_size, 1)

    def forward(self, obs):
        obs = obs if type(obs) == th.Tensor else th.tensor(obs).float()
        pi_emb = self.pi_module(obs)
        action_distribution = self.ac_distr(self.pi_head(pi_emb))
        v_emb = self.v_module(obs)
        vpred = self.v_head(v_emb)

        return action_distribution, vpred


def mlp_model(env, layer_sizes=[64, 32]):
    obs_dim = np.prod(env.observation_space.shape[1:])

    policy_layers = []
    value_layers = []
    for i, units in enumerate(layer_sizes):
        if i == 0:
            policy_layers.extend([nn.Flatten(), nn.Linear(obs_dim, units), nn.Tanh()])
            value_layers.extend([nn.Flatten(), nn.Linear(obs_dim, units), nn.Tanh()])
        else:
            policy_layers.extend([nn.Linear(layer_sizes[i-1], units), nn.Tanh()])
            value_layers.extend([nn.Linear(layer_sizes[i-1], units), nn.Tanh()])

    policy_module = nn.Sequential(*policy_layers)
    value_module = nn.Sequential(*value_layers)

    return BasicActorCritic(env.action_space, pi_module=policy_module, pi_emb_size=layer_sizes[-1],
                            v_module=value_module, v_emb_size=layer_sizes[-1])
