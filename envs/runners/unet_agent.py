import glob
import os
import platform
import random
import math
from collections import deque
from itertools import zip_longest
from typing import Dict, Iterable, Optional, Tuple, Union
import time

from .unet import UNetModel, UNetModel_Q, SquashedNormal
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import random
import matplotlib.pyplot as plt
from torch.distributions import Normal
from .buffers import ReplayBuffer


from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)

def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo

class unetAgent(object):
    """
    env generating agent with score-based model style
    """
    def __init__(self, args=None, in_channels=1,
        model_channels = 32,
        out_channels=1,
        num_res_blocks=1,
        attention_resolutions=[8, 4],
        dropout=0,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        learning_rate=1e-4,
        optimizer="AdamW",
        beta_schedule="linear",
        num_workers=32,
        state_dim=169,
        action_dim=169,
        env_name="Minigrid",
        gamma = 0.995,
        initial_temperature = 0.1,
        tau = 0.05,
        max_step = 100,
        pre_train_epoch=100):
        
        if args is None:
            self.device = "cuda:0"
        else:
            if not args.no_cuda:
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        
        def build_unet():
            return UNetModel(in_channels=in_channels,
                        model_channels=model_channels,
                        out_channels=out_channels,
                        num_res_blocks=num_res_blocks,
                        attention_resolutions=attention_resolutions,
                        dropout=dropout,
                        channel_mult=channel_mult,
                        conv_resample=conv_resample,
                        dims=dims,
                        num_classes=num_classes,
                        use_checkpoint=use_checkpoint,
                        num_heads=num_heads,
                        num_heads_upsample=num_heads_upsample,
                        use_scale_shift_norm=use_scale_shift_norm)
        def build_unetQ():
            return UNetModel_Q(in_channels=in_channels,
                        model_channels=model_channels,
                        out_channels=out_channels,
                        num_res_blocks=num_res_blocks,
                        attention_resolutions=attention_resolutions,
                        dropout=dropout,
                        channel_mult=channel_mult,
                        conv_resample=conv_resample,
                        dims=dims,
                        num_classes=num_classes,
                        use_checkpoint=use_checkpoint,
                        num_heads=num_heads,
                        num_heads_upsample=num_heads_upsample,
                        use_scale_shift_norm=use_scale_shift_norm,
                        env_name=env_name)
        # self.algo = SAC
        # self.storage = storage        
        
        ## build actor(policy)
        self.policy_model = build_unet().to(self.device)
        
        if optimizer == "AdamW":
            self.policy_optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=learning_rate)
            
        
        ## std schedule for NCSN style generation
        self.epsilon = 3e-5
        self.std_schedule = torch.linspace(0.3, 3e-5, 100)
        
        ## Replay Buffer
        
        # num_workers = args.num_processes
        self.replay_buffer = ReplayBuffer(buffer_size=4000, state_dim=state_dim, action_dim=action_dim, n_envs=num_workers, optimize_memory_usage=False, device=self.device)
        
        ## Build Q networks

        
        def build_Q():
            network = nn.Sequential(build_unetQ(), linear(channel_mult[-1] * model_channels * 4 * 2, 128), SiLU(), linear(128,1))
            return network

        
        self.Q1_network = build_unetQ().to(self.device)
        self.Q2_network = build_unetQ().to(self.device)
        self.Q1_target_network = build_unetQ().to(self.device)
        self.Q2_target_network = build_unetQ().to(self.device)
        self.Q1_target_network.load_state_dict(self.Q1_network.state_dict())
        self.Q2_target_network.load_state_dict(self.Q2_network.state_dict())        
        
        self.Q1_optimizer = torch.optim.AdamW(self.Q1_network.parameters(), lr=learning_rate)
        self.Q2_optimizer = torch.optim.AdamW(self.Q2_network.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.log_alpha = torch.tensor(np.log(initial_temperature))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        self.target_entropy = -action_dim
        
        self.tau = tau
        self.batch_size = num_workers
        self.env = env_name
        
        if env_name == "Minigrid":
            self.n_row = int(math.sqrt(state_dim))
        
        self.max_timestep = max_step
        self.pre_train_epoch = pre_train_epoch
        # self.pre_train_epoch = 1
        
        self.state_dim = state_dim
        self.action_dim = action_dim
    def update(self):
        
        replay_data = self.replay_buffer.sample(self.batch_size)
        
        obs = replay_data.observations.clone().detach()
        next_obs = replay_data.next_observations.clone().detach()
        actions = replay_data.actions.clone().detach()
        timesteps = replay_data.timesteps.clone().detach()
        
        if self.env == "Minigrid":
            obs = obs.reshape(obs.shape[0], 1, int(math.sqrt(obs.shape[1])), int(math.sqrt(obs.shape[1])))
            next_obs = next_obs.reshape(obs.shape[0], 1, int(math.sqrt(next_obs.shape[1])), int(math.sqrt(next_obs.shape[1])))
            actions = actions.reshape(actions.shape[0], 1, int(math.sqrt(actions.shape[1])), int(math.sqrt(actions.shape[1])))
            timesteps = timesteps.squeeze()
            
        with torch.no_grad():
            next_mu, next_log_std = self.policy_model(next_obs, timesteps + 1)
            next_action_dist = SquashedNormal(next_mu, next_log_std.exp())
            next_action = next_action_dist.sample()
            next_log_prob = next_action_dist.log_prob(next_action)
            if len(next_log_prob.shape) > 1:
                next_log_prob = next_log_prob.sum(dim=2).sum(dim=2)
            else:
                next_log_prob = next_log_prob.sum(dim=1).unsqueeze(dim=1)
            # next_action, next_log_prob = self.act(next_obs.clone().cpu().numpy(), timesteps.clone().cpu().numpy() + 1)
            next_Q_target_1 = self.Q1_target_network(next_obs, next_action, timesteps + 1)
            next_Q_target_2 = self.Q2_target_network(next_obs, next_action, timesteps + 1)
            next_Q_target = torch.min(next_Q_target_1, next_Q_target_2)
            
            y = replay_data.rewards + self.gamma * (next_Q_target - self.log_alpha.exp().detach() * next_log_prob)
        
        Q1 = self.Q1_network(obs, actions, timesteps)
        Q2 = self.Q2_network(obs, actions, timesteps)
        
        self.Q1_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()
        Q1_loss = F.mse_loss(Q1, y)
        Q2_loss = F.mse_loss(Q2, y)
        Q1_loss.backward()
        Q2_loss.backward()
        self.Q1_optimizer.step()
        self.Q2_optimizer.step()
        
        current_mu, current_log_std = self.policy_model(obs, timesteps)
        current_action_dist = SquashedNormal(current_mu, current_log_std.exp())
        current_action = current_action_dist.rsample()
        current_log_prob = current_action_dist.log_prob(current_action)
        if len(current_log_prob.shape) > 2:
            current_log_prob = current_log_prob.sum(dim=2).sum(dim=2)
        else:
            current_log_prob = current_log_prob.sum(dim=1).unsqueeze(dim=1)
        # current_action, current_log_prob = self.act(obs.clone().cpu().numpy(), timesteps.clone().cpu().numpy())
        Q1 = self.Q1_network(obs, current_action, timesteps)
        Q2 = self.Q2_network(obs, current_action, timesteps)
        Q = torch.min(Q1, Q2)
        
        self.policy_optimizer.zero_grad()
        actor_loss = (self.log_alpha.exp().detach() * current_log_prob - Q).mean()
        actor_loss.backward()
        self.policy_optimizer.step()
        
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.log_alpha.exp() * (- current_log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        for target_param, param in zip_strict(self.Q1_target_network.parameters(), self.Q1_network.parameters()):
            target_param.data.mul_(1 - self.tau)
            torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
            
        for target_param, param in zip_strict(self.Q2_target_network.parameters(), self.Q2_network.parameters()):
            target_param.data.mul_(1 - self.tau)
            torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
            
        return
    
    def act(self, x, t):
        
        t = np.clip(t, 0, self.max_timestep - 1)
        
        x = torch.from_numpy(x.astype(np.float32))
        t = torch.from_numpy(t.astype(int))
        
        x = x.to(self.device)
        t = t.to(self.device)
        
        mu, log_std = self.policy_model(x,t)
        log_std = -5 + 0.5 * (2 - (-5)) * (log_std +1)
        
        action_dist = SquashedNormal(mu, log_std.exp())
        dx = action_dist.sample()
        log_prob = action_dist.log_prob(dx)
        dx = torch.clamp(dx, -0.1, 0.1)
        # std = self.std_schedule[t]
        
        # if len(mu.shape) == 4:
        #     # for minigrid
        #     std = torch.einsum('ijkw,i->ijkw',torch.ones(mu.shape), self.std_schedule[t])
        # elif len(mu.shape) == 3:
        #     # for minigrid
        #     std = torch.einsum('ijk,i->ijk',torch.ones(mu.shape), self.std_schedule[t])
            
        # std = std.to(self.device)
        # normal_layer = Normal(mu, std)
        # dx = normal_layer.sample()
        # log_prob = normal_layer.log_prob(dx)
        # log_prob = log_prob.reshape(log_prob.shape[0],-1,1)
        
        if len(log_prob.shape) > 2:
            log_prob = log_prob.sum(dim=2).sum(dim=2)
        else:
            log_prob = log_prob.sum(dim=1).unsqueeze(dim=1)
            

        # dx = torch.clamp(dx, -1, 1)
        return dx, log_prob
    
    def random_act(self):
        new_env= np.random.randn(self.state_dim) * 0.3
        
        return np.clip(new_env, -1, 1)
    def get_Q_values(self, obs, action, t):
        
        
        Q1 = self.Q1_network(obs, action, t)
        Q2 = self.Q2_network(obs, action, t)

        return Q1, Q2
    
    def easy_act(self, state, timestep):
        '''
        get action when state: np.ndarray(self.batch_size, self.state_dim), timestep: np.ndarray(self.batch_size) are given
        '''
        
        if self.env == "Minigrid":
            x = state.reshape(self.batch_size, 1, self.n_row, self.n_row)
        
        dx, log_prob = self.act(x, timestep)
        
        dx = dx.reshape(self.batch_size, self.state_dim)
        
        return dx, log_prob
    
    def unsup_pre_train(self):
        
        for e in range(self.pre_train_epoch):
            state = np.random.randn(self.batch_size * self.state_dim) * 0.3
            state = np.clip(state, -1, 1)
            state = state.reshape((self.batch_size, self.state_dim))
            for t in range(self.max_timestep):
                if self.env == "Minigrid":
                    action, log_prob = self.easy_act(state, np.array([t] * self.batch_size))
                else:
                    action, log_prob = self.act(state, np.array([t] * self.batch_size))
                    
                
                action = action.clone().detach().cpu().numpy()
                action = action.reshape((self.batch_size, self.action_dim))
                
                next_state = state + action
                
                next_state = np.clip(next_state, -1, 1)
                                 
                timesteps = np.array([t] * self.batch_size)
                
                dones = np.zeros(self.batch_size)
                
                reward = self.replay_buffer.reward_for_pre_training(next_state, 5)
                                
                self.replay_buffer.add(state, next_state, action, reward, timesteps, dones, None)
                
                self.update()
                                
                state = next_state
        
        self.replay_buffer.reset()
                
                # self.pre_replay_buffer.add(state, next_state, action, )
    
    def save_model(self, path):
        torch.save({
            'policy_model': self.policy_model.state_dict(),
            'Q1_network': self.Q1_network.state_dict(),
            'Q2_network': self.Q2_network.state_dict(),
            'Q1_target_network': self.Q1_target_network.state_dict(),
            'Q2_target_network': self.Q2_target_network.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'Q1_optimizer': self.Q1_optimizer.state_dict(),
            'Q2_optimizer': self.Q2_optimizer.state_dict()
        }, path)
    
    def load_model(self,ckpt):
        self.policy_model.load_state_dict(ckpt['policy_model'])
        self.Q1_network.load_state_dict(ckpt['Q1_network'])
        self.Q2_network.load_state_dict(ckpt['Q2_network'])
        self.Q1_target_network.load_state_dict(ckpt['Q1_target_network'])
        self.Q2_target_network.load_state_dict(ckpt['Q2_target_network'])
        self.policy_optimizer.load_state_dict(ckpt['policy_optimizer'])
        self.Q1_optimizer.load_state_dict(ckpt['Q1_optimizer'])
        self.Q2_optimizer.load_state_dict(ckpt['Q2_optimizer'])
        
    def train_mode(self):
        self.policy_model.train()
        self.Q1_network.train()
        self.Q2_network.train()
        self.Q1_target_network.train()
        self.Q2_target_network.train()
        
    def eval_mode(self):
        self.policy_model.eval()
        self.Q1_network.eval()
        self.Q2_network.eval()
        self.Q1_target_network.eval()
        self.Q2_target_network.eval()
        
    def to(self, device):
        self.policy_model.to(device)
        self.Q1_network.to(device)
        self.Q2_network.to(device)
        self.Q1_target_network.to(device)
        self.Q2_target_network.to(device)
                
                
                
        