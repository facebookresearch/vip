"""
This implements a shooting trajectory optimization algorithm.
The closest known algorithm is perhaps MPPI and hence we stick to that terminology.
Uses a filtered action sequence to generate smooth motions.
"""

import numpy as np
from mjrl.utils.logger import DataLog

from trajopt.algos.trajopt_base import Trajectory
from trajopt.utils.utils import gather_paths_parallel

class MPPI(Trajectory):
    def __init__(self, env, H, paths_per_cpu,
                 num_cpu=1,
                 kappa=1.0,
                 gamma=1.0,
                 mean=None,
                 filter_coefs=None,
                 default_act='repeat',
                 warmstart=True,
                 seed=123,
                 env_kwargs=None
                 ):
        self.env, self.seed = env, seed
        self.env_kwargs = env_kwargs
        self.n, self.m = env.observation_dim, env.action_dim
        self.H, self.paths_per_cpu, self.num_cpu = H, paths_per_cpu, num_cpu
        self.warmstart = warmstart

        self.logger = DataLog() 
        
        self.mean, self.filter_coefs, self.kappa, self.gamma = mean, filter_coefs, kappa, gamma
        if mean is None:
            self.mean = np.zeros(self.m)
        if filter_coefs is None:
            self.filter_coefs = [np.ones(self.m), 1.0, 0.0, 0.0]
        self.default_act = default_act

        self.sol_state = []
        self.sol_act = []
        self.sol_reward = []
        self.sol_obs = []
        self.sol_embedding = [] 
        self.sol_info = [] 

        self.env.reset()
        self.env.set_seed(seed)
        self.env.reset(seed=seed)
        self.sol_state.append(self.env.get_env_state().copy())
        self.sol_obs.append(self.env.get_obs())
        self.act_sequence = np.ones((self.H, self.m)) * self.mean
        self.init_act_sequence = self.act_sequence.copy()

        if env_kwargs['embedding_reward']:
            self.sol_embedding.append(self.env.env.get_views(embedding=True))

    def update(self, paths):
        num_traj = len(paths)
        act = np.array([paths[i]["actions"] for i in range(num_traj)])
        R = self.score_trajectory(paths)
        S = np.exp(self.kappa*(R-np.max(R)))

        # blend the action sequence
        weighted_seq = S*act.T
        act_sequence = np.sum(weighted_seq.T, axis=0)/(np.sum(S) + 1e-6)
        self.act_sequence = act_sequence

    def advance_time(self, act_sequence=None):
        act_sequence = self.act_sequence if act_sequence is None else act_sequence
        # accept first action and step
        action = act_sequence[0].copy()
        self.env.real_env_step(True)
        s, r, _, info = self.env.step(action)

        self.sol_act.append(action)
        self.sol_state.append(self.env.get_env_state().copy())
        self.sol_obs.append(self.env.get_obs())
        self.sol_reward.append(r)
        self.sol_info.append(info)

        if 'obs_embedding' in info:
            self.sol_embedding.append(self.env.env.get_views(embedding=True))    
        
        # get updated action sequence
        if self.warmstart:
            self.act_sequence[:-1] = act_sequence[1:]
            if self.default_act == 'repeat':
                self.act_sequence[-1] = self.act_sequence[-2]
            else:
                self.act_sequence[-1] = self.mean.copy()
        else:
            self.act_sequence = self.init_act_sequence.copy()

    def score_trajectory(self, paths):
        scores = np.zeros(len(paths))
        
        for i in range(len(paths)):
            scores[i] = paths[i]["rewards"][-1] # -V(s_T;g)
        return scores

    def do_rollouts(self, seed):
        paths = gather_paths_parallel(self.env,
                                      self.sol_state[-1],
                                      self.act_sequence,
                                      self.filter_coefs,
                                      seed,
                                      self.paths_per_cpu,
                                      self.num_cpu,
                                      env_kwargs=self.env_kwargs
                                      )
        return paths

    def train_step(self, niter=1):
        t = len(self.sol_state) - 1
        for _ in range(niter):
            paths = self.do_rollouts(self.seed+t)
            self.update(paths)
        self.advance_time()
