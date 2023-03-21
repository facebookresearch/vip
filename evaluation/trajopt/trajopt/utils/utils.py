"""
Utility functions useful for TrajOpt algos
"""
import copy
import numpy as np
import multiprocessing as mp
import concurrent.futures
from mjrl.samplers.core import _try_multiprocess
from trajopt.envs.obs_wrappers import env_constructor
from trajopt.utils import tensor_utils
from trajopt.envs.utils import get_environment

def do_env_rollout(env, start_state, act_list, env_kwargs=None):
    """
        1) Construt env based on desired behavior and set to start_state.
        2) Generate rollouts using act_list.
           act_list is a list with each element having size (H,m).
           Length of act_list is the number of desired rollouts.
    """
    e = env
    # e = copy.deepcopy(env)
    # e = env_constructor(**env_kwargs)
    e.reset()
    e.real_env_step(False)
    paths = []
    H = act_list[0].shape[0]
    N = len(act_list)

    for i in range(N):
        e.set_env_state(start_state)
        obs = []
        act = []
        rewards = []
        env_infos = []
        states = []

        for k in range(H):
            obs.append(e.get_obs())
            act.append(act_list[i][k])
            env_infos.append(e.get_env_infos())
            states.append(e.get_env_state())
            s, r, d, ifo = e.step(act[-1])
            rewards.append(r)

        path = dict(observations=np.array(obs),
                    actions=np.array(act),
                    rewards=np.array(rewards),
                    env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
                    states=states)
        paths.append(path)

    return paths


def discount_sum(x, gamma, discounted_terminal=0.0):
    """
    discount sum a sequence with terminal value
    """
    y = []
    run_sum = discounted_terminal
    for t in range( len(x)-1, -1, -1):
        run_sum = x[t] + gamma*run_sum
        y.append(run_sum)

    return np.array(y[::-1])


def generate_perturbed_actions(base_act, filter_coefs):
    """
    Generate perturbed actions around a base action sequence
    """
    sigma, beta_0, beta_1, beta_2 = filter_coefs
    eps = np.random.normal(loc=0, scale=1.0, size=base_act.shape) * sigma
    for i in range(2, eps.shape[0]):
        eps[i] = beta_0*eps[i] + beta_1*eps[i-1] + beta_2*eps[i-2]
    return base_act + eps


def generate_paths(env, start_state, N, base_act, filter_coefs, 
                    base_seed, env_kwargs, *args, **kwargs):
    """
    first generate enough perturbed actions
    then do rollouts with generated actions
    set seed inside this function for multiprocessing
    """
    np.random.seed(base_seed)
    act_list = []
    for i in range(N):
        act = generate_perturbed_actions(base_act, filter_coefs)
        act_list.append(act)
    paths = do_env_rollout(env, start_state, act_list, env_kwargs)
    return paths


def gather_paths_parallel(env, start_state, base_act, filter_coefs, base_seed, 
                          paths_per_cpu, num_cpu=None, env_kwargs=None,
                          *args, **kwargs):
    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == 'max' else num_cpu
    assert type(num_cpu) == int

    if num_cpu == 1:
        input_dict = dict(env=env, start_state=start_state, N=paths_per_cpu, env_kwargs=env_kwargs,
                          base_act=base_act, filter_coefs=filter_coefs, base_seed=base_seed)
        return generate_paths(**input_dict)

    # do multiprocessing if necessary
    input_dict_list = []
    for i in range(num_cpu):
        cpu_seed = base_seed + i*paths_per_cpu
        input_dict = dict(env=env, start_state=start_state, N=paths_per_cpu, env_kwargs=env_kwargs,
                          base_act=base_act, filter_coefs=filter_coefs, base_seed=cpu_seed)
        input_dict_list.append(input_dict)

    results = _try_multiprocess(func=generate_paths, input_dict_list=input_dict_list,
                                    num_cpu=num_cpu, max_process_time=2400, max_timeouts=4)
    paths = []
    for result in results:
        for path in result:
            paths.append(path)

    return paths
