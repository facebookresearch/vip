import gym
import mjrl.envs
import trajopt.envs
from mjrl.utils.gym_env import GymEnv

def get_environment(env, env_kwargs=None):
    # get the correct env behavior
    if type(env) == str:
        env = GymEnv(env)
    elif isinstance(env, GymEnv):
        env = env
    elif callable(env):
        env = env(**env_kwargs)
    else:
        print("Unsupported environment format")
        raise AttributeError
    return env