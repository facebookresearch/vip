"""
Base trajectory class
"""

import numpy as np

class Trajectory:
    def __init__(self, env, H=32, seed=123):
        self.env, self.seed = env, seed
        self.n, self.m, self.H = env.observation_dim, env.action_dim, H

        # following need to be populated by the trajectory optimization algorithm
        self.sol_state = []
        self.sol_act = []
        self.sol_reward = []
        self.sol_obs = []

        self.env.reset_model(seed=self.seed)
        self.sol_state.append(self.env.get_env_state())
        self.sol_obs.append(self.env._get_obs())
        self.act_sequence = np.zeros((self.H, self.m))

    def update(self, paths):
        """
        This function should accept a set of trajectories
        and must update the solution trajectory
        """
        raise NotImplementedError

    def animate_rollout(self, t, act):
        """
        This function starts from time t in the solution trajectory
        and animates a given action sequence
        """
        self.env.set_env_state(self.sol_state[t])
        for k in range(act.shape[0]):
            try:
                self.env.env.env.mujoco_render_frames = True
            except AttributeError:
                self.env.render()
            self.env.set_env_state(self.sol_state[t+k])
            self.env.step(act[k])
            print(self.env.env_timestep)
            print(self.env.real_step)
        try:
            self.env.env.env.mujoco_render_frames = False
        except:
            pass

    def animate_result(self):
        self.env.reset(self.seed)
        self.env.set_env_state(self.sol_state[0])
        for k in range(len(self.sol_act)):
            self.env.env.env.mujoco_render_frames = True
            self.env.render()
            self.env.step(self.sol_act[k])
        self.env.env.env.mujoco_render_frames = False

    def animate_result_offscreen(self,camera_name=None):
        if camera_name == "default":
            camera_name = None 

        frames = []
        self.env.reset(self.seed)
        self.env.set_env_state(self.sol_state[0])
        for t in range(len(self.sol_act)):
            frame_t = generate_frame(self.env, frame_size=(256,256), camera_name=camera_name)
            frames.append(frame_t.copy())
            self.env.step(self.sol_act[t])
        return frames


def generate_frame(e, frame_size, camera_name):
    env_id = e.env_id
    if env_id.startswith('dmc'):
        frame = e.env.unwrapped.render(mode='rgb_array', width=frame_size[0], height=frame_size[1])
    else:
        frame = e.env.unwrapped.sim.render(width=frame_size[0], height=frame_size[1],
                                            mode='offscreen', camera_name=camera_name, device_id=0)
        frame = frame[::-1,:,:]
    return frame.copy()