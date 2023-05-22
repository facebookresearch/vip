# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import namedtuple
import numpy as np
import gym
from gym.spaces.box import Box
import glob
import omegaconf
import torch
import torch.nn as nn
from torch.nn.modules.linear import Identity
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from PIL import Image
from pathlib import Path
import pickle
from torchvision.utils import save_image
import hydra
import os
import sys 
from trajopt.envs.gym_env import GymEnv

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def _get_embedding(embedding_name='resnet50', load_path="", *args, **kwargs):
    if load_path == "random":
        prt = False
    else:
        prt = True
    if embedding_name == 'resnet34':
        model = models.resnet34(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == 'resnet18':
        model = models.resnet18(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == 'resnet50':
        model = models.resnet50(pretrained=prt, progress=False)
        embedding_dim = 2048
    else:
        print("Requested model not available currently")
        raise NotImplementedError
    # make FC layers to be identity
    # NOTE: This works for ResNet backbones but should check if same
    # template applies to other backbone architectures
    model.fc = Identity()
    model = model.eval()
    return model, embedding_dim

def env_constructor(env_name, device='cuda', image_width=256, image_height=256,
                    camera_name=None, embedding_name='resnet50', pixel_based=True,
                    embedding_reward=True,
                    render_gpu_id=0, load_path="", proprio=False, goal_timestep=49, init_timestep=0):
    # print("Constructing environment with GPU", render_gpu_id)
    if not pixel_based and not embedding_reward: 
            env = GymEnv(env_name)
    else:
        env = gym.make(env_name)
        ## Wrap in pixel observation wrapper
        env = MuJoCoPixelObs(env, width=image_width, height=image_height, 
                           camera_name=camera_name, device_id=render_gpu_id)
        ## Wrapper which encodes state in pretrained model (additionally compute reward)
        env = StateEmbedding(env, embedding_name=embedding_name, device=device, load_path=load_path, 
                        proprio=proprio, camera_name=camera_name,
                         env_name=env_name, pixel_based=pixel_based, 
                         embedding_reward=embedding_reward,
                          goal_timestep=goal_timestep, init_timestep=init_timestep)
        env = GymEnv(env)
    return env

class ClipEnc(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, im):
        e = self.m.encode_image(im)
        return e


class StateEmbedding(gym.ObservationWrapper):
    """
    This wrapper places a convolution model over the observation.

    From https://pytorch.org/vision/stable/models.html
    All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    where H and W are expected to be at least 224.

    Args:
        env (Gym environment): the original environment,
        embedding_name (str, 'baseline'): the name of the convolution model,
        device (str, 'cuda'): where to allocate the model.

    """
    def __init__(self, env, embedding_name=None, device='cuda', load_path="", checkpoint="",
    proprio=0,
     camera_name=None, env_name=None, pixel_based=True, embedding_reward=False,
      goal_timestep=49, init_timestep=0):
        gym.ObservationWrapper.__init__(self, env)

        self.env_name = env_name 
        self.cameras = [camera_name]
        self.camera_name = self.cameras[0]

        self.proprio = proprio
        self.load_path = load_path
        self.start_finetune = False

        if "vip" in load_path:
            print(f"Loading pre-trained {load_path} model!")
            from vip import load_vip 
            rep = load_vip()
            rep.eval()
            embedding_dim = rep.module.hidden_dim
            embedding = rep
            self.transforms = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()]) # ToTensor() divides by 255
        elif "r3m" in load_path:
            print(f"Loading pre-trained {load_path} model!")
            from r3m import load_r3m_reproduce
            rep = load_r3m_reproduce(load_path)
            rep.eval()
            embedding_dim = rep.module.outdim
            embedding = rep
            self.transforms = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()]) # ToTensor() divides by 255        
        elif load_path == "clip":
            import clip
            model, cliptransforms = clip.load("RN50", device="cuda")
            embedding = ClipEnc(model)
            embedding.eval()
            embedding_dim = 1024
            self.transforms = cliptransforms
        elif (load_path == "random") or (load_path == "resnet"):
                embedding, embedding_dim = _get_embedding(load_path=load_path)
                self.transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(), # ToTensor() divides by 255
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            raise NameError("Invalid Model")
        embedding.eval()

        if device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device
        embedding.to(device=device)

        self.embedding, self.embedding_dim = embedding, embedding_dim
        self.pixel_based = pixel_based
        self.embedding_reward = embedding_reward 
        self.init_state = None
        if self.pixel_based:
            self.observation_space = Box(
                        low=-np.inf, high=np.inf, shape=(self.embedding_dim+self.proprio,))
        else:
            self.observation_space = self.env.unwrapped.observation_space

        if self.embedding_reward: 
            self.init_timestep = init_timestep
            self.goal_timestep = goal_timestep
            
            # evaluation information
            from trajopt import DEMO_PATHS
            demopath = DEMO_PATHS[self.env_name] 
            demo_id = demopath[-1] 
            traj_path = demopath[:-1] + f'traj_{demo_id}.pickle'
            demo = pickle.load(open(traj_path, 'rb'))
            self.goal_robot_pose = demo.sol_info[self.goal_timestep]['obs_dict']['robot_jnt']
            self.goal_object_pose = demo.sol_info[self.goal_timestep]['obs_dict']['objs_jnt']
            self.goal_end_effector = demo.sol_info[self.goal_timestep]['obs_dict']['end_effector']
 
            self.goal_embedding = {} 
            for camera in self.cameras:
                
                # mj_envs MPPI demo for goal embedding 
                if init_timestep != 0:
                    self.init_state = {}
                    for key in demo.sol_state[init_timestep]:
                        self.init_state[key] = demo.sol_state[init_timestep][key]
                    self.init_state['env_timestep'] = init_timestep + 1 

                video_paths = [demopath + f'/{camera}']
                num_vid = len(video_paths)
                end_goals = [] 
                for i in range(num_vid):
                    vid = f"{video_paths[i]}"
                    img = Image.open(f"{vid}/{self.goal_timestep}.png")
                    cur_dir = os.getcwd() 
                    img.save(f"{cur_dir}/goal_image_{camera}.png") # save goal image
                    end_goals.append(img)
                
                # hack to get when there is only one goal image working
                if len(end_goals) == 1:
                    end_goals.append(end_goals[-1])
                
                goal_embedding = self.encode_batch(end_goals)
                self.goal_embedding[camera] = goal_embedding.mean(axis=0) 

    def observation(self, observation):
        ### INPUT SHOULD BE [0,255]
        if self.embedding is not None and len(observation.shape) > 1:
            if isinstance(observation, np.ndarray):
                o = Image.fromarray(observation.astype(np.uint8))

            inp = self.transforms(o).reshape(-1, 3, 224, 224)
            if  "vip" in self.load_path or "r3m" in self.load_path:
                inp *= 255.0
            inp = inp.to(self.device)

            with torch.no_grad():                
                emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()

            ## IF proprioception add it to end of embedding
            if self.proprio:
                try:
                    proprio = self.env.unwrapped.get_obs()[:self.proprio]
                except:
                    proprio = self.env.unwrapped._get_obs()[:self.proprio]
                emb = np.concatenate([emb, proprio])

            return emb
        else:
            return observation

    def encode_batch(self, obs, finetune=False):
        ### INPUT SHOULD BE [0,255]
        inp = []
        for o in obs:
            if isinstance(o, np.ndarray):
                o = Image.fromarray(o.astype(np.uint8))
            o = self.transforms(o).reshape(-1, 3, 224, 224)
            if "vip" in self.load_path or "r3m" in self.load_path:
                o *= 255.0
            inp.append(o)

        inp = torch.cat(inp)
        inp = inp.to(self.device)
        if finetune and self.start_finetune:
            emb = self.embedding(inp).view(-1, self.embedding_dim)
        else:
            with torch.no_grad():
                emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
        return emb

    def get_obs(self):
        if self.embedding is not None and self.pixel_based:
            return self.observation(self.env.observation(None))
        else:
            return self.env.unwrapped.get_obs()
    def get_views(self, embedding=False):
        views = {}
        embeddings = {}
        for camera in self.cameras:
            view = self.env.get_image(camera_name=camera)
            views[camera] = view
            if embedding:
                embeddings[camera] = self.observation(view)
        if embedding:
            return embeddings 
        return views  

    def start_finetuning(self):
        self.start_finetune = True
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action) 
        obs_embedding = self.observation(observation)
        info['obs_embedding'] = obs_embedding 
        if self.embedding_reward:
            rewards = []
            # Note: only single camera evaluation is supported 
            for camera in self.cameras:
                img_camera = self.env.get_image(camera_name=camera)
                obs_embedding_camera = self.observation(img_camera)
                obs_embedding_camera = obs_embedding_camera if self.proprio == 0 else obs_embedding_camera[:-self.proprio]
                reward_camera = -np.linalg.norm(obs_embedding_camera-self.goal_embedding[camera])
                rewards.append(reward_camera) 
            # some state-based info for evaluating learned reward func.
            if 'end_effector' in info['obs_dict']:
                info['obs_dict']['ee_error'] = np.linalg.norm(self.goal_end_effector-info['obs_dict']['end_effector'])
            if 'hand_jnt' in info['obs_dict']:
                info['obs_dict']['robot_error'] = np.linalg.norm(self.goal_robot_pose-info['obs_dict']['hand_jnt'])
            elif 'robot_jnt' in info['obs_dict']:
                info['obs_dict']['robot_error'] = np.linalg.norm(self.goal_robot_pose-info['obs_dict']['robot_jnt'])
            if 'objs_jnt' in info['obs_dict']:
                info['obs_dict']['objs_error'] = np.linalg.norm(self.goal_object_pose-info['obs_dict']['objs_jnt'])
            
            reward = min(rewards)
        if not self.pixel_based:
            state = self.env.unwrapped.get_obs()
        else: 
            state = obs_embedding 

        return state, reward, done, info
    
    def reset(self):
        observation = self.env.reset()
        try:
            if self.init_state is not None:
                self.env.set_env_state(self.init_state)
        except Exception as e:
            print("Resetting Initial State Error")
            print("Unexpected error:", sys.exc_info()[0])
            print(e)
        if not self.pixel_based:
            observation = self.env.unwrapped.get_obs()
        else:
            observation = self.observation(observation) # This is needed for IL, but would it break other evaluations?
        return observation

class MuJoCoPixelObs(gym.ObservationWrapper):
    def __init__(self, env, width, height, camera_name, device_id=-1, depth=False, *args, **kwargs):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0., high=255., shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id
        if "v2" in env.spec.id:
            self.get_obs = env._get_obs

    def get_image(self, camera_name=None):
        if camera_name is None:
            camera_name = self.camera_name
        if camera_name == "default" or camera_name == "all":
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
             device_id=self.device_id)
        else:
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                            camera_name=camera_name, device_id=self.device_id)
        img = img[::-1,:,:]
        return img

    def observation(self, observation=None):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        return self.get_image()
        