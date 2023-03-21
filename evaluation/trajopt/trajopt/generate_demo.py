"""
This is a launcher script for launching mjrl training using hydra
"""
import numpy as np
import os
import time as timer
import hydra
from omegaconf import OmegaConf
import pickle 
from tqdm import tqdm 
import multiprocessing as mp
from moviepy.editor import ImageSequenceClip
import skvideo.io
import os 
from PIL import Image 

from mj_envs.envs.env_variants import register_env_variant
from trajopt.envs.obs_wrappers import env_constructor
from trajopt.algos.mppi import MPPI

# ===============================================================================
# Process Inputs and configure job
# ===============================================================================
@hydra.main(config_name="mppi_config", config_path="config")
def configure_jobs(job_data):

    # Replace OUT_DIR with your absolute path!
    OUT_DIR = f"/home/exx/Projects/vip/evaluation/dataset/{job_data['env']}"
    os.makedirs(OUT_DIR, exist_ok=True)

    if 'env_hyper_params' in job_data.keys():
        job_data.env = register_env_variant(job_data.env, job_data.env_hyper_params)
    
    env_kwargs = job_data['env_kwargs']
    env = env_constructor(**env_kwargs)
    mean = np.zeros(env.action_dim)
    sigma = 1.0*np.ones(env.action_dim)
    filter_coefs = [sigma, job_data['filter']['beta_0'], job_data['filter']['beta_1'], job_data['filter']['beta_2']]

    for i in range(job_data['num_traj']):
        start_time = timer.time()
        print("Currently optimizing trajectory : %i" % i)
        seed = job_data['seed'] + i*12345
        env = env_constructor(**env_kwargs)
        env.reset(seed=seed) 

        agent = MPPI(env,
                    H=job_data['plan_horizon'],
                    paths_per_cpu=job_data['paths_per_cpu'],
                    num_cpu=job_data['num_cpu'],
                    kappa=job_data['kappa'],
                    gamma=job_data['gamma'],
                    mean=mean,
                    filter_coefs=filter_coefs,
                    default_act=job_data['default_act'],
                    seed=seed,
                    env_kwargs=env_kwargs)

        # Trajectory optimization
        for t in tqdm(range(job_data['H_total'])):
            agent.train_step(job_data['num_iter'])
        
        # Save trajectory
        SAVE_FILE = OUT_DIR + '/traj_%i.pickle' % i
        pickle.dump(agent, open(SAVE_FILE, 'wb'))
        
        end_time = timer.time()
        print("Trajectory reward = %f" % np.sum(agent.sol_reward))
        print("Optimization time for this trajectory = %f" % (end_time - start_time))

        # Save trajectory video
        for camera in ['default', 'left_cam', 'right_cam']:
            os.makedirs(f"{OUT_DIR}/{i}/{camera}", exist_ok=True)
            frames = agent.animate_result_offscreen(camera_name=camera)
            VID_FILE = OUT_DIR + f'/{i}/{i}_{job_data.embedding}_{camera}' + '.gif'
            cl = ImageSequenceClip(frames, fps=20)
            cl.write_gif(VID_FILE, fps=20)
            frames = np.array(frames)
            for t2 in range(frames.shape[0]):
                img = frames[t2]
                result = Image.fromarray((img).astype(np.uint8))
                result.save(f"{OUT_DIR}/{i}/{camera}/{t2}.png")

if __name__ == "__main__": 
    mp.set_start_method('spawn')
    configure_jobs()