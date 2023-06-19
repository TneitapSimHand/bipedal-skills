
import argparse
import numpy as np
import gym
import bisk
import cv2 

assert gym.__version__ == '0.26.1' or gym.__version__ == '0.21.0', "Unmatched Gym version"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='walker') # walker | humanoid | halfcheetah
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    env_names_all = {
    'hurdles': 'BiskHurdles-v1',
    'limbo': 'BiskLimbo-v1',
    'hurdleslimbo': 'BiskHurdlesLimbo-v1',
    'gaps': 'BiskGaps-v1',
    'stairs': 'BiskStairs-v1',
    'goalwall': 'BiskGoalWall-v1',
    'polebalance': 'BiskPoleBalance-v1',
    'gototarget': 'BiskGoToTarget-v1',
    'butterflies': 'BiskButterflies-v1',
    }
    for env_key, env_fullname in env_names_all.items():
        print("*********%s*******"%(env_key))
        env = gym.make(env_fullname, robot=args.robot) # not support render_mode = "human"
        print(f'timestep {env.p.model.opt.timestep}s x frameskip {env.frameskip} = dt {env.dt}s')

        # print("reorder observations: ", env.env.env.observation_space.keys())
        print("gym wrapped obs dim: ", env.observation_space.shape)
        print("gym wrapped act dim: ", env.action_space.shape)
        test_episode = 1
        for epi_i in range(test_episode):
            print("episode %02d"%(epi_i))
            obs, infos = env.reset()
            for _ in range(1000): 
                action = np.random.randn(*env.action_space.shape) # random policy

                if gym.__version__ == '0.26.1': 
                    obs, reward, done, trunc, infos = env.step(action) # gym26: 4 outs
                else: 
                    obs, reward, done, infos = env.step(action)

                img_Arr = env.render(width=640, height=480, camera=0)
                # cv2.imshow("%s-%s"%(args.robot, env_key), img_Arr[:, :, ::-1])
                # cv2.waitKey(1)