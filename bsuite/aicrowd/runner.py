import os
import bsuite
import numpy as np

from tqdm.notebook import tqdm
from bsuite.utils import gym_wrapper

class Runner:
    def __init__(self, env_id, agent, verbose=True, log_interval=100, eval=False):
        '''
        PARAMETERS:
        'env_id'       - Environment ID eg: environments.CARTPOLE
        'agent'        - Instance of an Agent class with the necessary methods implemented
        'verbose'      - True: prints logs, False: doesn't print logs
        'log_interval' - Interval between episodes to print logs at
        'eval'         - Use custom private results path as results dir
        '''
        self.agent = agent
        self.env_id = env_id
        self.verbose = verbose
        self.log_interval = log_interval

        if(eval):
            results_dir = os.environ.get('PRIVATE_RESULTS_DIR')
        else:
            results_dir = os.environ.get('RESULTS_DIR')

        env = bsuite.load_and_record_to_csv(env_id, results_dir=results_dir, overwrite=True)
        self.env = gym_wrapper.GymFromDMEnv(env)
    
    def play_episodes(self):
        episode_rewards = []
        episode_lengths = []
        mean_rewards = []

        for episode_n in tqdm(range(self.env.bsuite_num_episodes)):
            done = False
            total_reward = 0
            total_length = 0

            observation = self.env.reset()
            state = self.agent.get_state(observation)
            
            while not done:
                action = self.agent.get_action(state)

                next_observation, reward, done, _ = self.env.step(action)
                next_state = self.agent.get_state(next_observation)

                self.agent.learn(state, action, reward, next_state, done)

                state = next_state

                total_reward += reward
                total_length += 1

            episode_rewards.append(total_reward)
            episode_lengths.append(total_length)
            mean_reward = np.mean(episode_rewards[-100:])
            mean_rewards.append(mean_reward)
            
            if ( (((episode_n+1) % self.log_interval) == 0) and self.verbose):
                print("EPISODE: ",episode_n+1,"\tREWARD: ",total_reward,"\tMEAN_REWARD: ",round(mean_reward,2),"\tEPISODE_LENGTH: ",total_length)