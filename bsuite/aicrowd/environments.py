import bsuite
from bsuite.utils import gym_wrapper

CATCH = 'catch/0'
CATCH_NOISE = 'catch_noise/1'
CARTPOLE = 'cartpole/0'
CARTPOLE_NOISE = 'cartpole_noise/1'
MOUNTAINCAR = 'mountain_car/0'
MOUNTAINCAR_NOISE = 'mountain_car_noise/1'

def load_env(env_id):
    env = bsuite.load_from_id(env_id)
    env = gym_wrapper.GymFromDMEnv(env)
    return env