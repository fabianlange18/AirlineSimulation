from stable_baselines3.sac import SAC
from airline_environment import AirlineEnvironment

env = AirlineEnvironment()

model = SAC('MlpPolicy', env)