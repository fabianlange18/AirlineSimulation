from stable_baselines3.ppo import PPO
from airline_environment import AirlineEnvironment

env = AirlineEnvironment()

model = PPO('MlpPolicy', env, gamma=0.99999)

model.learn(1000000, progress_bar=True)

print(model.predict([0], deterministic=True))
print(model.predict([1], deterministic=True))
print(model.predict([2], deterministic=True))
print(model.predict([3], deterministic=True))
print(model.predict([4], deterministic=True))
print(model.predict([5], deterministic=True))