from stable_baselines3.ppo import PPO
from airline_environment import AirlineEnvironment

import matplotlib.pyplot as plt

env = AirlineEnvironment()

model = PPO('MlpPolicy', env, gamma=0.99999)

n_steps = 10000000

model.learn(n_steps, progress_bar=True)

env.reset()
state = env.state

price_trajectory = []
state_trajectory = []
reward_trajectory = []

for i in range(env.booking_time):
    action = model.predict(state, deterministic=True)[0]
    state, reward, done, _ = env.step(action)
    price_trajectory.append(action[0])
    state_trajectory.append(state.copy())
    reward_trajectory.append(reward)

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

state_trajectory = [list(i) for i in zip(*state_trajectory)]

ax1.plot(price_trajectory)
ax1.set_title("Prices")
ax2.plot(state_trajectory[0])
ax2.plot(state_trajectory[1])
ax2.set_title("State")
ax3.plot(reward_trajectory)
ax3.set_title("Profits")

plt.show()