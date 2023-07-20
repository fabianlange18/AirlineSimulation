from duopoly_environment import DuopolyEnvironment
from models.policyiteration import PolicyIteration
from models.valueiteration import ValueIteration
from models.approximatedp import ADP
from models.backwardinduction import BackwardInduction

from stable_baselines3.ppo.ppo import PPO

env = DuopolyEnvironment(continuous_action_space=False)

bi = BackwardInduction(env, debug=True)

bi.solve()

print(bi.policy)

# model = PPO('MlpPolicy', env)

# model.learn(1000000, progress_bar=True)


import numpy as np
import matplotlib.pyplot as plt

def simulation_run(model):

    env = DuopolyEnvironment(continuous_action_space=False)

    state = env.s

    own_price_trajectory = []
    comp_price_trajectory = []
    comp_price_trajectory.append(state[3])
    own_state_trajectory = []
    comp_state_trajectory = []
    own_reward_trajectory = []
    comp_reward_trajectory = []
    events = []

    for _ in range(env.booking_time):
        action = model.predict(state, deterministic=True)[0]
        state, reward, done, info = env.step(int(action))

        own_price_trajectory.append(action)
        comp_price_trajectory.append(state[3])

        own_state_trajectory.append(state[1])
        comp_state_trajectory.append(state[2])

        own_reward_trajectory.append(reward)
        comp_reward_trajectory.append(info['comp_rew'])

        events.append(info['i'])


    events = [list(i) for i in zip(*events)]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)

    ax1.plot(own_price_trajectory)
    ax1.plot(comp_price_trajectory)
    ax1.set_title("Prices")
    ax2.plot(own_state_trajectory)
    ax2.plot(comp_state_trajectory)
    ax2.set_title("State")
    ax3.plot(own_reward_trajectory)
    ax3.plot(comp_reward_trajectory)
    ax3.set_title(f"Profits (own total: {np.sum(own_reward_trajectory)})")
    ax4.plot(events[0])
    ax4.plot(events[1])
    ax4.plot(events[2])
    ax4.set_title("Events")

    plt.subplots_adjust(hspace=0.35)

    plt.show()


# simulation_run(model)