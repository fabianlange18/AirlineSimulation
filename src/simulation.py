import numpy as np
import matplotlib.pyplot as plt

from airline_environment import AirlineEnvironment
from duopoly_environment import DuopolyEnvironment

def simulation_run(policy, duopol, model_name = '', episodes = '', plot = True):

    if duopol:
        return duopol_simulation(policy, model_name, episodes, plot)
    else:
        return monopol_simulation(policy, model_name, episodes, plot)


def monopol_simulation(policy, model_name, episodes, plot):

    env = AirlineEnvironment()

    state = env.s

    price_trajectory = []
    state_trajectory = []
    reward_trajectory = []

    for _ in range(env.booking_time):
        if isinstance(policy, np.ndarray):
            action = policy[*state]
        else:
            action = policy.predict(state, deterministic=True)[0]
        state, reward, done, _ = env.step(action)
        price_trajectory.append(action)
        state_trajectory.append(state)
        reward_trajectory.append(reward)

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

        state_trajectory = [list(i) for i in zip(*state_trajectory)]

        ax1.plot(price_trajectory)
        ax1.set_title("Prices")
        ax2.plot(state_trajectory[0])
        ax2.plot(state_trajectory[1])
        ax2.set_title("State")
        ax3.plot(reward_trajectory)
        ax3.set_title(f"Profits (total: {np.sum(reward_trajectory)})")

        plt.subplots_adjust(hspace=0.35)

        plt.savefig(f'./plots/simulations/{model_name}_{episodes}')
        plt.close()
    
    return np.sum(reward_trajectory)

def duopol_simulation(policy, model_name, episodes, plot):
    env = DuopolyEnvironment(continuous_action_space=False, stochastic_customers=False)

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
        if isinstance(policy, np.ndarray):
            action = policy[*state]
        else:
            action = policy.predict(state, deterministic=True)[0]
        state, reward, done, info = env.step(int(action))

        own_price_trajectory.append(action)
        comp_price_trajectory.append(state[3])

        own_state_trajectory.append(state[1])
        comp_state_trajectory.append(state[2])

        own_reward_trajectory.append(reward)
        comp_reward_trajectory.append(info['comp_rew'])

        events.append(info['i'])

    if plot:
        
        events = [list(i) for i in zip(*events)]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)

        ax1.plot(own_price_trajectory)
        ax1.plot(comp_price_trajectory[:-1])
        ax1.set_title("Prices")
        ax2.plot(own_state_trajectory)
        ax2.plot(comp_state_trajectory)
        ax2.set_title("State")
        ax3.plot(own_reward_trajectory)
        ax3.plot(comp_reward_trajectory)
        ax3.set_title(f"Profits (own: {np.sum(own_reward_trajectory)}, comp: {np.sum(comp_reward_trajectory)}))")
        ax4.plot(events[0])
        ax4.plot(events[1])
        ax4.plot(events[2])
        ax4.set_title("Events")

        plt.subplots_adjust(hspace=0.35)

        plt.savefig(f'./plots/simulations/{model_name}_{episodes}')
        plt.close()

    return np.sum(own_reward_trajectory)