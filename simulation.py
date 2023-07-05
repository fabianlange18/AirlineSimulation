import numpy as np
import matplotlib.pyplot as plt

from airline_environment import AirlineEnvironment

def simulation_run(policy, model_name, episodes, plot = True):

    env = AirlineEnvironment()

    env.reset()
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