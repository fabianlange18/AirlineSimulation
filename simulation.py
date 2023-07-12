import numpy as np
import matplotlib.pyplot as plt

from airline_environment import AirlineEnvironment, AirlineDuopoly


def simulation_run_monopoly(policy, model_name, episodes):
    env = AirlineDuopoly()

    env.reset()
    state = env.s

    price_trajectory = []
    state_trajectory = []
    reward_agent_trajectory = []

    for _ in range(env.booking_time):
        if isinstance(policy, np.ndarray):
            action = policy[*state]
        else:
            action = policy.predict(state, deterministic=True)[0]
        state, reward, done, _ = env.step(action)
        price_trajectory.append(action)
        state_trajectory.append(state)
        reward_agent_trajectory.append(reward)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    state_trajectory = [list(i) for i in zip(*state_trajectory)]

    ax1.plot(price_trajectory)
    ax1.set_title("Prices")
    ax2.plot(state_trajectory[0])
    ax2.plot(state_trajectory[1])
    ax2.set_title("State")
    ax3.plot(reward_agent_trajectory)
    ax3.set_title(f"Profits Agent (total: {np.sum(reward_agent_trajectory)})")

    plt.subplots_adjust(hspace=0.35)

    plt.savefig(f'./plots/simulations/{model_name}_{episodes}')
    plt.close()

    cum_rewards = np.sum(reward_agent_trajectory)

    return cum_rewards

def simulation_run_duopoly(policy, model_name, episodes):

    env = AirlineDuopoly()

    env.reset()
    state = env.s

    price_trajectory = []
    state_trajectory = []
    reward_agent_trajectory = []
    reward_comp_trajectory = []

    for _ in range(env.booking_time):
        if isinstance(policy, np.ndarray):
            action = policy[*state]
        else:
            action = policy.predict(state, deterministic=True)[0]
        state, reward, done, output_dict = env.step(action)
        price_trajectory.append(action)
        state_trajectory.append(state)
        reward_agent_trajectory.append(reward)
        reward_comp_trajectory.append(output_dict["comp_1/reward_per_step"])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)

    state_trajectory = [list(i) for i in zip(*state_trajectory)]

    ax1.plot(price_trajectory)
    ax1.set_title("Prices")
    ax2.plot(state_trajectory[0])
    ax2.plot(state_trajectory[1])
    ax2.set_title("State")
    ax3.plot(reward_agent_trajectory)
    ax3.set_title(f"Profits Agent (total: {np.sum(reward_agent_trajectory)})")
    ax4.plot(reward_comp_trajectory)
    ax4.set_title(f"Profits Competitor (total: {np.sum(reward_comp_trajectory)})")


    plt.subplots_adjust(hspace=0.35)

    plt.savefig(f'./plots/simulations/{model_name}_{episodes}')
    plt.close()

    cum_rewards = []
    cum_rewards.append(np.sum(reward_agent_trajectory))
    cum_rewards.append(np.sum(reward_comp_trajectory))
    
    return cum_rewards