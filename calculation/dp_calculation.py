import timeit
import signal
import numpy as np

from models.choose import choose_model
from simulation import simulation_run

from util.colormap import plot_policy

timeout = 3600

def timeout_handler(signum, frame):
    raise TimeoutError(f"Function takes longer than {timeout} seconds.")


def calculate_perfect_policy(env, estimator = None, print_policy = False, just_result=False, duopol=False):

    if just_result:
        bi = choose_model('bi_est', env, estimator) if estimator else choose_model('bi', env)
        bi_solved = calculation_time_track(bi, "Backward Induction")
        perfect_reward = simulation_run(bi.policy, duopol, plot=False)
        return bi.policy, bi.value, perfect_reward

    if estimator:
        bi = choose_model('bi_est', env, estimator)
        pi = choose_model('pi_est', env, estimator)
        vi = choose_model('vi_est', env, estimator)
    else:
        bi = choose_model('bi', env)
        pi = choose_model('pi', env)
        vi = choose_model('vi', env)


    bi_solved = calculation_time_track(bi, "Backward Induction")
    pi_solved = calculation_time_track(pi, "Policy Iteration")
    vi_solved = calculation_time_track(vi, "Value Iteration")

    if bi_solved and pi_solved:
        assert(np.all(bi.policy == pi.policy))
    if bi_solved and vi_solved:
        assert(np.all(bi.policy == vi.policy))
    if pi_solved and vi_solved:
        assert(np.all(pi.policy == vi.policy))

    if bi_solved or pi_solved or vi_solved:
        perfect_policy = bi.policy if bi_solved else pi.policy if pi_solved else vi.policy
        perfect_value = bi.value if bi_solved else pi.value if pi_solved else vi.value
        if not duopol:
            plot_policy(perfect_policy, '0_DP', 0, 'Optimal')
        if print_policy:
            print("Perfect policy calculated by Dynamic Programming")
            print(perfect_policy)
        if env.stochastic_customers:
            perfect_rewards = []
            for _ in range(100):
                perfect_rewards.append(simulation_run(perfect_policy, duopol, '0_DP_Optimal', '0'))
            perfect_reward = np.mean(perfect_rewards)
        else:
            perfect_reward = simulation_run(perfect_policy, duopol, '0_DP_Optimal', '0')

        return perfect_policy, perfect_value, perfect_reward
    
    else:
        AssertionError("No Dynamic Programming Method solved in time.")


def calculation_time_track(model, name):
    signal.signal(signal.SIGALRM, timeout_handler)

    try:
        signal.alarm(timeout)
        model_time = timeit.timeit(model.solve, number=1)
        signal.alarm(0)
        print(f'{name} takes {round(model_time, 3)} seconds.')
        return True
    except TimeoutError:
        print(f'{name} takes longer than {timeout} seconds.')
        return False
