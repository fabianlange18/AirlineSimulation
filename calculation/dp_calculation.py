import timeit
import signal
import numpy as np

from models.choose import choose_model
from simulation import simulation_run_monopoly

from util.colormap import plot_policy

timeout = 1000

def timeout_handler(signum, frame):
    raise TimeoutError(f"Function takes longer than {timeout} seconds.")


def calculate_perfect_policy(env, print_policy=True):
    #bi = choose_model('bi', env)
    pi = choose_model('pi', env)
    #vi = choose_model('vi', env)

    #bi_solved = calculation_time_track(bi, "Backward Induction")
    pi_solved = calculation_time_track(pi, "Policy Iteration")
    #vi_solved = calculation_time_track(vi, "Value Iteration")

    """if bi_solved and pi_solved:
        assert(np.all(bi.policy == pi.policy))
    if bi_solved and vi_solved:
        assert(np.all(bi.policy == vi.policy))
    if pi_solved and vi_solved:
        assert(np.all(pi.policy == vi.policy))"""

    if pi_solved:
        print("Optimal Policy calculated by Dynamic Programming:")
        perfect_policy = pi.policy
        # if bi_solved else pi.policy if pi_solved else vi.policy
        perfect_value = pi.value
        # if bi_solved else pi.value if pi_solved else vi.value
        plot_policy(perfect_policy, '0_DP', 0, 'Optimal')
        if print_policy:
            print("Perfect policy calculated by Dynamic Programming")
            print(perfect_policy)
        perfect_reward = simulation_run_monopoly(perfect_policy, '0_DP_Optimal', '0')

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
