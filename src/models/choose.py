from models.backwardinduction import BackwardInduction
from models.policyiteration import PolicyIteration
from models.valueiteration import ValueIteration
from models.approximatedp import ADP
from models.qlearning import QLearning

from stable_baselines3.dqn import DQN
from stable_baselines3.ddpg import DDPG
from stable_baselines3.a2c import A2C
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO
from stable_baselines3.td3 import TD3

def choose_model(model_name, env, estimator = None, debug = False, policy = 'MlpPolicy'):

    assert model_name in ['bi', 'vi', 'pi', 'adp', 'bi_est', 'vi_est', 'pi_est', 'adp_est', 'ql', 'dqn', 'ddpg', 'a2c', 'td3', 'sac', 'ppo'], "Model name must be one of bi, vi, pi, adp, ql, dqn, ddpg, a2c, sac or ppo"

    if model_name == 'bi':
        return BackwardInduction(env, debug)
    elif model_name == 'pi':
        return PolicyIteration(env, debug)
    elif model_name == 'vi':
        return ValueIteration(env, debug)
    elif model_name == 'adp':
        return ADP(env, debug)
    if model_name == 'bi_est':
        return BackwardInduction(env, estimator, debug)
    elif model_name == 'pi_est':
        return PolicyIteration(env, estimator, debug)
    elif model_name == 'vi_est':
        return ValueIteration(env, estimator, debug)
    elif model_name == 'adp_est':
        return ADP(env, estimator, debug)
    elif model_name == 'ql':
        return QLearning(env, debug)
    elif model_name == 'dqn':
        return DQN(policy, env)
    elif model_name == 'ddpg':
        return DDPG(policy, env)
    elif model_name == 'a2c':
        return A2C(policy, env)
    elif model_name == 'td3':
        return TD3(policy, env)
    elif model_name == 'sac':
        return SAC(policy, env)
    elif model_name == 'ppo':
        return PPO(policy, env)