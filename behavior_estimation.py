from collections import defaultdict

import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from models.solver import Solver
from models.backwardinduction import BackwardInduction

from util.colormap import plot_colormap
from simulation import simulation_run

from airline_environment import AirlineEnvironment


class BehaviorEstimator(Solver):

    def generate_samples(self, n = 50):

        trajectories = defaultdict(list)
        s = self.env.initial_state

        for _ in range(n):
            
            a = self.env.random_action()
            
            trajectories['s0'].append(s[0])
            trajectories['a'].append(a)
            
            s, r, done, info = self.env.step(a)

            trajectories['i'].append(info['i'])
            trajectories['r'].append(r)

            if done:
                self.env.reset()

        return trajectories


    def fit_customer_behavior(self, trajectories, plot = False):

        # features
        t = np.array(trajectories['s0'])
        x = np.array(trajectories['a'])
        t_square = np.power(t, 2)
        x_square = np.power(x, 2)
        t_root = np.sqrt(t)
        x_root = np.sqrt(x)
        t_x = t * x


        X = np.column_stack((t, x, t_square, x_square, t_root, x_root, t_x))
        Y = np.array(trajectories['i'])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


        regression = LinearRegression()

        regression.fit(X_train, Y_train)

        Y_pred = regression.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)

        def plane_function(x, t):
            coef = regression.coef_
            intercept = regression.intercept_
            return coef[0] * t + coef[1] * x + coef[2] * np.power(t, 2) + coef[3] * np.power(x, 2) + coef[4] * np.sqrt(t) + coef[5] * np.sqrt(x) + coef[6] * x * t + intercept
        
        print(f"n={len(trajectories['s0'])} -> MSE: {mse}")


        if plot:
            print("Coefficients: t, x, t_square, x_square, t_root, x_root, t_x")
            print(regression.coef_)
            print("Intercept:")
            print(regression.intercept_)
            print("MSE:")
            print(mse)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, t, Y, s=1)



            x_grid, y_grid = np.meshgrid(np.linspace(0, 20, 20), np.linspace(0, 20, 20))
            z_grid = plane_function(x_grid, y_grid)

            # Plot the plane
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5)


            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_zlabel('y')

            plt.title("Fitted Probabilities")

            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            def optimal_function(x, t):
                return self.env.calculate_p(x, t)

            x_grid, y_grid = np.meshgrid(np.linspace(0, 20, 20), np.linspace(0, 20, 20))
            z_grid = optimal_function(x_grid, y_grid)

            # Plot the plane
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5)

            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_zlabel('y')
            plt.title("Real Probabilities")
            plt.show()


            plt.figure()
            plt.hist(poisson.rvs(mu=plane_function(x=5, t=5), size=100000), range=(0, 10), density=True, edgecolor='black')
            plt.plot(np.arange(10) + 0.5, [self.env.get_event_p(i, 5, [5]) for i in range(10)])
            plt.plot(np.arange(10) + 0.5, [poisson.pmf(i, mu=plane_function(x=5, t=5)) for i in range(10)])
            plt.title("Probabilites for price=5, t=5")
            plt.show()

            plt.figure()
            plt.hist(poisson.rvs(mu=plane_function(x=1, t=10), size=100000), range=(0, 10), density=True, edgecolor='black')
            plt.plot(np.arange(10) + 0.5, [self.env.get_event_p(i, 1, [10]) for i in range(10)])
            plt.plot(np.arange(10) + 0.5, [poisson.pmf(i, mu=plane_function(x=1, t=10)) for i in range(10)])
            plt.title("Probabilites for price=1, t=10")
            plt.show()

            plt.figure()
            plt.hist(poisson.rvs(mu=plane_function(x=9, t=2), size=100000), range=(0, 10), density=True, edgecolor='black')
            plt.plot(np.arange(10) + 0.5, [self.env.get_event_p(i, 9, [2]) for i in range(10)])
            plt.plot(np.arange(10) + 0.5, [poisson.pmf(i, mu=plane_function(x=9, t=2)) for i in range(10)])
            plt.title("Probabilites for price=9, t=2")
            plt.show()

        return plane_function
    


    # Backward Induction with the estimated behavior

    def comp_expected_reward_estimated(self, plane_function, t, future):
        r = np.zeros((self.env.flight_capacity + 1,))
        a = np.zeros((self.env.flight_capacity + 1,))
        for s in range(self.env.flight_capacity + 1):
            s = np.array([t, s])
            a_max = max((
                (a, sum(
                    poisson.pmf(i, mu=plane_function(x=a, t=s[0])) * (self.env.get_reward(i, a, s) + future[self.env.transit_state(i, a, s)[1]])
                    for i in range(self.env.customers_per_round + 1)
                ))
                for a in range(self.env.action_space_max + 1)
            ), key=lambda o: o[1])
            r[s[1]] = a_max[1]
            a[s[1]] = a_max[0]
        return r, a

    def solve_estimated(self, n):
        self.reset()
        trajectories = self.generate_samples(n)
        plane_function = self.fit_customer_behavior(trajectories)
        for t in range(self.env.booking_time - 1, -1, -1):
            self.value[t], self.policy[t] = self.comp_expected_reward_estimated(plane_function, t, self.value[t+1] if t + 1 < self.env.booking_time else np.zeros((self.env.flight_capacity + 1,)))


env = AirlineEnvironment(False)

estimator = BehaviorEstimator(env)
bi = BackwardInduction(env)

n = 100

# plot for demonstration purposes
trajectories = estimator.generate_samples(n)
estimator.fit_customer_behavior(trajectories, plot=True)


reward_trajectory = []
n = np.arange(10, 500, step=10)

for n_i in n:

    estimator.solve_estimated(n=n_i)
    bi.solve()

    # plot_colormap(estimator.policy, bi.policy, model_name= "0_Estimation", episodes = n_i, title="Policy")
    reward = simulation_run(estimator.policy, model_name="0_Estimation", episodes=n_i, plot = False)
    reward_trajectory.append(reward)

plt.figure()
plt.plot(n, reward_trajectory)
plt.show()