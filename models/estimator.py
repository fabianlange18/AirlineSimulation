import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from scipy.stats import poisson
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class Estimator():

    def __init__(self, env, n, plot = False) -> None:
        self.n = n
        self.plot = plot
        self.env = env
        self.estimate_function, self.mse = self.estimate_function()


    def generate_samples(self):

        trajectories = defaultdict(list)
        s = self.env.initial_state

        for _ in range(self.n):
            
            a = self.env.random_action()
            
            trajectories['s0'].append(s[0])
            trajectories['a'].append(a)
            
            s, r, done, info = self.env.step(a)

            trajectories['i'].append(info['i'])
            trajectories['r'].append(r)

            if done:
                self.env.reset()

        return trajectories



    def fit_customer_behavior(self, trajectories):

        # Features
        t = np.array(trajectories['s0'])
        x = np.array(trajectories['a'])
        t_square = np.power(t, 2)
        x_square = np.power(x, 2)
        t_root = np.sqrt(t)
        x_root = np.sqrt(x)
        t_x = t * x

        # Stack Features
        X = np.column_stack((t, x, t_square, x_square, t_root, x_root, t_x))
        Y = np.array(trajectories['i'])

        # Split Test/Train
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # Regression
        regression = LinearRegression()
        regression.fit(X_train, Y_train)

        Y_pred = regression.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)

        def estimated_function(x, t):
            coef = regression.coef_
            intercept = regression.intercept_
            return coef[0] * t + coef[1] * x + coef[2] * np.power(t, 2) + coef[3] * np.power(x, 2) + coef[4] * np.sqrt(t) + coef[5] * np.sqrt(x) + coef[6] * x * t + intercept

        if self.plot:
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
            z_grid = estimated_function(x_grid, y_grid)

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

            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5)

            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_zlabel('y')
            plt.title("Real Probabilities")
            plt.show()


            plt.figure()
            plt.hist(poisson.rvs(mu=estimated_function(x=5, t=5), size=100000), range=(0, 10), density=True, edgecolor='black')
            plt.plot(np.arange(10) + 0.5, [self.env.get_event_p(i, 5, [5]) for i in range(10)])
            plt.plot(np.arange(10) + 0.5, [poisson.pmf(i, mu=estimated_function(x=5, t=5)) for i in range(10)])
            plt.title("Probabilites for price=5, t=5")
            plt.show()

            plt.figure()
            plt.hist(poisson.rvs(mu=estimated_function(x=1, t=10), size=100000), range=(0, 10), density=True, edgecolor='black')
            plt.plot(np.arange(10) + 0.5, [self.env.get_event_p(i, 1, [10]) for i in range(10)])
            plt.plot(np.arange(10) + 0.5, [poisson.pmf(i, mu=estimated_function(x=1, t=10)) for i in range(10)])
            plt.title("Probabilites for price=1, t=10")
            plt.show()

            plt.figure()
            plt.hist(poisson.rvs(mu=estimated_function(x=9, t=2), size=100000), range=(0, 10), density=True, edgecolor='black')
            plt.plot(np.arange(10) + 0.5, [self.env.get_event_p(i, 9, [2]) for i in range(10)])
            plt.plot(np.arange(10) + 0.5, [poisson.pmf(i, mu=estimated_function(x=9, t=2)) for i in range(10)])
            plt.title("Probabilites for price=9, t=2")
            plt.show()

        return estimated_function, mse
    
    def estimate_function(self):
        trajectories = self.generate_samples()
        estimated_function, mse = self.fit_customer_behavior(trajectories)
        return estimated_function, mse