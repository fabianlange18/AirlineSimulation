import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from scipy.stats import poisson
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class Estimator():

    def __init__(self, env, n, plot = False) -> None:
        self.n = n
        self.save_plot_dir = './plots/estimations/' if plot else None
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

        # self.env.reset()

        return trajectories



    def fit_customer_behavior(self, trajectories):

        # Features
        t = np.array(trajectories['s0'])
        x = np.array(trajectories['a'])
        t_square = np.power(t, 2)
        x_square = np.power(x, 2)
        t_root_1 = np.sqrt(t+1)
        x_root_1 = np.sqrt(x+1)
        t_log_1 = np.log(t+1)
        x_log_1 = np.log(x+1)
        t_x = t * x

        # Stack Features
        X = np.column_stack((t, x, t_square, x_square, t_root_1, x_root_1, t_log_1, x_log_1, t_x))
        Y = np.array(trajectories['i'])

        # Split Test/Train
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # Regression
        regression = Lasso()
        regression.fit(X_train, Y_train)

        Y_pred = regression.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)

        def estimated_function(x, t):
            coef = regression.coef_
            intercept = regression.intercept_
            return coef[0] * t + coef[1] * x + coef[2] * np.power(t, 2) + coef[3] * np.power(x, 2) + coef[4] * np.sqrt(t+1) + coef[5] * np.sqrt(x+1) + coef[6] * np.log(t+1) + coef[7] * np.log(x+1) + coef[8] * x * t + intercept

        if self.save_plot_dir:

            f = open(f'{self.save_plot_dir}/summary.txt', 'a')
            f.write(f"Regression for {self.n} data points:\n")
            f.write("Coefficients: t, x, t_square, x_square, t_root+1, x_root+1, t_log+1, x_log+1, t_x\n")
            [f.write(f'{round(coef, 4)}  ') for coef in regression.coef_]
            f.write(f"\nIntercept: {regression.intercept_}\n")
            f.write(f"MSE: {mse}\n\n")
            f.close()

            x_scale = np.arange(self.env.max_price + 1)
            t_scale = np.arange(self.env.booking_time + 1)

            fig = plt.figure(figsize=(12, 4))
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.scatter(x, t, Y, s=1)

            x_grid, y_grid = np.meshgrid(np.linspace(0, self.env.max_price, self.env.max_price), np.linspace(0, self.env.booking_time, self.env.booking_time))
            z_grid = estimated_function(x_grid, y_grid)

            # Plot the plane
            ax1.plot_surface(x_grid, y_grid, z_grid, alpha=0.5)
            
            # Plot lines
            ax1.plot(x_scale, [0] * len(x_scale), estimated_function(x=x_scale, t=0), color='black')
            ax1.plot([0] * len(t_scale), t_scale, estimated_function(x=0, t=t_scale), color='blue')
            ax1.plot(x_scale, [self.env.booking_time / 2] * len(x_scale), estimated_function(x=x_scale, t=self.env.booking_time / 2), color='black', linestyle=':')
            ax1.plot([self.env.max_price / 2] * len(t_scale), t_scale, estimated_function(x=self.env.max_price / 2, t=t_scale), color='blue', linestyle=':')
            ax1.plot(x_scale, [self.env.booking_time] * len(x_scale), estimated_function(x=x_scale, t=self.env.booking_time), color='black')
            ax1.plot([self.env.max_price] * len(t_scale), t_scale, estimated_function(x=self.env.max_price, t=t_scale), color='blue')

            ax1.set_xlabel('x')
            ax1.set_ylabel('t')
            ax1.set_zlabel('y')
            ax1.set_title("Fitted Probabilities")

            ax2 = fig.add_subplot(132)
            ax2.plot(x_scale, estimated_function(x=x_scale, t=0), color='black')
            ax2.plot(x_scale, estimated_function(x=x_scale, t=self.env.booking_time / 2), color='black', linestyle=':')
            ax2.plot(x_scale, estimated_function(x=x_scale, t=self.env.booking_time), color='black')
            ax2.set_title(f'Price Factor')
            ax2.set_ylim(0, self.env.customers_per_round)
            
            ax3 = fig.add_subplot(133)
            ax3.plot(t_scale, estimated_function(x=0, t=t_scale), color='blue')
            ax3.plot(t_scale, estimated_function(x=self.env.max_price / 2, t=t_scale), color='blue', linestyle=':')
            ax3.plot(t_scale, estimated_function(x=self.env.max_price, t=t_scale), color='blue')
            ax3.set_title(f'Time Factor')
            ax3.set_ylim(0, self.env.customers_per_round)

            plt.savefig(f"{self.save_plot_dir}/probabilities_{self.n}")
            plt.show()
            plt.close()

            fig = plt.figure(figsize=(12, 4))
            ax1 = fig.add_subplot(131, projection='3d')

            def optimal_function(x, t):
                result = self.env.calculate_p(x, t) * self.env.customers_per_round
                return np.full_like(x_scale, result) if np.isscalar(result) else result

            x_grid, y_grid = np.meshgrid(np.linspace(0, self.env.max_price, self.env.max_price), np.linspace(0, self.env.booking_time, self.env.booking_time))
            z_grid = optimal_function(x_grid, y_grid)

            ax1.plot_surface(x_grid, y_grid, z_grid, alpha=0.5)

            # Plot lines
            ax1.plot(x_scale, [0] * len(x_scale), optimal_function(x=x_scale, t=0), color='black')
            ax1.plot([0] * len(t_scale), t_scale, optimal_function(x=0, t=t_scale), color='blue')
            ax1.plot(x_scale, [self.env.booking_time / 2] * len(x_scale), optimal_function(x=x_scale, t=self.env.booking_time / 2), color='black', linestyle=":")
            ax1.plot([self.env.max_price / 2] * len(t_scale), t_scale, optimal_function(x=self.env.max_price / 2, t=t_scale), color='blue', linestyle=":")
            ax1.plot(x_scale, [self.env.booking_time] * len(x_scale), optimal_function(x=x_scale, t=self.env.booking_time), color='black')
            ax1.plot([self.env.max_price] * len(t_scale), t_scale, optimal_function(x=self.env.max_price, t=t_scale), color='blue')

            ax1.set_xlabel('x')
            ax1.set_ylabel('t')
            ax1.set_zlabel('y')
            ax1.set_title("Real Probabilities")

            ax2 = fig.add_subplot(132)
            ax2.plot(x_scale, optimal_function(x=x_scale, t=0), color='black')
            ax2.plot(x_scale, optimal_function(x=x_scale, t=self.env.booking_time / 2), color='black', linestyle=":")
            ax2.plot(x_scale, optimal_function(x=x_scale, t=self.env.booking_time), color='black')
            ax2.set_title(f'Price Factor')
            ax2.set_ylim(0, self.env.customers_per_round)
            
            ax3 = fig.add_subplot(133)
            ax3.plot(t_scale, optimal_function(x=0, t=t_scale), color='blue')
            ax3.plot(t_scale, optimal_function(x=self.env.max_price / 2, t=t_scale), color='blue', linestyle=':')
            ax3.plot(t_scale, optimal_function(x=self.env.max_price, t=t_scale), color='blue')
            ax3.set_title(f'Time Factor')
            ax3.set_ylim(0, self.env.customers_per_round)

            plt.savefig(f"{self.save_plot_dir}/real_probabilities")
            plt.show()
            plt.close()


            plt.figure()
            mu = estimated_function(x=5, t=5) if estimated_function(x=5, t=5) > 0 else 0
            plt.hist(poisson.rvs(mu=mu), range=(0, self.env.customers_per_round), density=True, edgecolor='black')
            plt.plot(np.arange(self.env.customers_per_round) + 0.5, [self.env.get_event_p(i, 5, [5]) for i in range(self.env.customers_per_round)])
            plt.plot(np.arange(self.env.customers_per_round) + 0.5, [poisson.pmf(i, mu=estimated_function(x=5, t=5)) for i in range(self.env.customers_per_round)])
            plt.title("Probabilites for price=5, t=5")
            plt.savefig(f"{self.save_plot_dir}/probabilities_{self.n}_middle")
            plt.close()

            plt.figure()
            mu = estimated_function(x=1, t=10) if estimated_function(x=1, t=10) > 0 else 0
            plt.hist(poisson.rvs(mu=mu), range=(0, self.env.customers_per_round), density=True, edgecolor='black')
            plt.plot(np.arange(self.env.customers_per_round) + 0.5, [self.env.get_event_p(i, 1, [10]) for i in range(self.env.customers_per_round)])
            plt.plot(np.arange(self.env.customers_per_round) + 0.5, [poisson.pmf(i, mu=estimated_function(x=1, t=10)) for i in range(self.env.customers_per_round)])
            plt.title("Probabilites for price=1, t=10")
            plt.savefig(f"{self.save_plot_dir}/probabilities_{self.n}_high")
            plt.close()

            plt.figure()
            mu = estimated_function(x=9, t=2) if estimated_function(x=9, t=2) > 0 else 0
            plt.hist(poisson.rvs(mu=mu, size=100000), range=(0, self.env.customers_per_round + 1), density=True, edgecolor='black')
            plt.plot(np.arange(self.env.customers_per_round + 1) + 0.5, [self.env.get_event_p(i, 9, [2]) for i in range(self.env.customers_per_round + 1)])
            plt.plot(np.arange(self.env.customers_per_round + 1) + 0.5, [poisson.pmf(i, mu=estimated_function(x=9, t=2)) for i in range(self.env.customers_per_round + 1)])
            plt.title("Probabilites for price=9, t=2")
            plt.savefig(f"{self.save_plot_dir}/probabilities_{self.n}_low")
            plt.close()

        return estimated_function, mse
    
    def estimate_function(self):
        trajectories = self.generate_samples()
        estimated_function, mse = self.fit_customer_behavior(trajectories)
        return estimated_function, mse