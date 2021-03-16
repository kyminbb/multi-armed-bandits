from typing import List
import numpy as np
import matplotlib.pyplot as plt
from agents import epsilon_greedy, optimistic_initialization, ucb, boltzmann


class Testbed:
    def __init__(self, bandits: int, arms: int, time_steps: int, q_true_mu: float, q_true_sigma: float) -> None:
        '''Initialize a testbed.

        Args: 
            bandits (int): The number of bandits (independent experiments) for each agent. Since each experiment is 
                noisy, take the average to evaluate performance.
            arms (int): The number of arms of each bandit.
            time_steps (int): The number of steps we run each bandit.
            q_true_mu (float): The mean of the true average rewards for each arm.
            q_true_sigma (float): The standard deviation of the true average rewards for each arm.
        '''
        self.bandits = bandits
        self.arms = arms
        self.time_steps = time_steps
        self.q_true = np.random.normal(q_true_mu, q_true_sigma**2, (self.bandits, self.arms))
        self.agents = dict()
        self.rewards = dict()

    def register(self, algorithm: str, hyperparameters: List[float]) -> None:
        '''Register an agent that uses a given exploration algorithm.

        Args:
            algorithm (str): The name of an exploration algorithm.
            hyperparameters (list of (float)): Different hyperparameters to test the algorithm.
        '''
        if algorithm == 'epsilon-greedy':
            self.agents[algorithm] = epsilon_greedy.EpsilonGreedy(
                self.bandits, self.arms, self.time_steps, hyperparameters, self.q_true)
        elif algorithm == 'optimistic-initialization':
            self.agents[algorithm] = optimistic_initialization.OptimisticInitialization(
                self.bandits, self.arms, self.time_steps, hyperparameters, self.q_true)
        elif algorithm == 'ucb':
            self.agents[algorithm] = ucb.UCB(self.bandits, self.arms, self.time_steps, hyperparameters, self.q_true)
        elif algorithm == 'boltzmann':
            self.agents[algorithm] = boltzmann.Boltzmann(
                self.bandits, self.arms, self.time_steps, hyperparameters, self.q_true)

    def test_all(self, plot: bool = False) -> None:
        '''Experiment with all the agents registered in the testbed.

        Args:
            plot (bool): The flag  to indicate whether to plot the results.
        '''
        for algorithm in self.agents:
            self.test(algorithm, plot)
        if plot:
            self._plot_best()

    def test(self, algorithm: str, plot: bool = False) -> None:
        '''Experiment with the agent that uses a given exploration algorithm.

        Args:
            algorithm (str): The name of an exploration algorithm.
            plot (bool): The flag to indicate whether to plot the results.
        '''
        try:
            agent = self.agents[algorithm]
        except:
            print(f'There is no {algorithm} agent.')
            return
        rewards = agent.learn()
        self.rewards[algorithm] = rewards
        if plot:
            self._plot(algorithm)

    def _plot(self, algorithm: str) -> None:
        '''Plot the expected rewards of the agent that uses a given exploration algorithm over the time steps.

        Args:
            algorithm (str): The name of an exploration algorithm.  
        '''
        colors = ['tab:green', 'tab:olive', 'tab:red', 'tab:blue', 'tab:purple']
        figure = plt.figure().add_subplot(111)
        agent = self.agents[algorithm]
        for i in range(len(agent.hyperparameters)):
            # Plot the rewards for each value of hyperparameter
            figure.plot(range(0, self.time_steps + 1), self.rewards[algorithm][i, :], colors[i])

        figure.title.set_text(f'Average Reward of {agent} in {self.arms}-Armed Bandit')
        figure.set_xlabel('Time Steps')
        figure.set_ylabel('Average Reward')
        legend = tuple([f'{agent.hyperparameter_name}={hyperparameter}' for hyperparameter in agent.hyperparameters])
        figure.legend(legend, loc='best')
        plt.show()

    def _plot_best(self) -> None:
        '''Plot the best-performing hyperparameter setting for each agent over the time steps.
        '''
        colors = ['tab:green', 'tab:olive', 'tab:red', 'tab:blue', 'tab:purple']
        figure = plt.figure().add_subplot(111)
        legend = []
        i = 0
        for algorithm, rewards in self.rewards.items():
            agent = self.agents[algorithm]
            # Plot the rewards corresponding to the best hyperparameter
            best_hyperparameter = np.argmax(rewards[:, self.time_steps - 1])
            figure.plot(range(0, self.time_steps + 1), rewards[best_hyperparameter, :], colors[i])
            legend.append(f'{agent} ({agent.hyperparameter_name}={agent.hyperparameters[best_hyperparameter]})')
            i += 1

        figure.title.set_text('Average Reward of Best Hyperparameter Settings')
        figure.set_xlabel('Time Steps')
        figure.set_ylabel('Average Reward')
        figure.legend(tuple(legend), loc='best')
        plt.show()
