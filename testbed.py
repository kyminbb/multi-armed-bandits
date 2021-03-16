from typing import List
import numpy as np
import matplotlib.pyplot as plt
from agents import epsilon_greedy, optimistic_initialization, ucb


class Testbed:
    def __init__(self,
                 bandits: int = 20,
                 arms: int = 10,
                 time_steps: int = 1000,
                 q_true_mu: float = 1.0,
                 q_true_sigma: float = 1.0):
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

    def register(self, algorithm: str, hyperparameters: List[float]):
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
            pass

    def test_all(self):
        '''Experiment with all the agents registered in the testbed.
        '''
        for algorithm in self.agents:
            self.test(algorithm)

    def test(self, algorithm: str):
        '''Experiment with the agent that uses a given exploration algorithm.

        Args:
            algorithm (str): The name of an exploration algorithm.
        '''
        try:
            agent = self.agents[algorithm]
        except:
            print(f'There is no {algorithm} agent.')
            return
        rewards = agent.learn()
        self.rewards[algorithm] = rewards
