from typing import List
import numpy as np
from . import Agent


class Boltzmann(Agent):
    def __init__(self, bandits: int, arms: int, time_steps: int, temperatures: List[float], q_true: np.numarray):
        '''Initialize a multi-armed bandit agent using UCB exploration.

        Args:
            bandits (int): The number of bandits (independent experiments). Since each experiment is noisy, take the 
                average to evaluate performance.
            arms (int): The number of arms of each bandit.
            time_steps (int): The number of steps we run each bandit.
            temperatures (list of (float)): Different temperature values to test the algorithm.
            q_true (numpy array): The true average rewards for each arm. 
        '''
        super().__init__(bandits, arms, time_steps, temperatures, q_true)
        self.hyperparameter_name = 'temperature'

    def learn(self) -> np.numarray:
        '''Perform reinforcement learning on the agent.

        Returns:
            rewards (numpy array): The array of the expected rewards over the time steps for each temperature value.
        '''
        return super()._learn(self._learn_temperature)

    def _learn_temperature(self, temperature: float) -> np.numarray:
        '''Perform Boltzmann exploration with a given temperature value.

        Args:
            temperature (float): The temperature value.

        Returns:
            average_expected_reward (numpy array): The array of the expected rewards over the time steps.
        '''
        average_expected_reward = np.zeros(self.time_steps + 1)
        for t in range(1, self.time_steps + 1):
            expected_rewards = np.zeros(self.bandits)
            for bandit in range(self.bandits):
                weighted_probabilities = self._softmax(bandit, temperature)
                action = np.random.choice(range(self.arms), p=weighted_probabilities)

                # Probability of each action is given by the softmax
                expected_rewards[bandit] = np.dot(self.q_true[bandit, :], weighted_probabilities)

                # Update the action-value table and the count table
                reward = np.random.normal(self.q_true[bandit, action], 1)
                self.N[bandit, action] += 1
                self.Q[bandit, action] += (reward - self.Q[bandit, action]) / self.N[bandit, action]
            average_expected_reward[t] = np.average(expected_rewards)
        return average_expected_reward

    def _softmax(self, bandit: int, temperature: float) -> np.numarray:
        '''Calculate the softmax of each arm of a given bandit.

        Args:
            bandit (int): The index of the current bandit.
            temperature (float): The temperature value.

        Returns:
            softmax (numpy array): The array of the softmax values.
        '''
        exponentiated = np.exp(self.Q[bandit, :] * temperature)
        return exponentiated / np.sum(exponentiated)

    def __repr__(self):
        return 'Boltzmann Exploration'
