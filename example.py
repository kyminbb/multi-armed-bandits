import testbed

bandits = 50
arms = 10
time_steps = 1000
mu = 1
sigma = 1
env = testbed.Testbed(bandits=bandits, arms=arms, time_steps=time_steps, q_true_mu=mu, q_true_sigma=sigma)

epsilons = [0.0, 0.001, 0.01, 0.1, 1.0]
env.register('epsilon-greedy', hyperparameters=epsilons)

initial_values = [0, 1, 2, 5, 10]
env.register('optimistic-initialization', hyperparameters=initial_values)

c_values = [0, 1, 2, 5]
env.register('ucb', hyperparameters=c_values)

temperatures = [1, 3, 10, 30, 100]
env.register('boltzmann', hyperparameters=temperatures)

env.test_all(plot=True)
