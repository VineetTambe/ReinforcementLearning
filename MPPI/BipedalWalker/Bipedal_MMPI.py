import gym
import numpy as np
from copy import deepcopy


class BipedalMPPI():
    """This is an implementation of the MPPI Controller (as per the paper)
    extending the controller to a Bipedal walker
    terminology in this code is from: https://www.youtube.com/watch?v=19QLyMuQ_BE&ab_channel=NeuromorphicWorkshopTelluride
    """

    def __init__(self, env, n, K, T, U, uinit=[0.25, 0.25, 0.25, 0.25], lambda_=1, noise_mu=0, noise_sigma=1):
        # do some initialization here
        self.n = n
        self.K = K
        self.T = T
        self.U = U
        self.uinit = uinit
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma,
                                      size=(self.K, self.T, env.action_space.shape[0]))
        self.lambda_ = lambda_
        self.env = env
        self.env.reset()
        self.S = np.zeros(shape=self.K)
        self.beta = 0
        self.eta = 0
        self.omega = 0
        # self.x_init = self.env.env.state

        self.last_reward = 0

    def simulate_rollouts(self, k, copy_env):
        # self.env.env.state = self.x_init
        for t in range(0, self.T):
            rollout_action_t = self.U[t, :] + self.noise[k, t, :]
            _, reward, _, _ = copy_env.step(rollout_action_t)
            # copy_env.render()
            self.S[k] = self.S[k] - reward
        # copy_env.close()

    def calcBeta(self):
        self.beta = np.min(self.S)

    def calcEta(self):
        self.eta = np.sum(np.exp((-1.0 / self.lambda_) * (self.S - self.beta)))

    def calcOmega(self):
        self.omega = (1.0 / self.eta) * (np.exp((-1.0 / self.lambda_) * (self.S - self.beta)))

    def control(self):
        for _ in range(0, self.n):
            # simulation loop
            for k in range(0, self.K):
                temp = deepcopy(self.env)
                self.simulate_rollouts(k, temp)

            # from the above sampled random trajectories of control calculate the best traj
            self.calcBeta()
            self.calcEta()
            self.calcOmega()
            for actuator in range(env.action_space.shape[0]):
                self.U[:, actuator] += [np.sum(self.omega * self.noise[:, t, actuator]) for t in range(self.T)]
            # self.U[:, 1] += [np.sum(self.omega * self.noise[:, t, 1]) for t in range(self.T)]
            # self.U[:, 2] += [np.sum(self.omega * self.noise[:, t, 2]) for t in range(self.T)]
            # self.U[:, 3] += [np.sum(self.omega * self.noise[:, t, 3]) for t in range(self.T)]

            # self.env.env.state = self.x_init
            _, r, _, _ = self.env.step(self.U[0, :])
            # self.cumulative_reward += r
            print("action taken: " + str(self.U[0, :]) + " reward received: " + str(r))
            self.env.render()

            # if self.cumulative_reward < -400:
            #     env.reset()
            #     self.cumulative_reward = 0
            #     self.S[:] = 0
            #     continue
            self.U = np.roll(self.U, -4)  # shift all elements to the left
            self.U[-1, :] = self.uinit  #
            self.S[:] = 0

            if abs(self.last_reward - r) < 0.000015:
                self.last_reward = r
                self.env.reset()
            elif r == -100:
                self.env.reset()

            # self.x_init = self.env.env.state


if __name__ == "__main__":
    ENV_NAME = "BipedalWalker-v3"
    # TIMESTEPS = 20  # T
    # N_SAMPLES = 1000  # K

    TIMESTEPS = 2  # T
    N_SAMPLES = 50  # K

    ACTION_LOW = -1.0
    ACTION_HIGH = 1.0

    noise_mu = 0
    noise_sigma = 1
    lambda_ = 1
    iter = 1000

    env = gym.make(ENV_NAME)
    # env = gym.make("BipedalWalker-v3")
    U = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=(TIMESTEPS, env.action_space.shape[0]))

    bipedal_mppi_gym = BipedalMPPI(env=env, n=iter, K=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mu=noise_mu,
                                   noise_sigma=noise_sigma, uinit=0)
    bipedal_mppi_gym.control()
