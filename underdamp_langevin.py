import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from model.flags import get_flags
from model.event import plot_figs
from model.langevin import over_damped_exact, over_damped_stochastic, under_damped_exact, under_damped_stochastic


def run_func(args):
    # Hyperparameters
    dim = args.dim
    gamma = args.gamma
    batch = args.batch_size
    n_arms = args.n_arm
    n_rounds = args.n_round
    step_size = args.step_size
    n_iterations = args.n_iter
    prior_mean = np.zeros(dim)
    prior_variance = np.eye(dim)
    arm_covariances = [np.eye(dim) for _ in range(n_arms)]  # Known covariance for all arms

    # Simulated true means (2-dimensional)
    true_means = np.random.randn(n_arms, dim) * 5
    idx = np.argsort(-np.linalg.norm(true_means, axis=1))
    true_means = true_means[idx].copy()

    # print(f"True means:\n {true_means}")
    # print(f"Estimated means:\n {sum_rewards / (counts[:, None] + 1e-10)}")  # Broadcasting to avoid division by zero

    # Thompson Sampling with Underdamped Langevin dynamics
    counts = np.zeros(n_arms)  # number of times to play arm
    sum_rewards = np.zeros((n_arms, dim))
    choose_arm_logs = []
    regret_total = 0
    regret_logs = [regret_total]
    observation = []
    current_position = []
    current_velocity = []
    for i in range(n_arms):
        observation.append([])
        current_position.append(prior_mean)
        current_velocity.append(np.zeros(prior_mean.shape))

    # n_rounds = 100
    pbar = tqdm(range(n_rounds), dynamic_ncols=True, smoothing=0.1, desc='Underdamped Langevin')
    for e in pbar:
        sampled_means = []

        for arm in range(n_arms):
            if counts[arm] == 0:
                if len(observation[arm]) == 0:
                    sampled_means.append(
                        np.random.multivariate_normal(prior_mean, prior_variance))  # No observation, sample from prior
                else:
                    # sampled_mean = under_damped_exact(observation[arm], step_size, n_iterations, prior_mean,
                    #                                   np.linalg.inv(prior_variance))
                    sampled_mean, vel = under_damped_stochastic(
                        observation[arm], step_size, n_iterations, prior_mean, np.linalg.inv(prior_variance),
                        [current_position[arm], current_velocity[arm]], gamma=gamma, batch_size=batch)
                    sampled_means.append(sampled_mean)
                    current_position[arm] = sampled_mean
                    current_velocity[arm] = vel

            else:
                # Sample from posterior using Langevin dynamics
                obs = np.random.multivariate_normal(true_means[arm], arm_covariances[arm])
                observation[arm].append(obs)
                # sampled_mean = under_damped_exact(observation[arm], step_size, n_iterations, prior_mean,
                #                                   np.linalg.inv(prior_variance))
                sampled_mean, vel = under_damped_stochastic(
                    observation[arm], step_size, n_iterations, prior_mean, np.linalg.inv(prior_variance),
                    [current_position[arm], current_velocity[arm]], gamma=gamma, batch_size=batch)
                sampled_means.append(sampled_mean)
                current_position[arm] = sampled_mean
                current_velocity[arm] = vel

        chosen_arm = np.argmax([np.linalg.norm(mean) for mean in sampled_means])  # Select arm with highest norm
        choose_arm_logs.append(chosen_arm)
        reward = np.random.multivariate_normal(true_means[chosen_arm], arm_covariances[chosen_arm])
        if chosen_arm == 0:
            regret = 0
        else:
            optimal_reward = np.random.multivariate_normal(true_means[0], arm_covariances[0])
            regret = np.linalg.norm(optimal_reward) - np.linalg.norm(reward)
        regret_total += regret
        regret_logs.append(regret_total)

        # Update counts and observed rewards
        counts[chosen_arm] += 1
        sum_rewards[chosen_arm] += reward

        pbar.set_postfix({
            'Reward': '{0:1.4e}'.format(np.linalg.norm(reward)),
            'Regret': '{0:1.4e}'.format(regret_total)})

        # plot_figs(regret_logs, regret_logs, args)

    return counts, regret_logs, choose_arm_logs


if __name__ == "__main__":
    # Get flags
    flags = get_flags()
    for i in range(100):
        count, logs, arm_logs = run_func(flags)
        print(count)

        logs_ = np.loadtxt('./logs/results/underdamped_batch010.txt', dtype=float)
        logs_ = np.vstack((logs_, logs))
        np.savetxt('./logs/results/underdamped_batch010.txt', logs_, fmt='%1.2f')

