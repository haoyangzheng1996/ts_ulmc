import numpy as np
import random


def over_damped_exact(prior_mean, prior_variance_inv,
                      observations=None, step_size=1e-4, n_iterations=1e3):
    mu = prior_mean.copy()
    covariance = 2 * step_size * np.identity(mu.shape[0])

    for _ in range(n_iterations):
        n = len(observations)
        gradient = -prior_variance_inv.dot(mu - prior_mean) + n * (np.mean(observations, axis=0) - mu)
        mu = np.random.multivariate_normal(mean=mu - step_size * gradient, cov=covariance)

    return mu


def under_damped_exact(prior_mean, prior_variance_inv,
                       observations=None, sample_v=None, step_size=1e-4, n_iterations=1e3, gamma=2, L=1):
    x_mean = prior_mean.copy()
    v_mean = np.zeros_like(x_mean)
    dim = prior_mean.shape[0]
    t = step_size

    for _ in range(n_iterations):
        gradient = -prior_variance_inv.dot(x_mean - prior_mean) + len(observations) * (
                np.mean(np.concatenate(observations, axis=0), axis=0) - x_mean)

        x_sub = x_mean + 0.5 * (1 - np.exp(-2 * t)) * v_mean - 1 / (2 * L) * (
                t - 0.5 * (1 - np.exp(-2 * t))) * gradient
        v_sub = v_mean * np.exp(-2 * t) - 1 / (2 * L) * (1 - np.exp(-2 * t)) * gradient
        mean_ = np.hstack((x_sub, v_sub)).reshape(-1)

        var_1 = 1 / L * (t - 1 / 4 * np.exp(-4 * t) - 3 / 4 + np.exp(-2 * t)) * np.identity(dim)
        var_2 = 1 / L * (1 - np.exp(-4 * t)) * np.identity(dim)
        var_3 = 1 / (2 * L) * (1 + np.exp(-4 * t) - 2 * np.exp(-2 * t)) * np.identity(dim)
        covariance_ = np.vstack((np.hstack((var_1, var_3)), np.hstack((var_3, var_2))))
        try:
            mean_next = np.random.multivariate_normal(mean=mean_, cov=covariance_)
        except ValueError:
            print('valur error')
        x_mean = mean_next[:dim]
        v_mean = mean_next[dim:]

    return x_mean, v_mean


# def over_damped_stochastic(prior_mean, prior_variance_inv,
#                            observations=None, step_size=1e-4, n_iterations=1e3, batch_size=10):
#     mu = prior_mean.copy()
#     covariance = 2 * step_size * np.identity(mu.shape[0])
#
#     for _ in range(n_iterations):
#         n = len(observations)
#         if batch_size >= n:
#             gradient = -prior_variance_inv.dot(mu - prior_mean) + n * (np.mean(observations, axis=0) - mu)
#         else:
#             sub_observe = random.sample(observations, k=batch_size)
#             gradient = -prior_variance_inv.dot(mu - prior_mean) + n * (np.mean(sub_observe, axis=0) - mu)
#         mu = np.random.multivariate_normal(mean=mu - step_size * gradient, cov=covariance)
#
#     return mu
#
#
# def under_damped_stochastic(prior_mean, prior_variance_inv,
#                             observations=None, sample_v=None, step_size=1e-4, n_iterations=1e3, gamma=2, L=1, batch_size=10):
#     x_mean = prior_mean.copy().reshape(-1)
#     if sample_v is None:
#         v_mean = np.zeros_like(x_mean)
#     else:
#         v_mean = sample_v
#     dim = prior_mean.shape[0]
#     t = step_size
#
#     for _ in range(n_iterations):
#         if observations is None:
#             observations = np.zeros(x_mean.shape)
#             n = 0
#         else:
#             n = len(observations)
#
#         if batch_size >= n:
#             gradient = -prior_variance_inv.dot(x_mean - prior_mean) + n * (np.mean(observations, axis=0) - x_mean)
#         else:
#             sub_observe = random.sample(observations, k=batch_size)
#             try:
#                 gradient = -prior_variance_inv.dot(x_mean - prior_mean) + n * (np.mean(sub_observe, axis=0) - x_mean)
#             except ValueError:
#                 print("value error")
#         x_sub = x_mean + 0.5 * (1 - np.exp(-2 * t)) * v_mean - 1 / (2 * L) * (
#                 t - 0.5 * (1 - np.exp(-2 * t))) * gradient
#         v_sub = v_mean * np.exp(-2 * t) - 1 / (2 * L) * (1 - np.exp(-2 * t)) * gradient
#         mean_ = np.hstack((x_sub, v_sub)).reshape(-1)
#
#         var_1 = 1 / L * (t - 1 / 4 * np.exp(-4 * t) - 3 / 4 + np.exp(-2 * t)) * np.identity(dim)
#         var_2 = 1 / L * (1 - np.exp(-4 * t)) * np.identity(dim)
#         var_3 = 1 / (2 * L) * (1 + np.exp(-4 * t) - 2 * np.exp(-2 * t)) * np.identity(dim)
#         covariance_ = np.vstack((np.hstack((var_1, var_3)), np.hstack((var_3, var_2))))
#         try:
#             mean_next = np.random.multivariate_normal(mean=mean_, cov=covariance_)
#         except ValueError:
#             print('valur error')
#         x_mean = mean_next[:dim]
#         v_mean = mean_next[dim:]
#
#     return x_mean, v_mean


def over_damped_stochastic(observations, step_size, n_iterations, prior_mean, prior_variance_inv, current,
                           batch_size=10, prior_type=None):
    x_mean = current.copy()
    # covariance = 2 * step_size * np.identity(mu.shape[0])

    for _ in range(n_iterations):
        n = len(observations)
        if batch_size < n:
            sub_observe = random.sample(observations, k=batch_size)
            try:
                if prior_type is not None:
                    gradient = n * (np.mean(sub_observe, axis=0) - x_mean)
                else:
                    gradient = -prior_variance_inv.dot(x_mean - prior_mean) + n * (
                            np.mean(sub_observe, axis=0) - x_mean)
            except ValueError:
                print("value error")
        elif n == 0:
            gradient = -prior_variance_inv.dot(x_mean - prior_mean)
        else:
            if prior_type is not None:
                gradient = n * (np.mean(observations, axis=0) - x_mean)
            else:
                gradient = -prior_variance_inv.dot(x_mean - prior_mean) + n * (np.mean(observations, axis=0) - x_mean)
                
        x_mean = x_mean - step_size * gradient + np.sqrt(2 * step_size) * np.random.randn(x_mean.shape[0])

    return x_mean


def under_damped_stochastic(observations, step_size, n_iterations, prior_mean, prior_variance_inv, current,
                            gamma=2, L=1, batch_size=10, prior_type=None):
    # print("Gamma %.4f" % gamma)
    x_mean = current[0].copy()
    v_mean = current[1].copy()
    dim = prior_mean.shape[0]
    t = step_size

    for _ in range(n_iterations):
        n = len(observations)
        if batch_size < n:
            sub_observe = random.sample(observations, k=batch_size)
            try:
                if prior_type is not None:
                    gradient = n * (np.mean(sub_observe, axis=0) - x_mean)
                else:
                    gradient = -prior_variance_inv.dot(x_mean - prior_mean) + n * (
                            np.mean(sub_observe, axis=0) - x_mean)
            except ValueError:
                print("value error")
        elif n == 0:
            gradient = -prior_variance_inv.dot(x_mean - prior_mean)
        else:
            if prior_type is not None:
                gradient = n * (np.mean(observations, axis=0) - x_mean)
            else:
                gradient = -prior_variance_inv.dot(x_mean - prior_mean) + n * (np.mean(observations, axis=0) - x_mean)
        # x_sub = x_mean + (1 - np.exp(-gamma * t)) * v_mean / gamma - (
        #         t - (1 - np.exp(-gamma * t)) / gamma) * gradient * (gamma * L)
        # v_sub = v_mean * np.exp(-gamma * t) - (1 - np.exp(-gamma * t)) / (gamma * L) * gradient
        # mean_ = np.hstack((x_sub, v_sub)).reshape(-1)
        #
        # var_1 = 2 * (t - np.exp(-2 * gamma * t) / (2 * gamma) - 3 / (2 * gamma) +
        #              2 * np.exp(-gamma * t) / gamma) / (gamma * L) * np.identity(dim)
        # var_2 = (1 - np.exp(-2 * gamma * t)) / L * np.identity(dim)
        # var_3 = (1 + np.exp(-2 * gamma * t) - 2 * np.exp(-gamma * t)) / (gamma * L) * np.identity(dim)
        # covariance_ = np.vstack((np.hstack((var_1, var_3)), np.hstack((var_3, var_2))))
        # try:
        #     mean_next = np.random.multivariate_normal(mean=mean_, cov=covariance_)
        # except ValueError:
        #     print('valur error')
        # x_mean = mean_next[:dim]
        # v_mean = mean_next[dim:]

        x_sub = x_mean + (1 - np.exp(-gamma * t)) * v_mean / gamma - (
                t - (1 - np.exp(-gamma * t)) / gamma) * gradient * (gamma * L)
        v_sub = v_mean * np.exp(-gamma * t) - (1 - np.exp(-gamma * t)) / (gamma * L) * gradient

        # Diagonal variances computed separately
        var_1 = 2 * (t - np.exp(-2 * gamma * t) / (2 * gamma) - 3 / (2 * gamma) +
                     2 * np.exp(-gamma * t) / gamma) / (gamma * L) 
        var_2 = (1 - np.exp(-2 * gamma * t)) / L 

        # Generating samples using independent Gaussian draws
        x_mean = np.random.normal(loc=x_sub, scale=np.sqrt(var_1), size=dim)
        v_mean = np.random.normal(loc=v_sub, scale=np.sqrt(var_2), size=dim)

        # u = 1 / L
        #
        # # Explicit methods
        # delta_v = -gamma * v_mean * t - u * gradient * t + np.sqrt(2 * gamma * u * t) * np.random.randn(
        #     dim)
        # v_mean = v_mean + delta_v
        #
        # # # Implicit methods
        # # numerator = v_mean - u * gradient * t + np.sqrt(2 * gamma * u * t) * np.random.randn(dim)
        # # v_next = numerator / (1 + gamma * t)
        #
        # x_mean = x_mean + v_mean * t

    return x_mean, v_mean