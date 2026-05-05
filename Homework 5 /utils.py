import numpy as np

def win_prob(x_i, x_j):
    """Logistic function"""
    return 1 / (1 + np.exp(-(x_i - x_j)))

def sample_truncated_poisson(N):
    probs = np.exp(-np.arange(1, N+1)/(N/3))
    probs /= probs.sum()
    return np.random.choice(np.arange(1, N+1), p=probs)