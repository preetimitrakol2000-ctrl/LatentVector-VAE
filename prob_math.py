import math

def calculate_kl_divergence(mean, log_variance):
    """Computes structural relative entropy boundaries against Gaussian targets."""
    return -0.5 * (1.0 + log_variance - (mean ** 2) - math.exp(log_variance))

def compute_mse_loss(reconstruction, target_origin):
    return sum((r - t) ** 2 for r, t in zip(reconstruction, target_origin)) / len(target_origin)
