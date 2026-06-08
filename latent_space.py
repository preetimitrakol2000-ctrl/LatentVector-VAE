import math
from prob_math import calculate_kl_divergence

class LatentEncoderNode:
    def __init__(self):
        self.weight_mu = 0.45
        self.weight_logvar = -0.12

    def map_to_distribution(self, input_feature):
        """Encodes explicit target variables into distribution metrics."""
        mean = input_feature * self.weight_mu
        log_var = input_feature * self.weight_logvar
        kl_loss = calculate_kl_divergence(mean, log_var)
        return mean, log_var, kl_loss
