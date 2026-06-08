import random
import math
from latent_space import LatentEncoderNode
from prob_math import compute_mse_loss

if __name__ == "__main__":
    encoder = LatentEncoderNode()
    
    raw_metric_input = 2.85
    mean, log_variance, kl_divergence_cost = encoder.map_to_distribution(raw_metric_input)
    
    # Execute structural reparameterization sampling steps (epsilon variation)
    epsilon_noise = random.gauss(0, 1)
    standard_deviation = math.exp(0.5 * log_variance)
    sampled_latent_vector = mean + (epsilon_noise * standard_deviation)
    
    # Mock decoding reconstitution match step
    synthetic_reconstruction = [sampled_latent_vector * 1.1]
    reconstruction_loss = compute_mse_loss(synthetic_reconstruction, [raw_metric_input])
    
    print("🧬 Activating LatentVector-VAE Deep Statistical Space Layer...")
    print(f"   • Calculated Spatial Mean Point (μ): {mean:.4f}")
    print(f"   • Structural Kullback-Leibler (KL) Cost Variance: {kl_divergence_cost:.4f}")
    print(f"   • Reparameterized Sampled Latent Vector Space: {sampled_latent_vector:.4f}")
    print(f"   • Generated Reconstruction Loss Matrix Score: {reconstruction_loss:.4f}")
