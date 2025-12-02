from gpi_pack.TarNet import estimate_k_ate
import numpy as np
import pytest

def test_estimate_k_ate():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    hiddens = np.random.rand(n_samples, 4096)  # 4096-dimensional hidden states
    treatment = np.random.randint(0, 2, n_samples)  # Binary treatment
    outcome = np.random.rand(n_samples)  # Continuous outcome
    
    # Estimate the average treatment effect (ATE) using the synthetic data
    ate, se = estimate_k_ate(
        R=hiddens,
        Y=outcome,
        T=treatment,
        K=2,  # K-fold cross-fitting
        lr=2e-5,  # learning rate
        nepoch=1,
        architecture_y=[200, 1],  # outcome model architecture
        architecture_z=[2048],  # deconfounder architecture
    )
    
    # Assert that ate is a float
    assert isinstance(ate, float)
    # Assert that standard error is positive
    assert se > 0