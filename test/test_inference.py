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


def test_estimate_k_ate_with_convs():
    np.random.seed(0)
    n_samples = 20
    channels, height, width = 3, 8, 8
    images = np.random.rand(n_samples, channels, height, width).astype(np.float32)
    treatment = np.random.randint(0, 2, n_samples)
    outcome = np.random.rand(n_samples).astype(np.float32)

    conv_layers = [
        {"in_channels": channels, "out_channels": 4, "kernel_size": 3, "padding": 1,
         "pool": {"kernel_size": 2}},
        {"out_channels": 8, "kernel_size": 3, "padding": 1}
    ]

    ate, se = estimate_k_ate(
        R=images,
        Y=outcome,
        T=treatment,
        K=2,
        lr=1e-4,
        nepoch=1,
        architecture_y=[16, 1],
        architecture_z=[32],
        conv_layers=conv_layers,
        batch_size=4,
        valid_perc=0.2,
        verbose=False,
        plot_propensity=False,
    )

    assert isinstance(ate, float)
    assert se >= 0