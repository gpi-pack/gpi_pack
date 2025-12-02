from gpi_pack.TarNet import estimate_k_ate, estimate_k_categorical
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

#def test_estimate_k_categorical():
#    # Generate synthetic data
#    np.random.seed(42)
#    n_samples = 100
#    hiddens = np.random.rand(n_samples, 4096)  # 4096-dimensional hidden states
#    treatment = np.random.randint(0, 2, n_samples)  # Binary treatment
#    outcome_categorical = np.random.randint(0, 3, n_samples)  # Categorical outcome
#    
#    # Estimate categorical pseudo-outcome
#    ate, se = estimate_k_categorical(
#        R=hiddens,
#        Y=outcome_categorical,
#        T=treatment,
#        K=2,  # K-fold cross-fitting
#        lr=2e-5,  # learning rate
#        nepoch=1,
#        num_categories=3,  # number of categories
#        architecture_y=[200, 3],  # outcome model architecture
#        architecture_z=[2048],  # deconfounder architecture
#    )
#    
#    # Assert that ate is a numpy array
#    assert isinstance(ate, np.ndarray), "ATE should be a numpy array"
#    
#    # Assert that ate has length 3
#    assert len(ate) == 3, f"ATE array should have length 3, but has length {len(ate)}"
#    
#    # Also check if se has the same length
#    assert isinstance(se, np.ndarray), "SE should be a numpy array"
#    assert len(se) == 3, f"SE array should have length 3, but has length {len(se)}"
#    
#    # Assert that all standard errors are positive
#    assert np.all(se > 0), "All standard errors should be positive"