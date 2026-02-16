import numpy as np
def bandit(arm):
    if arm == 1:
        return np.random.normal(0.0, 1.0)
    elif arm == 2:
        return np.random.normal(2.0, 1.0)
    elif arm == 3:
        return np.random.normal(1.0, 3.0)
    else:
        raise ValueError("Arm must be 1, 2, or 3.")

def bandit_s(arm):
    """
    3-armed bandit with arm-dependent shock probability and magnitude.
    
    Each arm:
        - Has its own normal distribution
        - Has its own shock probability
        - Has its own shock interval
        - Lower shock probability -> larger shock interval
    """

    # Normal distributions
    means = {1: 0.0, 2: 2.0, 3: 1.0}
    stds  = {1: 1.0, 2: 1.0, 3: 3.0}

    # Shock probabilities
    shock_probs = {1: 0.20, 2: 0.10, 3: 0.02}

    # Shock intervals (low prob -> large range)
    shock_ranges = {
        1: (1, 5),
        2: (1, 15),
        3: (1, 100)
    }

    if arm not in [1, 2, 3]:
        raise ValueError("Arm must be 1, 2, or 3.")

    # Shock event
    if np.random.rand() < shock_probs[arm]:
        low, high = shock_ranges[arm]
        return np.random.uniform(low, high)

    # Otherwise normal reward
    return np.random.normal(means[arm], stds[arm])