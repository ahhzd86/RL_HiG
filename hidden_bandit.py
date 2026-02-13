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

import numpy as np
