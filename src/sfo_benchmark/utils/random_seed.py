import random
import numpy as np

def set_global_seed(seed:int)->np.random.Generator:
    random.seed(seed); np.random.seed(seed); return np.random.default_rng(seed)
