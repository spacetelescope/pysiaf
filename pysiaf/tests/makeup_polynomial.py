import numpy as np

def makeup_polynomial():
    # invent a random but plausible polynomial array
    order = 5
    terms = (order + 1) * (order + 2) // 2
    a = np.zeros(terms)
    a[1] = 0.05 + 0.01 * np.random.rand(1)
    a[2] = 0.0001 * np.random.rand(1)
    a[3:6] = 1.0e-7 * np.random.rand(3)
    a[6:10] = 1.0e-10 * np.random.rand(4)
    a[10:15] = 1.0e-13 * np.random.rand(5)
    a[15:21] = 1.0e-15 * np.random.rand(6)
    return(a)
