import numpy as np
import matplotlib.pyplot as plt



ar = np.zeros((4,4,4),dtype=float)
ar[2][3] = 10
print(ar)

index = np.argmax(ar)

print(np.unravel_index(index,ar.shape))
print(ar[(2,3)])