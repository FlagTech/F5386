import numpy as np

a = np.array([[11,22,13,74,35,6,27,18]])

min_value = np.min(a)
max_value = np.max(a)
print(min_value, max_value)

min_idx = np.argmin(a)
max_idx = np.argmax(a)
print(min_idx, max_idx)
