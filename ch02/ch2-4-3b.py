import numpy as np

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
b = a.reshape((3, 3))
print(b)
c = b.flatten()
print(c)
