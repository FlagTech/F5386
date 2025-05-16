import numpy as np

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a)
b, c, d = a[1:3], a[:4], a[3:]
print(b, c, d)
