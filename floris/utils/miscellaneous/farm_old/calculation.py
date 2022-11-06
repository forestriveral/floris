import os
import shutil
import numpy as np



def calculation(a, n, price):
    return (1 - (1 - (a / 100))**n) * 100, price * n


# print(calculation(5, 20, 700))

# print(calculation(10, 10, 1800))

# print(calculation(5, 6, 3000))


# a = np.arange(12).reshape((3, 4))
# print(a)

# # b = a[:, 0, :]
# # b[b <= 3] = 0.
# print(np.sum(a, axis=0))
