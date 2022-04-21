import numpy as np


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def Geometric_Sum(deficits, i, **kwargs):
    return 1 - np.cumprod(1 - deficits[:i, i])[-1]


def Linear_Sum(deficits, i, **kwargs):
    return np.sum(deficits[:i, i])


def Energy_Balance(deficits, i, **kwargs):
    a = deficits[:i, i]
    b = deficits[:i, -1]
    inflow = kwargs["inflow"]
    return 1 - (np.sqrt(inflow**2 - np.sum((b**2) - (((1 - a) * b)**2))) / inflow)


def Sum_Squares(deficits, i, **kwargs):
    return np.sqrt(np.sum(deficits[:i, i]**2))


if __name__ == "__main__":
    Geometric_Sum()