import numpy as np
import matplotlib.pyplot as plt


def Hermite_nongaussian_soft(y, a_3, a_4):
    h_3 = (a_3 / 6) * (1 - 0.015 * np.abs(a_3) + 0.3 * a_3**2) / \
        (1 + 0.2 * (a_4 - 3))
    h_40 = ((1 + 1.25 * (a_4 - 3))**(1/3) - 1) / 10
    h_4 = h_40 * (1 - (1.43 * a_3**2 / (a_4 - 3)))**(1 - 0.4 * a_4**0.8)
    k = (1 + 2 * h_3**2 + 6 * h_4**2)**-0.5

    a = h_3 / 3 * h_4
    b = 1 / 3 * h_4
    c = (b - 1 - a**2)**3
    xi_y = 1.5 * b * (a + y / k) - a**3
    x = (np.sqrt(xi_y**2 + c) + xi_y)**(1/3) - (np.sqrt(xi_y**2 + c) - xi_y)**(1/3) - a
    norm = 1 / np.sqrt(2 * np.pi)

    print("h_3: ", h_3)
    print("h_40: ", h_40)
    print("h_4: ", h_4)
    print("k: ", k)
    print("a: ", a)
    print("b: ", b)
    print("c: ", c)
    print("xi_y: ", xi_y)
    print("xi_y**2 + c: ", xi_y**2 + c)
    print("norm: ", norm)

    return  norm * np.exp(- x**2 / 2) / np.abs(k * (1 + 2 * h_3 * x + 3 * h_4 * x**2 - 3 * h_4))


def Hermite_nongaussian_hard():
    pass


def main():
    x = np.arange(0, 10, 0.1)
    y = Hermite_nongaussian_soft(x, 2., 5.)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    ax.plot(x, y, 'k-')

    plt.show()

if __name__ == '__main__':
    main()