import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

X_MIN, X_MAX = -1, 1
T_MIN, T_MAX = 0, 1

N_T = 100
N_X = 100

DX = (X_MAX - X_MIN) / N_X
DT = (T_MAX - T_MIN) / N_T

A = 0.1

PLOT_STRIDE = 10


def initial_state(x):
    return np.array([1 if x_i < 0 else 0 for x_i in x])


def lax_friedrichs_step(u) -> np.ndarray:
    out = np.array([])
    for i_x in range(N_X):
        u_p = u[max(0, i_x-1)]
        u_n = u[min(i_x+1, N_X-1)]

        f_1 = 0.5
        f_2 = 0.5 * (DT/DX) * A

        u_i = f_1 * (u_p + u_n) + f_2 * (u_p - u_n)
        out = np.append(out, u_i)
    return out


def plot(x, u):
    for u_i in u[::PLOT_STRIDE]:
        plt.plot(x, u_i)
    plt.show()


if __name__ == "__main__":
    x = np.linspace(X_MIN, X_MAX, N_X)
    u0 = initial_state(x)
    u = [u0]

    for i_t in tqdm(range(N_T)):
        u.append(lax_friedrichs_step(u[-1]))

    plot(x, u)
