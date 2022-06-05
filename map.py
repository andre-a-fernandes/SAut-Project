import numpy as np
import matplotlib.pyplot as plt


def choose_landmark_map(form, scale):
    l = -1
    if form == "square":
        l = scale*np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elif form == "line":
        l = scale*np.array([[0, 0], [0.5, 0], [1, 0]])
    elif form == "slightly":
        l = scale*np.array([[0, 0], [0.2, 0], [0.5, 0.1], [1, -0.1]])
    return l


if __name__ == '__main__':

    # Choose landmark map here
    l = choose_landmark_map("square")

    # Number of landmarks
    N = len(l)

    # Robot
    p = np.array([0.1, 0.8])

    # Variance of measurements
    zvar = 0.1**2

    # Simulate measurements
    noise = np.random.normal(0, zvar, 4)
    zp = np.linalg.norm(p - l, axis=1) + noise
    #print(n, "\n", zp)

    plt.figure()
    plt.scatter(l[:, 0], l[:, 1])
    for i in range(N):
        plt.annotate(f"landmark_{i}", (l[i, 0], l[i, 1]))
    plt.scatter(p[0], p[1])
    plt.annotate("robot", (p[0], p[1]))
    plt.axis('equal')
    plt.show()
