import numpy as np
import matplotlib.pyplot as plt
from utils import sim_measurements


def choose_landmark_map(form, scale=2) -> np.ndarray:
    """
    Set up landmark map (with IDs).

    ## Parameters

    form : String containing a code for choosing the map

    ## Returns

    m : map of landmarks position and ID (N, 3)
    N : number of landmarks
    """
    # Setup map positions
    if form == "square":
        l = scale*np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elif form == "line":
        l = scale*np.array([[0, 0], [0.5, 0], [1, 0]])
    elif form == "slightly":
        l = scale*np.array([[0, 0], [0.2, 0], [0.5, 0.1], [1, -0.1]])
    elif form == "iss":
        l = np.array([[11.5, -10], [12, -6], [9.5, -8], [9.5, -3.5], [12, -1.25],
                     [10, 1.25], [6, 1.25], [2, 1.25], [8, -1.25], [4, -1.25], [-1, 0],
                     [1, -1]])
    else :
        l = -1
    
    # Finish map setup (IDs and count)
    N = l.shape[0]
    l_id = np.transpose([np.linspace(0, N-1, N)])
    m = np.hstack((l, l_id))
    return m, N


if __name__ == '__main__':

    # Mov. Data
    realPose = np.load("data\pose3D.npy")[1:]

    # Choose landmark map
    m, N = choose_landmark_map("iss")
    print("Map shape", m.shape)

    # Robot
    p = np.array([9.81, -9.81, np.deg2rad(90)])

    # Variance of measurements
    Qt = np.diag([0.02, np.deg2rad(0.1)]) ** 2
    zvar = Qt[0][0]

    # Simulate measurements
    noise = np.random.normal(0, zvar, N)
    zp = np.linalg.norm(p[:2] - m[:, :2], axis=1) + noise
    print('z:', zp)

    # More realistic Sim.
    zp_meu, FOV = sim_measurements(p, Qt, m)
    #print('zp_meu:', zp_meu)
    z = zp_meu.T
    print('Measurements z:', z.T)
    print("shape of measurements", z.shape)

    plt.figure()
    plt.plot(realPose[:, 0], realPose[:, 1], ".-.")
    plt.scatter(m[:, 0], m[:, 1], color="tab:red", marker="d")
    for i in range(N):
        plt.annotate(f"l_{i}", (m[i, 0], m[i, 1]))
    plt.scatter(p[0], p[1], color="tab:orange")
    plt.annotate("robot", (p[0], p[1]))
    # Draw FOV lines
    plt.plot([p[0], p[0]+5*np.cos(p[2] + FOV/2)], [p[1], p[1]+5*np.sin(p[2] + FOV/2)], color="gold")
    plt.plot([p[0], p[0]+5*np.cos(p[2] - FOV/2)], [p[1], p[1]+5*np.sin(p[2] - FOV/2)], color="gold")
    plt.show()
