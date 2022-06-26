import numpy as np
import time
import matplotlib.pyplot as plt

from map import choose_landmark_map
from ekf_slam import ExtendedKalmanFilter
from ekf_unknown_correspondences import ExtendedKalmanFilter as EKFUnknown
from utils import sim_measurements
from matching.icp import icp
from plot import plot_environment_and_estimation, plot_state, plot_error

UNKNOWN = False
VERBOSE = 0
PLOT_ELLIPSES = False
DOWNSAMPLE = 1

def main():
    # Loading (Pre-processed) Simulation Data
    realPose = np.load("data\pose3D2.npy")[1:]
    realTwist = np.load("data\\twist3D2.npy")[1:]
    imu = np.load("data\\theta+w2.npy")[1:]
    if VERBOSE:
        print("Simulator Data:", realPose.shape, realTwist.shape, imu.shape)

    # Simulation Info
    dt = 1/62.5 * DOWNSAMPLE # for this data

    # Define Landmark Map
    m, n_landmarks = choose_landmark_map("iss", 20)
    if VERBOSE:
        print("Map shape:", m.shape)
    #n_landmarks += 1

    # Starting Guesstimate (prob.)
    x0 = np.array([0, 0, 0]).T
    mu0 = np.append(x0, np.zeros((2*n_landmarks, 1)))
    # "Infinity" on the diagonal plus (3x3) zero Cov. for the robot
    #sigma0 = 1e6 * np.eye(2*(n_landmarks)+3)
    #sigma0[:3,:3] = np.zeros((3,3))
    #sigma0 = np.zeros((2*(n_landmarks)+3, 2*(n_landmarks)+3))
    #sigma0[3:, 3:] = 1e6 * np.ones((2*n_landmarks, 2*n_landmarks))
    # All "infinity" except state Cov.
    sigma0 = 1e5 * np.ones((2*n_landmarks+3, 2*n_landmarks+3))
    sigma0[:3, :3] = np.zeros((3,3))
    if VERBOSE:
        print("Robot zero-state, x0:", x0, x0.shape)
        print("Mean State and Covariance Matrix dims:", mu0.shape, sigma0.shape)

    # Process noise Cov. matrix
    Rt = np.diag([0.1, 0.1, np.deg2rad(20.0)]) ** 2
    # Observation noise Cov. matrix
    Qt = np.diag([0.02, np.deg2rad(6)]) ** 2

    # Init. Kalman Filter
    if UNKNOWN:
        EKF = EKFUnknown(Rt, Qt, mu0, sigma0, dt)
    else:
        EKF = ExtendedKalmanFilter(Rt, Qt, mu0, sigma0, dt)
    if VERBOSE:
        print("Rt: \n", EKF.Rt)
        print("Qt: \n", EKF.Qt, "\n")

    # Init. Actions and Measurements
    u_l = np.array([0, 0, 0])
    u = np.array([0, 0])
    z = np.zeros((2, 1))

    # Setup history vectors
    real_position = []
    measurements = []
    pred = []
    cov = []
    time = []

    # Robot in an environment
    for timestep in range(2, realPose.shape[0], DOWNSAMPLE):
        # Moving / Sensing
        t = timestep*dt
        x = realPose[timestep]
        u_l = realTwist[timestep]
        V_est = np.sqrt(u_l[0]**2 + u_l[1]**2)
        # Ensure Constant Velocity
        if V_est > 0.05:
            V_est = 0.20
        u = np.array([V_est, u_l[2]])

        # Simulate measurements
        zp, FOV = sim_measurements(x, Qt, m)
        if zp.size:
            z = zp

        # Run EKF-SLAM
        EKF.do_filter(u, z.T, VERBOSE>1)
        
        if VERBOSE > 1:
            print("Real position: ", x[:2].T)
            print("Measurements z:\n", z)
            print("Time:", dt*timestep, " Position: (",
                  EKF.mu[0], ",", EKF.mu[1], ")\n")

        # Collect data for display later
        time.append(t)
        real_position.append(x)
        measurements.append(z[0, :])
        pred.append(EKF.mu)
        cov.append(EKF.sigma)

    """
    Plotting:
    """
    real_position = np.array(real_position)
    pred = np.array(pred)
    # Align prediction via landmarks
    pred[:, 0] += 9.80
    pred[:, 1] -= 9.81
    pred[:, 3::2] += 9.80
    pred[:, 4::2] -= 9.81
    #transf_history, aligned = icp(m[:,:2], pred[:, ])

    fig1 = plt.figure(1)
    plot_environment_and_estimation(fig1, real_position, m, pred, cov, PLOT_ELLIPSES)

    # Plot Error
    ax2 = fig1.add_subplot(132)
    ax3 = fig1.add_subplot(133)
    plot_error(ax2, ax3, time, real_position, m, pred)

    # State Vars
    fig2 = plt.figure(2)
    plot_state(fig2, time, real_position, pred)

    # Show Graphics
    plt.show(block=True)


if __name__ == '__main__':
    startTime = time.time()
    main()
    print("Program took", time.time() - startTime, "seconds")
