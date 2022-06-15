from map import choose_landmark_map
import numpy as np
from ekf_slam import ExtendedKalmanFilter
import matplotlib.pyplot as plt
from utils import sim_measurements, draw_cov_ellipse
import time

PLOT_ELLIPSES = False
VERBOSE = 1

def main():
    # Loading (Pre-processed) Simulation Data
    realPose = np.load("data\pose3D.npy")[1:]
    realTwist = np.load("data\\twist3D.npy")[1:]
    imu = np.load("data\\theta+w.npy")[1:]
    if VERBOSE:
        print("Simulator Data:", realPose.shape, realTwist.shape, imu.shape)

    # Simulation Info
    MAX_TIME = 12
    dt = 1/62.5 #* 6 # for this data

    # Define Landmark Map
    m, n_landmarks = choose_landmark_map("iss", 20)
    if VERBOSE:
        print("Map shape:", m.shape)

    # Starting Guesstimate (prob.)
    x0 = np.array([9.8, -9.8, 0]).T
    print("Robot zero-state, x0:", x0, x0.shape)
    mu0 = np.append(x0, np.zeros((2*n_landmarks, 1)))
    # "Infinity" on the diagonal plus (3x3) zero Cov. for the robot
    #sigma0 = 1e6 * np.eye(2*(n_landmarks)+3)
    #sigma0[:3,:3] = np.zeros((3,3))
    #sigma0 = np.zeros((2*(n_landmarks)+3, 2*(n_landmarks)+3))
    #sigma0[3:, 3:] = 1e6 * np.ones((2*n_landmarks, 2*n_landmarks))
    # All "infinity" except state Cov.
    sigma0 = 1e6 * np.ones((2*n_landmarks+3, 2*n_landmarks+3))
    sigma0[:3, :3] = np.zeros((3,3))
    if VERBOSE:
        print("Mean State and Covariance Matrix dims:", mu0.shape, sigma0.shape)

    # Process noise Cov. matrix
    Rt = np.diag([0.1, 0.1, np.deg2rad(20.0)]) ** 2
    # Observation noise Cov. matrix
    #Qt = np.diag([1.4, 1.1]) ** 2
    # For LASER:
    Qt = np.diag([0.02, np.deg2rad(6)]) ** 2

    # Init. Kalman Filter
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
    for timestep in range(2, realPose.shape[0]):#, 6):
        # Moving / Sensing
        t = timestep*dt
        x = realPose[timestep]
        u_l = realTwist[timestep]
        V_est = np.sqrt(u_l[0]**2 + u_l[1]**2)
        # Ensure Constant Velocity
        if V_est > 0.05:
            V_est = 0.2
        u = np.array([V_est, u_l[2]])
        if VERBOSE > 1:
            print("Real position: ", x[:2].T)

        # Simulate measurements
        zp, FOV = sim_measurements(x, Qt, m)
        if zp.size:
            z = zp
        if VERBOSE > 1:
            print("Measurements z:\n", z)

        # Run EKF-SLAM
        EKF.do_filter(u, z.T, VERBOSE>2)
        if VERBOSE > 1:
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
    # Plot trajectory and true environment
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(121)
    real_position = np.array(real_position)
    plt.plot(real_position[:, 0], real_position[:, 1], ".-.")
    plt.scatter(m[:, 0], m[:, 1], color="tab:red", marker="d")

    # Plot Measurements
    #measurements = np.array(measurements)
    #plt.plot(measurements[:, 0], measurements[:, 1], ".", alpha=0.4)

    # Plot Predicted Position
    pred = np.array(pred)
    plt.plot(pred[:, 0], pred[:, 1], ".-.", color="green")
    if PLOT_ELLIPSES:
        i = 0
        for element in cov:
            draw_cov_ellipse(pred[i, 0:2], element, ax1)
            i += 1
    plt.scatter(pred[-1, 3::2], pred[-1, 4::2], color="gold", marker="s")
    ax1.title.set_text("True Environment vs SLAM")
    plt.legend(["Real Position", "Landmarks", "EKF Prediction", "Estimated Landmarks"])
    plt.ylabel("y")
    plt.xlabel("x")

    # Plot Error
    ax2 = fig1.add_subplot(122)
    plt.plot(
        time, np.linalg.norm(real_position[:, 0:2] - pred[:, 0:2], axis=1) 
        #+ 
        #np.sqrt((m[:, 0] - pred[:, 3::2])**2 + (m[:, 1] - pred[:, 4::2])**2)
        )
    plt.xlabel("Time (s)")
    plt.ylabel("RMSE")
    ax2.title.set_text("State Error (Pose + Landmarks)")

    # State Vars
    fig2 = plt.figure(2)
    ax = plt.gca()
    plt.subplot(311)
    plt.title("State Variables (Estimates vs. Truth)")
    plt.ylabel("x")
    plt.plot(time, pred[:, 0])
    plt.plot(time, real_position[:, 0])
    plt.subplot(312)
    plt.ylabel("y")
    plt.plot(time, pred[:, 1])
    plt.plot(time, real_position[:, 1])
    plt.subplot(313)
    plt.ylabel("$\Theta$")
    plt.plot(time, np.degrees(pred[:, 2]))
    plt.plot(time, np.degrees(real_position[:, 2]))

    # Show Graphics
    plt.show()


if __name__ == '__main__':
    startTime = time.time()
    main()
    print("Program took", time.time() - startTime, "seconds")
