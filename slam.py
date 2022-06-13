from map import choose_landmark_map
import numpy as np
from ekf import ExtendedKalmanFilter
import matplotlib.pyplot as plt
import utils
import time

PLOT_ELLIPSES = False
INNERTIAL = True
TEST_SIMPLE = True
VERBOSE = False


def main():
    # Loading (Pre-processed) Simulation Data
    realPose = np.nan_to_num(np.load("data\pose3D.npy"), nan=10)
    realTwist = np.nan_to_num(np.load("data\\twist3D.npy"), nan=10)
    print(realPose.shape, realTwist.shape)

    # Simulation Info
    MAX_TIME = 12
    dt = 1/62.5  # for this data
    #dt = 0.2

    # Define Landmark Map
    l = choose_landmark_map("iss", 20)
    n_landmarks = l.shape[0]
    l_id = np.transpose([np.linspace(0, n_landmarks-1, n_landmarks, dtype='uint16')])
    #print(l.shape, l_id.shape)
    m = np.hstack((l, l_id))
    print(m.shape)

    # Starting Guesstimate (prob.)
    x0 = np.array([0, 0, 0]).T
    print("x0:", x0, x0.shape)
    mu0 = np.append(x0, np.zeros((2*n_landmarks, 1)))
    sigma0 = 1e7*np.eye(2*(n_landmarks)+3)
    sigma0[:3,:3] = np.zeros((3,3))
    print("Mean State and Covariance Matrix dims:", mu0.shape, sigma0.shape)

    # Process noise Cov. matrix
    R = np.diag([1, 1, np.deg2rad(30.0)]) ** 2
    # Observation noise Cov. matrix
    Q = np.diag([1.4, 1.1]) ** 2
    # For LASER: Qt = np.diag([0.02, np.deg2rad(0.1)]) ** 2

    # Init. Kalman Filter
    EKF = ExtendedKalmanFilter(R, Q, mu0, sigma0, dt)
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
    for timestep in range(2, realPose.shape[0]):
        # Moving / Sensing
        t = timestep*dt
        pose = realPose[timestep]
        x = pose[0:2]
        u_l = realTwist[timestep]
        V_est = np.sqrt(u_l[0]**2 + u_l[1]**2)
        u = np.array([V_est, u_l[2]])
        if VERBOSE:
            print("Real position: ", x.T)

        # Simulate measurements
        if not TEST_SIMPLE:
            zvar = 1.1 ** 2
            noise = np.random.normal(0, zvar, 4)
            zp = np.linalg.norm(x - m, axis=1) + noise
            print("Range meas. ", zp)
            # Just considering dist. to second landmark
            z = np.array([zp[1], 0.01])
        else:
            z = np.array([x[0] + 0.1*np.random.normal(0, 0.15),
                          x[1] + 0.1*np.random.normal(0, 0.13)])
        if VERBOSE:
            print("Measurement z: ", z.T)

        # Run Localization
        EKF.do_filter(u, None) # z
        if VERBOSE:
            print("Time:", dt*timestep, " Position: (",
                  EKF.mu[0], ",", EKF.mu[1], ")\n")

        # Collect data for display later
        time.append(t)
        # real_position.append(x)
        real_position.append(pose)
        measurements.append(z)
        pred.append(EKF.mu)
        cov.append(EKF.sigma)

    """
    Plotting:
    """
    # Plot trajectory
    fig1 = plt.figure(1)
    plt.subplot(121)
    ax = plt.gca()
    real_position = np.array(real_position)
    plt.plot(real_position[:, 0], real_position[:, 1], ".-.")

    # Plot Measurements
    measurements = np.array(measurements)
    plt.plot(measurements[:, 0], measurements[:, 1], ".", alpha=0.4)

    # Plot Predicted Position
    pred = np.array(pred)
    plt.plot(pred[:, 0], pred[:, 1], ".")
    if PLOT_ELLIPSES:
        i = 0
        for element in cov:
            utils.draw_cov_ellipse(pred[i, 0:2], element, ax)
            i += 1
    plt.legend(["Real Position", "Measurements", "EKF Prediction"])
    plt.ylabel("y")
    plt.xlabel("x")

    # Plot Error
    plt.subplot(122)
    plt.plot(time, np.linalg.norm(
        real_position[:, 0:2] - pred[:, 0:2], axis=1))
    plt.xlabel("Time (s)")
    plt.ylabel("RMSE")

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
