from map import choose_landmark_map
import numpy as np
from ekf import ExtendedKalmanFilter
import matplotlib.pyplot as plt
from utils import sim_measurements, draw_cov_ellipse
import time

PLOT_ELLIPSES = False
TEST_SIMPLE = False
VERBOSE = 1
JUST_PRED = False

def main():
    # Loading (Pre-processed) Simulation Data
    realPose = np.load("data\pose3D.npy")[1:]
    realTwist = np.load("data\\twist3D.npy")[1:]
    imu = np.load("data\\theta+w.npy")[1:]
    if VERBOSE:
        print("Simulator Data:", realPose.shape, realTwist.shape, imu.shape)

    # Simulation Info
    DOWNSAMPLE = 60
    dt = 1/62.5 * DOWNSAMPLE  # for this data

    # Define Landmark Map
    m, n_landmarks = choose_landmark_map("iss", 20)
    if VERBOSE:
        print("Map shape:", m.shape)

    # Starting Guesstimate (prob.)
    mu0 = np.transpose([9.8, -9.8, np.deg2rad(10.0)])
    sigma0 = np.diag([1.2, 1.2, np.deg2rad(45.0)]) ** 2
    if VERBOSE:
        print("Mean State and Covariance Matrix dims:", mu0.shape, sigma0.shape)

    # Process noise Cov. matrix
    Rt = np.diag([0.01, 0.01, np.deg2rad(20.0)]) ** 2
    # Observation noise Cov. matrix
    #Qt = np.diag([0.8, 0.7]) ** 2
    # For LASER: 
    Qt = np.diag([0.02, np.deg2rad(0.1)]) ** 2

    # Init. Kalman Filter
    EKF = ExtendedKalmanFilter(Rt, Qt, mu0, sigma0, dt, map=m, TEST_DUMMY=TEST_SIMPLE)
    if VERBOSE:
        print("Rt: \n", Rt)
        print("Qt: \n", Qt, "\n")

    # Init. Actions and Measurements
    u_l = np.array([0, 0, 0])
    u = np.array([0, 0])
    z = np.zeros((2, 1))
    V_prev = 0

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
        #V_est = (V_est + V_prev)/2
        if V_est > 0.05:
            V_est = 0.15
        u = np.array([V_est, u_l[2]])
        #V_prev = V_est
        if VERBOSE > 1:
            print("Real position: ", x[:2].T)

        # Simulate measurements
        if not TEST_SIMPLE:
            # More realistic Sim.
            zp, FOV = sim_measurements(x, Qt, m)
            if zp.size:
                z = zp
        # "Position" Sensor
        else:
            z = np.array([x[0] + 0.1*np.random.normal(0, 0.15),
                          x[1] + 0.1*np.random.normal(0, 0.13)])
        if VERBOSE > 1:
            print("Measurements z:\n", z)

        # Run Localization
        if JUST_PRED:
            EKF.do_filter(u, None)
        else:
            EKF.do_filter(u, z.T, VERBOSE>1)
        if VERBOSE > 1:
            print("Time:", dt*timestep, " Position: (",
                  EKF.mu[0], ",", EKF.mu[1], ")\n")

        # Collect data for display later
        time.append(t)
        real_position.append(x)
        if TEST_SIMPLE:
            measurements.append(z)
        else:
            measurements.append(z[0, :])
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
    if TEST_SIMPLE:
        measurements = np.array(measurements)
        plt.plot(measurements[:, 0], measurements[:, 1], ".", alpha=0.4)

    # Plot Predicted Position
    pred = np.array(pred)
    plt.plot(pred[:, 0], pred[:, 1], ".")
    if PLOT_ELLIPSES:
        i = 0
        for element in cov:
            draw_cov_ellipse(pred[i, 0:2], element, ax)
            i += 1
    if TEST_SIMPLE:
        plt.legend(["Real Position", "Measurements", "EKF Prediction"])
    else:
        plt.legend(["Real Position", "EKF Prediction"])
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
