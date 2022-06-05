from map import choose_landmark_map
import numpy as np
from ekf import ExtendedKalmanFilter
import matplotlib.pyplot as plt
import utils, time

PLOT_ELLIPSES = True
INNERTIAL = True
TEST_SIMPLE = True


def main():
    # Simulation Info
    MAX_TIME = 12
    dt = 0.2

    # Define Landmark Map
    #m = np.array(choose_landmark_map("square", 20), dtype=np.float32)

    # Starting Guesstimate (prob.)
    mu0 = np.transpose([0, 0, np.deg2rad(45.0)])
    sigma0 = np.diag([1.5, 1.3, np.deg2rad(20.0)]) ** 2
    print("Mean State and Covariance Matrix dims:", mu0.shape, sigma0.shape)

    # Process noise Cov. matrix
    R = np.diag([0.1, 0.1, np.deg2rad(10.0)]) ** 2
    # Observation noise Cov. matrix
    Q = np.diag([1.4, 1.1]) ** 2
    # For LASER: Qt = np.diag([0.02, np.deg2rad(0.1)]) ** 2

    # Init. Kalman Filter
    EKF = ExtendedKalmanFilter(R, Q, mu0, sigma0, dt)
    print("Rt: \n", EKF.Rt)
    print("Qt: \n", EKF.Qt, "\n")

    # Init. Actions and Measurements
    u_l = np.array([np.sqrt(2), np.sqrt(2), 0])
    u = np.array([2, 0])
    z = np.zeros((2, 1))

    # Setup history vectors
    real_position = []
    measurements = []
    pred = []
    cov = []

    # Robot in an environment
    for timestep in range(int(MAX_TIME/dt)):
        
        # Moving / Sensing
        x = np.array([dt*timestep*np.sqrt(2), dt*timestep*np.sqrt(2)])
        #V_est = (u_l[0]/np.cos(x[2]) + u_l[1]/np.sin(x[2])) / 2
        #u = np.array([V_est, 0])
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
            z = np.array([x[0] + np.random.normal(0, 2.0),
                          x[1] + np.random.normal(0, 1.1)])
        print("Measurement z: ", z.T)

        # Run Localization
        EKF.do_filter(u, None)
        print("Time:", dt*timestep, " Position: (",
              EKF.mu[0], ",", EKF.mu[1], ")\n")

        # Collect data for display later
        real_position.append(x)
        measurements.append(z)
        pred.append(EKF.mu[0:2])
        cov.append(EKF.sigma)


    """
    Plotting:
    """
    # Plot trajectory
    plt.figure()
    plt.subplot(121)
    ax = plt.gca()
    real_position = np.array(real_position)
    plt.plot(real_position[:, 0], real_position[:, 1], ".-.")

    # Plot Measurements
    measurements = np.array(measurements)
    plt.plot(measurements[:, 0], measurements[:, 1], ".")

    # Plot Predicted Position
    pred = np.array(pred)
    plt.plot(pred[:, 0], pred[:, 1], ".")
    if PLOT_ELLIPSES:
        i = 0
        for element in cov:
            utils.draw_cov_ellipse(pred[i, :], element, ax)
            i += 1
    plt.legend(["Real Position", "Measurements", "EKF Prediction"])

    # Plot Error
    plt.subplot(122)
    plt.plot(np.linalg.norm(real_position - pred, axis=1))
    plt.xlabel("Time (s) * 5")
    plt.ylabel("RMSE")

    # Show Graphics
    plt.show()


if __name__ == '__main__':
    startTime = time.time()
    main()
    print("Program took", time.time() - startTime, "seconds")
