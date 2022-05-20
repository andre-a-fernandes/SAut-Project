import numpy as np
import math


class ExtendedKalmanFilter:
    def __init__(self, Rt, Qt, mu_0, sigma_0):
        # Mean and Covariance
        self.mu = mu_0
        self.sigma = sigma_0
        # Predicted mean and Cov.
        self.mu_bar = np.zeros_like(mu_0)
        self.sigma_bar = np.zeros_like(sigma_0)

        # State (R) and Measurement (Q) gaussian 
        # noise Covariance Matrices
        self.Rt = Rt
        self.Qt = Qt

        # State Variables
        #self.x = mu_0
        # Measurements
        #self.z = [0, 0]


    def g(self, x, u):
        # Sistem Model (Kinematics for ex)

        # Update Jacobian
        self.G = np.diag([2, 2, 0])

        return x + 2 #(constant speed)


    def h(self, x_bar):
        # Measurement/Sensor Model

        # Update Jacobian
        self.H = np.array([[1, 0, 0], [0, 1, 0]])

        return x_bar[0:2]


    def do_filter(self, ut, zt):
        """
        Kalman filter algorithm: given control u and 
        measurement z, updates px distribution
        """
        # Prediction Step
        self.mu_bar = self.g(self.mu, ut)
        self.sigma_bar = self.G @ self.sigma @ np.transpose(self.G) + self.Rt
        
        # Correction Step
        if zt is None:
            # If no measure is taken
            self.mu, self.sigma = self.mu_bar, self.sigma_bar
        else:
            # Innovation
            yt = zt - self.h(self.mu_bar)
            # Normal Update
            H_T = np.transpose(self.H)
            K = self.sigma_bar @ H_T @ np.linalg.inv(self.H @ self.sigma_bar @ H_T + self.Qt)
            self.mu = self.mu_bar + K @ yt
            self.sigma = (np.eye(3) - K @ self.H) @ self.sigma_bar
            
            # Save Innovation and Gains
            self.yt = yt
            self.K = K


def main():
    import matplotlib.pyplot as plt
    
    # Simulation Time
    MAX_TIME = 12

    # Probabilistic view
    mu0 = np.transpose([[2, 2, 0]]) # mu0 = np.zeros((3, 1))
    sigma0 = np.diag([1.0, 1.0, np.deg2rad(30.0)]) ** 2
    print("State and Cov dims: ", mu0.shape, sigma0.shape)

    # Covariance Matrices
    R = np.diag([0.5, 0.5, np.deg2rad(1.0)]) ** 2
    Q = np.diag([10.0, 10.0]) ** 2

    # Init. Kalman Filter
    EKF = ExtendedKalmanFilter(R, Q, mu0, sigma0)
    print("Rt: \n", EKF.Rt)
    print("Qt: \n", EKF.Qt, "\n")

    # Init. Actions and Measurements
    u = 0
    z = np.zeros((2, 1))

    # Robot moving in an environment
    real_position = []
    measurements = []
    pred = []
    for t in range(MAX_TIME):
        x = np.array([t*2, t*2], dtype=np.float32)
        z = np.array([x[0] + np.random.normal(0, 1.1), x[1] + np.random.normal(0, 1.1)])
        print("Real position: ", x.T)
        print("Measurement z: ", z.T)
        # Run Filter
        EKF.do_filter(u, z)
        print("Time:", t, " Position: (", EKF.mu[0], ",", EKF.mu[1], ")\n")
        # Collect for display later
        real_position.append(x)
        measurements.append(z)
        pred.append(EKF.mu[0])


    # Plot trajectory
    plt.figure()
    real_position = np.array(real_position)
    plt.plot(real_position[:, 0], real_position[:, 1], ".-.")
    measurements = np.array(measurements)
    plt.plot(measurements[:, 0], measurements[:, 1], ".-.")
    pred = np.array(pred)
    plt.plot(pred[:, 0], pred[:, 1], ".-.")
    plt.legend(["Real Position", "Measurements", "EKF Prediction"])
    
    # Plot Error
    plt.figure()
    plt.plot(np.linalg.norm(real_position - pred, axis=1))
    plt.xlabel("Time (s)")
    plt.ylabel("RMSE")
    
    # Show Graphics
    plt.show()


if __name__ == '__main__':
    main()
