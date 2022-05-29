import numpy as np
import math
import utils

PLOT_ELLIPSES = False
# Sensor Position relative to the Body-Frame
X_S = 0
Y_S = 0


class ExtendedKalmanFilter:
    def __init__(self, Rt, Qt, mu_0, sigma_0, dt, map=0, TEST_DUMMY=True, INNERTIAL=True):
        """
        Filter initialization method.

        Rt:     (n x n)
        Qt:     (m x n)

        mu_0:    mean state initial guess
        sigma_0: covariance matrix for state guess

        dt: time interval used for localiz. cycle
        """
        # Mean and Covariance
        self.mu = mu_0
        self.sigma = sigma_0
        # Predicted mean and Cov.
        self.mu_bar = np.zeros_like(mu_0)
        self.sigma_bar = np.zeros_like(sigma_0)

        # State (R) and Measurement (Q)
        # gaussian noise Covariance Matrices
        self.Rt = Rt
        self.Qt = Qt

        # Landmark Map
        self.map = map

        # Discretization
        self.dt = dt
        # Innertial Nav. flag
        self.innertial = INNERTIAL
        # Simulation with Dummy data
        self.dummy = TEST_DUMMY

    def g(self, x, u):
        """
        Gives a prediction for the next robot state
        from the current one via the 'Odometry Motion
        Model'. Aditionally, it computes the Jacobian
        for the linearization at that point.

        x: current state (x, y, theta)
        u: input (V, w) - LEGACY: 'action' input (rot_1, transl, rot_2)
        """
        # Innertial Model (Constant Velocity)
        if self.innertial:
            # Kinematics
            # K = np.array([[math.cos(x[2]), 0],
            #              [math.sin(x[2]), 0],
            #              [0,              1]])
            #x_dot = np.reshape(K @ u, (3, 1))
            #x_dot = np.zeros_like(x)
            #x_dot[0] = u[0]*math.cos(x[2])
            #x_dot[1] = u[0]*math.sin(x[2])
            #x_dot[2] = u[1]
            #x_next = x + x_dot*self.dt

            # Update Jacobian G
            self.G = np.diag([1, 1, 0])

            return x + math.sqrt(2)*self.dt

        if not self.innertial:
            x_next = np.zeros_like(x)

            # Odometry Motion Model
            x_next[0] = x[0] + u[1]*math.cos(x[2] + u[0])
            x_next[1] = x[1] + u[1]*math.sin(x[2] + u[0])
            x_next[2] = x[2] + u[0] + u[2]

            # Update Jacobian G
            self.G = np.array([[1, 0, -u[1]*math.sin(x[2] + u[0])],
                               [0, 1, -u[1]*math.cos(x[2] + u[0])],
                               [0, 0, 1]])

            return x_next

    def predict(self, ut):
        """
        EKF prediction step
        """
        self.mu_bar = self.g(self.mu, ut)
        self.sigma_bar = self.G @ self.sigma @ np.transpose(self.G) + self.Rt

    def h(self, x_bar, m):
        """
        Gives a prediction for the measurement that
        would be taken given the state mean via the
        'Observation/Sensor Model'. Aditionally the
        Jacobian is calculated for linearization 
        purposes.

        x_bar: current state prediction (x_bar, y_bar, theta_bar)
        m: landmark pose (x, y)

        Output -> z: observation (r, phi) for now without z[2]: index
        """
        # Use Different Sensor Model
        if self.dummy:
            self.H = np.array([[1, 0, 0],
                               [0, 1, 0]])
            return x_bar[0:2]

        # Setup
        m = np.reshape(m, (2, 1))
        z = np.zeros_like(x_bar)
        diffx = m[0] - x_bar[0]
        diffy = m[1] - x_bar[1]

        # Range-Bearing Model
        z[0] = np.linalg.norm(m - x_bar[0:2])
        z[1] = np.arctan2(m[1] - x_bar[1], m[0] - x_bar[0]) - x_bar[2]

        # Update Jacobian H
        a = np.asscalar(-diffx/z[0])
        b = np.asscalar(-diffy/z[0])
        c = np.asscalar(diffy/(z[0]**2))
        d = np.asscalar(-diffx/(z[0]**2))
        self.H = np.array([[a, b, 0],
                           [c, d, -1]])

        return z[0:2]

    def update(self, zt, x_sensor, landmark=0, just_calc=False):
        """
        EKF update step
        """
        # Innovation and Kalman gain
        self.yt = zt - self.h(x_sensor, landmark)
        H_T = np.transpose(self.H)
        self.K = self.sigma_bar @ H_T @ np.linalg.inv(
            self.H @ self.sigma_bar @ H_T + self.Qt)

        if not just_calc:
            # Update
            self.mu = self.mu_bar + self.K @ self.yt
            self.sigma = (np.eye(3) - self.K @ self.H) @ self.sigma_bar

    def do_filter(self, ut, zt):
        """
        Kalman filter algorithm: given control u and
        measurement z, updates P(x) distribution
        """
        # Prediction Step
        self.predict(ut)

        # Correction Step
        if zt is None:
            # If no measure is taken
            self.mu, self.sigma = self.mu_bar, self.sigma_bar
            return
        else:
            # Sensor location in body
            x_bar = self.mu_bar
            # ROT = np.array([[math.cos(x_bar[2]), -math.sin(x_bar[2]), 0],
            #                [math.sin(x_bar[2]), math.cos(x_bar[2]),  0],
            #                [0,                 0,                1]])
            x_sensor = x_bar  # + ROT @ [X_S, Y_S, zt[1]]

            # for landmark in self.map:
            #    self.update(zt, x_bar, True)
            #    min(np.trace((np.eye(3) - self.K @ self.H) @ self.sigma_bar))
            if self.dummy:
                self.update(zt, x_sensor)
                return
            self.update(zt, x_sensor, self.map[1])


def main():
    import matplotlib.pyplot as plt

    # Simulation Time
    MAX_TIME = 12
    dt = 0.2

    # Probabilistic view
    mu0 = np.transpose([[0, 0, np.deg2rad(45.0)]])
    sigma0 = np.diag([1.5, 1.5, np.deg2rad(20.0)]) ** 2
    print("State and Cov dims: ", mu0.shape, sigma0.shape)

    # Covariance Matrices
    R = np.diag([0.1, 0.1, np.deg2rad(10.0)]) ** 2
    Q = np.diag([1.3, 1.1]) ** 2

    # Init. Kalman Filter
    EKF = ExtendedKalmanFilter(R, Q, mu0, sigma0, dt)
    print("Rt: \n", EKF.Rt)
    print("Qt: \n", EKF.Qt, "\n")

    # Init. Actions and Measurements
    u = np.array([2, 0])
    z = np.zeros((2, 1))

    # Robot moving in an environment
    real_position = []
    measurements = []
    pred = []
    cov = []
    for timestep in range(int(MAX_TIME/dt)):
        x = np.array([dt*timestep*math.sqrt(2), dt*timestep*math.sqrt(2)])
        z = np.array([x[0] + np.random.normal(0, 2.0),
                     x[1] + np.random.normal(0, 1.1)])
        print("Real position: ", x.T)
        print("Measurement z: ", z.T)
        # Run Filter
        EKF.do_filter(u, z)
        print(EKF.mu.shape)
        print("Time:", dt*timestep, " Position: (",
              EKF.mu[0], ",", EKF.mu[1], ")\n")
        # Collect for display later
        real_position.append(x)
        measurements.append(z)
        pred.append(EKF.mu[0])
        cov.append(EKF.sigma)

    # Plot trajectory
    plt.figure()
    plt.subplot(121)
    ax = plt.gca()
    real_position = np.array(real_position)
    plt.plot(real_position[:, 0], real_position[:, 1], ".-.")

    # Plot Measurements
    measurements = np.array(measurements)
    plt.plot(measurements[:, 0], measurements[:, 1], ".")  # -.")

    # Plot Predicted Position
    pred = np.array(pred)
    plt.plot(pred[:, 0], pred[:, 1], ".")  # -.")
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
    main()
