import numpy as np
import math
import utils

PLOT_ELLIPSES = False
# Sensor Position relative to the Body-Frame
X_S = 0.16
Y_S = 0.16


class ExtendedKalmanFilter:
    def __init__(self, Rt, Qt, mu_0, sigma_0, dt, map=np.empty(0), TEST_DUMMY=True):
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
        # Simulation with Dummy data
        self.dummy = TEST_DUMMY

    def g(self, x, u):
        """
        Gives a prediction for the next robot state
        from the current one via a 'Constant Velocity
        Model'. Aditionally, it computes the Jacobian
        for the linearization at that point.

        ## Parameters

        x : current state (x, y, theta)
        u : input (V, w)
        """
        # Innertial Model (Constant Velocity)
        K = np.array([[math.cos(x[2]), 0],
                      [math.sin(x[2]), 0],
                      [0,              1]])
        x_dot = K @ u
        x_next = x + x_dot*self.dt

        # Update Jacobian G
        self.G = np.diag([1, 1, 0])

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
        z = np.zeros_like(x_bar)
        diffx = m[0] - x_bar[0]
        diffy = m[1] - x_bar[1]

        # Range-Bearing Model
        z[0] = np.linalg.norm(m - x_bar[0:2])
        z[1] = np.arctan2(m[1] - x_bar[1], m[0] - x_bar[0]) - x_bar[2]

        # Update Jacobian H
        self.H = np.array([[-diffx/z[0],        -diffy/z[0],    0],
                           [diffy/(z[0]**2), -diffx/(z[0]**2), -1]])

        return z[0:2]

    def update(self, zt, x_sensor, landmark=0, just_calc=False):
        """
        EKF update step

        ## Parameters


        ## Returns

        mu :

        sigma :

        """
        # Innovation and Kalman gain
        self.yt = zt - self.h(x_sensor, landmark)
        H_T = np.transpose(self.H)
        self.K = self.sigma_bar @ H_T @ np.linalg.inv(
                 self.H @ self.sigma_bar @ H_T + self.Qt)

        if just_calc:
            mu = self.mu_bar + self.K @ self.yt
            sigma = (np.eye(3) - self.K @ self.H) @ self.sigma_bar
            return mu, sigma
        else:
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
        if zt is None or zt.size == 0:
            # If no measure is taken
            self.mu, self.sigma = self.mu_bar, self.sigma_bar
            return
        else:
            # Sensor location in the body-frame
            x_bar = self.mu_bar
            ROT = np.array([[math.cos(x_bar[2]), -math.sin(x_bar[2]), 0],
                            [math.sin(x_bar[2]), math.cos(x_bar[2]),  0],
                            [0,                     0,                1]])
            x_sensor = x_bar + ROT @ [X_S, Y_S, 0]

            if self.dummy:
                self.update(zt, x_sensor)
                return
            
            # Choosing the best measurement w/ which to perform the update step
            min_id = 0
            min_trace = 1e6
            for l_measured in zt.T:
                #print("seen:", np.int(l_measured[2]))
                l_index = np.int(l_measured[2])
                _, sigma = self.update(l_measured[:2], x_sensor, self.map[l_index, :2], just_calc=True)
                if np.trace(sigma) < min_trace:
                    min_id = l_index
                    min_trace = np.trace(sigma)
            print("trace minimizer:", min_id)
            meas_id = np.asscalar(np.where(zt[2] == min_id)[0])
            #print(meas_id)
            self.update(zt[:2, meas_id], x_sensor, self.map[min_id, :2])

            # Look only at closest landmark detected (BEST RESULTS ???)
            #l_index = np.int(zt[2, 0])
            #self.update(zt[:2, 0], x_sensor, self.map[l_index, :2])
