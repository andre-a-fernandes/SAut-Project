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
        # List of observed features
        self.seen = []

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

        x : current state (x, y, theta, m_1, m_2, ..., m_N)
        u : input (V, w)
        """
        # Innertial Model (Constant Velocity)
        K = np.array([[math.cos(x[2]), 0],
                      [math.sin(x[2]), 0],
                      [0,              1]])
        x_dot = K @ u
        n2_landmarks = x.shape[0] - 3
        F_T = np.vstack((np.eye(3), np.zeros((n2_landmarks, 3))))
        # Update state estimate
        x_next = x + F_T @ x_dot*self.dt

        # Update Jacobian G
        self.G = np.eye(x.size)

        return x_next, F_T

    def predict(self, ut):
        """
        EKF prediction step
        """
        self.mu_bar, F_T = self.g(self.mu, ut)
        self.sigma_bar = self.G @ self.sigma @ np.transpose(self.G) + F_T @ self.Rt @ np.transpose(F_T)

    def h(self, mu_bar, j):
        """
        Gives a prediction for the measurement that
        would be taken given the state mean via the
        'Observation/Sensor Model'. Aditionally the
        Jacobian is calculated for linearization 
        purposes.

        ## Parameters

        mu_bar : current state prediction (x_bar, y_bar, 
        theta_bar, m_1, ..., m_N)

        m : landmark pose (x, y)

        ## Returns
        
        z : expected observation (r, phi)
        """
        # Use "Position Sensor" Model
        if self.dummy:
            self.H = np.array([[1, 0, 0],
                               [0, 1, 0]])
            return mu_bar[0:2]

        # Setup
        z = np.zeros(2)
        m = mu_bar[3 + j:5 + j]
        delta_x = m[0] - mu_bar[0]
        delta_y = m[1] - mu_bar[1]

        # Range-Bearing Model
        z[0] = np.linalg.norm(m - mu_bar[0:2])
        z[1] = np.arctan2(delta_y, delta_x) - mu_bar[2]

        # Update Jacobian H_j (w.r.t pose and landmark j)
        H_j = np.array([[-delta_x/z[0],        -delta_y/z[0],    0,     delta_x/z[0],       delta_y/z[0]],
                        [delta_y/(z[0]**2), -delta_x/(z[0]**2), -1, -delta_y/(z[0]**2), delta_x/(z[0]**2)]])
        
        # Map to the higher dim. state-space
        N = np.int((mu_bar.size - 3)/2)
        F_j = np.zeros((5, mu_bar.size))
        F_j[:3,:3] = np.eye(3)
        F_j[3:5, 3+j:5+j] = np.eye(2)
        """almost_eye1 = np.vstack((np.eye(3), np.zeros((2,3))))
        almost_eye2 = np.vstack((np.zeros((3,2)), np.eye(2)))
        if j > 1:
            F_j = np.hstack((almost_eye1, np.zeros((5, 2*j-2)), almost_eye2, np.zeros((5, 2*N-2*j))))
        else:
            F_j = np.hstack((almost_eye1, almost_eye2, np.zeros((5, 2*N-2*j))))"""
        print(F_j.shape)

        # Update real Jacobian
        self.H = H_j @ F_j
        return z

    def update(self, zt, x_sensor, just_calc=False):
        """
        EKF update step
        """
        # Innovation and Kalman gain
        self.yt = zt[:2] - self.h(self.mu_bar, np.int(zt[2])+1)
        H_T = np.transpose(self.H)
        self.K = self.sigma_bar @ H_T @ np.linalg.inv(
                 self.H @ self.sigma_bar @ H_T + self.Qt)
        dim = self.mu_bar.size

        if just_calc:
            mu = self.mu_bar + self.K @ self.yt
            sigma = (np.eye(dim) - self.K @ self.H) @ self.sigma_bar
            return mu, sigma
        else:
            # Update
            self.mu_bar = self.mu_bar + self.K @ self.yt
            self.sigma_bar = (np.eye(dim) - self.K @ self.H) @ self.sigma_bar 

    def do_filter(self, ut, zt, verbose=False):
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
            """
            # Sensor location in the body-frame
            x_bar = self.mu_bar[:3]
            ROT = np.array([[math.cos(x_bar[2]), -math.sin(x_bar[2]), 0],
                            [math.sin(x_bar[2]), math.cos(x_bar[2]),  0],
                            [0,                     0,                1]])
            x_sensor = x_bar + ROT @ [X_S, Y_S, 0]
            """
            x_sensor = self.mu_bar[:3]
            if self.dummy:
                self.update(zt, x_sensor)
                return
            
            # For all observed features/landmarks
            for l_measured in zt.T:
                l_index = np.int(l_measured[2])
                # Initialize landmark if not yet seen
                if l_index not in self.seen:
                    self.mu_bar[3+l_index] = x_sensor[0] + l_measured[0]*np.cos(l_measured[1] + x_sensor[2])
                    self.mu_bar[4+l_index] = x_sensor[1] + l_measured[0]*np.sin(l_measured[1] + x_sensor[2])
                    self.seen.append(l_index)
                # Update estimates as normal
                self.update(l_measured, self.mu_bar)

            # Correct for sensor offset w.r.t body-frame
            """ROT = np.array([[math.cos(self.mu[2]), -math.sin(self.mu[2]), 0],
                            [math.sin(self.mu[2]), math.cos(self.mu[2]),  0],
                            [0,                     0,                1]])
            self.mu[:3] = ROT @ self.mu_bar[:3]
            self.mu = self.mu_bar - np.concatenate(([X_S, Y_S], np.zeros(self.mu.size-2)))"""
            self.mu = self.mu_bar
            self.sigma = self.sigma_bar

            # Look only at closest landmark detected (BEST RESULTS ???)
            #l_index = np.int(zt[2, 0])
            #self.update(zt[:2, 0], x_sensor, self.map[l_index, :2])
