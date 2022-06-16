import numpy as np
import math

PLOT_ELLIPSES = False
# Sensor Position relative to the Body-Frame
X_S = 0.16
Y_S = 0.16
# Mahalanobis distance parameter
ALPHA = 1.5


class ExtendedKalmanFilter:
    def __init__(self, Rt, Qt, mu_0, sigma_0, dt, map=np.empty(0)):
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
        self.Nt = 0

        # Discretization
        self.dt = dt

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
        Gt = np.array([[1, 0, -u[0]*math.sin(x[2])],
                       [0, 1,  u[0]*math.cos(x[2])],
                       [0, 0, 1]])
        self.G = np.eye(x.size)
        self.G[:3,:3] = Gt
        #self.G = np.eye(x.size) + F_T @ Gt @ np.transpose(F_T)

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
        # Setup
        z = np.zeros(2)
        m = mu_bar[3 + 2*(j-1):5 + 2*(j-1)]
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
        F_j[3:5, 3+2*(j-1):5+2*(j-1)] = np.eye(2)
        
        #almost_eye1 = np.vstack((np.eye(3), np.zeros((2,3))))
        #almost_eye2 = np.vstack((np.zeros((3,2)), np.eye(2)))
        #if j > 1:
        #    F_j = np.hstack((almost_eye1, np.zeros((5, 2*j-2)), almost_eye2, np.zeros((5, 2*N-2*j))))
        #else:
        #    F_j = np.hstack((almost_eye1, almost_eye2, np.zeros((5, 2*N-2*j))))
        #print("F_j", F_j)

        # Update real Jacobian
        H = H_j @ F_j
        return z , H

    def update(self, x_sensor, zt, verbose : bool):
        """
        EKF update step
        """
        # Setup index of next observed landmark
        N_tplus1 = self.Nt + 1
        # Setup sum arrays/matrices
        dim = self.mu_bar.size
        sumKY = np.zeros_like(self.mu_bar)
        sumKH = np.zeros((dim, dim))
        
        # For all observed features/landmarks
        i = 0
        j = []
        pi = []
        Psi = []
        self.H = []
        K_t = []
        for l_obs in zt.T:
            self.mu_bar[3+2*N_tplus1] = x_sensor[0] + l_obs[0]*np.cos(l_obs[1] + x_sensor[2])
            self.mu_bar[4+2*N_tplus1] = x_sensor[1] + l_obs[0]*np.sin(l_obs[1] + x_sensor[2])
            # Has the landmark been seen ?
            for k in range(0, N_tplus1):              
                # Innovation and Kalman gain
                z_k, H_tk = self.h(self.mu_bar, k)
                yt = l_obs[:2] - z_k
                self.H.append(H_tk)
                Psi.append(self.H[k] @ self.sigma_bar @ self.H[k].T + self.Qt)
                pi.append(np.transpose(yt) @ np.linalg.inv(Psi[k]) @ yt)
                H = np.array(self.H)

            # Actually check for distance to others
            pi.append(ALPHA)
            j.append(np.argmin(pi))
            if verbose:
                print(i)
            self.Nt = max(self.Nt, j[i])
            print("Landmarks seen:", self.Nt)
            K_t.append(self.sigma_bar @ np.transpose(self.H[j[i]-1]) @ Psi[j[i]-1])

            # Calculate auxiliary sums
            z_ji, _ = self.h(self.mu_bar, j[i]-1)
            sumKY += K_t[i] @ (l_obs[:2] - z_ji)
            sumKH += K_t[i] @ self.H[j[i]-1]
            i += 1
        # Update
        self.mu = self.mu_bar + sumKY
        self.sigma = (np.eye(dim) - sumKH) @ self.sigma_bar

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
        else:            
            x_sensor = self.mu_bar[:3]
            self.update(x_sensor, zt, verbose)