import numpy as np
import matplotlib.pyplot as plt

from utils import draw_cov_ellipse

def plot_environment_and_estimation(fig1, real_position, m, pred, cov, PLOT_ELLIPSES : bool):
    # Plot trajectory and true environment
    ax1 = fig1.add_subplot(121)
    plt.plot(real_position[:, 0], real_position[:, 1], ".-.")
    plt.scatter(m[:, 0], m[:, 1], color="tab:red", marker="d")

    # Plot Measurements
    #measurements = np.array(measurements)
    #plt.plot(measurements[:, 0], measurements[:, 1], ".", alpha=0.4)

    # Plot Predicted Position
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

def plot_error(ax2, time, real_position, pred):
    ax2.plot(
        time, np.linalg.norm(real_position[:, 0:2] - pred[:, 0:2], axis=1) 
        #+ 
        #np.sqrt((m[:, 0] - pred[:, 3::2])**2 + (m[:, 1] - pred[:, 4::2])**2)
        )
    plt.xlabel("Time (s)")
    plt.ylabel("RMSE")
    ax2.title.set_text("State Error (Pose + Landmarks)")

def plot_state(fig2, time, real_position, pred):
    ax3 = plt.gca()
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

