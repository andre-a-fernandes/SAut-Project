import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bagpy import bagreader
from utils import euler_from_quaternion, create_pointcloud, update_3Dplot, update_2Dplot
import rosbag
import matplotlib.animation as animation

# OPTIONS:
# 0 - POSE
# 1 - IMU
# 2 - MAP
# 3 - SCAN
OPT = 3

# Read Rosbag
if OPT == 0:
    b = bagreader('data/bags/2truth+EKF.bag')
if OPT == 1:
    b = bagreader('data/bags/2imu.bag')
if OPT == 2:
    b = bagreader('data/bags/navCam.bag')
if OPT == 3:
    bag = rosbag.Bag('data/bags/2hazCam.bag')

# See the topics and save them
#print(b.topic_table)
if OPT == 0:
    pose = b.message_by_topic('/loc/pose')
    twist = b.message_by_topic('/loc/twist')
    pose_truth = b.message_by_topic('/loc/truth/pose')
    twist_truth = b.message_by_topic('/loc/truth/twist')
if OPT == 1:
    imu = b.message_by_topic('/hw/imu')
if OPT == 2:
    #navCam = b.message_by_topic('/hw/cam_nav')
    hazCam = b.message_by_topic('/hw/depth_haz/points')
if OPT == 3:
    cloud = []
    idx = 0
    for topic, msg, t in bag.read_messages():
        # Get only 1 Hz of data (1/5)
        if (idx % 5 == 0):
            print(idx, topic, t)
            # Read pointcloud in the message
            points_in_msg = create_pointcloud(msg)
            print("\n")
            cloud.append(points_in_msg)
        idx += 1


# Create numpy arrays for the data
if OPT == 0:
    pose_array = np.genfromtxt(pose, delimiter=',')
    twist_array = np.genfromtxt(twist, delimiter=',')
    true_pose_array = np.genfromtxt(pose_truth, delimiter=',')
    true_twist_array = np.genfromtxt(twist_truth, delimiter=',')
    #print(pose_array.shape, twist_array.shape)
    print(true_pose_array.shape, true_twist_array.shape)
if OPT == 1:
    imu_array = np.genfromtxt(imu, delimiter=',')
    print(imu_array.shape)
if OPT == 2:
    #xyz_array = point_cloud2.read_points_list(b)
    #hazcam_array = np.genfromtxt(hazCam, delimiter=',')
    hazcam_array = pd.read_csv(hazCam).to_numpy()
    print(hazcam_array.shape)
if OPT == 3:
    cloud_array = np.array(cloud)
    print(cloud_array.shape)


## DISPLAY ##
if OPT == 0:
    #pose_data = pd.read_csv(pose)
    # print(pose_data)
    #fig1, ax1 = bagpy.create_fig(1)
    #ax1[0].scatter(x='Time', y='data', data=pose_data)
    plt.figure(1)
    plt.title("Position in the World Frame")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(pose_array[:, 5], pose_array[:, 6])
    plt.plot(true_pose_array[:, 5], true_pose_array[:, 6])

    # State Vars
    plt.figure(2)
    plt.subplot(311)
    plt.title("State Variables (Estimates vs. Truth)")
    plt.ylabel("x")
    plt.plot(pose_array[:, 0], pose_array[:, 5])
    plt.plot(true_pose_array[:, 0], true_pose_array[:, 5])
    plt.subplot(312)
    plt.ylabel("y")
    plt.plot(pose_array[:, 0], pose_array[:, 6])
    plt.plot(true_pose_array[:, 0], true_pose_array[:, 6])
    plt.subplot(313)
    plt.ylabel("$\Theta$")
    yaw = []
    true_yaw = []
    for i in range(pose_array[:, 8].shape[0]):
        _, _, angle = euler_from_quaternion(
            pose_array[i, 8], pose_array[i, 9], pose_array[i, 10], pose_array[i, 11])
        yaw.append(angle)
    for i in range(true_pose_array[:, 8].shape[0]):
        _, _, true_angle = euler_from_quaternion(
            true_pose_array[i, 8], true_pose_array[i, 9], true_pose_array[i, 10], true_pose_array[i, 11])
        true_yaw.append(true_angle)
    plt.plot(pose_array[:, 0], np.degrees(yaw))
    plt.plot(true_pose_array[:, 0], np.degrees(true_yaw))
    plt.xlabel("Time")

    # Velocities (IMU)
    plt.figure(3)
    plt.subplot(311)
    plt.ylabel("$V_x$")
    plt.plot(twist_array[:, 0], twist_array[:, 5])
    plt.plot(true_twist_array[:, 0], true_twist_array[:, 5])
    plt.subplot(312)
    plt.ylabel("$V_y$")
    plt.plot(twist_array[:, 0], twist_array[:, 6])
    plt.plot(true_twist_array[:, 0], true_twist_array[:, 6])
    plt.subplot(313)
    plt.ylabel("$\omega$")
    plt.plot(true_twist_array[:, 0], true_twist_array[:, 10])
    plt.plot(true_twist_array[:, 0], true_twist_array[:, 10])
    plt.xlabel("Time")

    # Save data
    true_pose_all = np.vstack(
        (true_pose_array[:, 5], true_pose_array[:, 6], true_yaw))
    np.save("pose3D", true_pose_all.T)
    np.save("twist3D", true_twist_array[:, [5, 6, 10]])
    print((true_pose_all.T).shape)
    print(true_twist_array[:, [5, 6, 10]].shape)

if OPT == 1:

    # IMU Vars
    plt.figure(3)
    plt.subplot(311)
    plt.title("Measured Angles (IMU)")
    yaw_imu = []
    pitch_imu = []
    roll_imu = []
    for i in range(imu_array[:, 0].shape[0]):
        xangle, yangle, zangle = euler_from_quaternion(
            imu_array[i, 5], imu_array[i, 6], imu_array[i, 7], imu_array[i, 8])
        roll_imu.append(xangle)
        pitch_imu.append(yangle)
        yaw_imu.append(zangle)
    plt.ylabel("roll")
    plt.plot(imu_array[:, 0], np.degrees(roll_imu))
    plt.subplot(312)
    plt.ylabel("pitch")
    plt.plot(imu_array[:, 0], np.degrees(pitch_imu))
    plt.subplot(313)
    plt.ylabel("yaw ($\Theta$)")
    plt.plot(imu_array[:, 0], np.degrees(yaw_imu))
    plt.xlabel("Time")

    # Angular velocity measured
    plt.figure(4)
    plt.subplot(311)
    plt.title("Angular velocity (IMU)")
    plt.ylabel("$\omega_x$")
    plt.plot(imu_array[:, 0], imu_array[:, 18])
    plt.subplot(312)
    plt.ylabel("$\omega_y$")
    plt.plot(imu_array[:, 0], imu_array[:, 19])
    plt.subplot(313)
    plt.ylabel("$\omega_z = \omega$")
    plt.plot(imu_array[:, 0], imu_array[:, 20])
    plt.xlabel("Time")

    # Acceleration (CHECK CONST. VEL. HIPOTHESIS)
    plt.figure(5)
    plt.subplot(311)
    plt.title("Linear Acceleration (IMU)")
    plt.ylabel("$a_x$")
    plt.plot(imu_array[:, 0], imu_array[:, 30])
    plt.subplot(312)
    plt.ylabel("$a_y$")
    plt.plot(imu_array[:, 0], imu_array[:, 31])
    plt.subplot(313)
    plt.ylabel("$a_z$")
    plt.plot(imu_array[:, 0], imu_array[:, 32])
    plt.xlabel("Time")

    # Save data
    vel_all = np.vstack((yaw_imu, imu_array[:, 20]))
    print((vel_all.T).shape)
    np.save("theta+w", vel_all.T)

if OPT == 3:
    #"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scanplot, = ax.plot([], [], '.')
    #"""
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scanplot, = ax.plot([], [], [], '.')
    ax.view_init(elev=20, azim=-178)
    """
    plt.title("Point Cloud Scans")
    plt.xlabel("x")  # plt.xlim(-2.5, 2.5)
    plt.ylabel("y")  # plt.ylim(-2.5, 2.5)

    #"""
    # Flatten Point Cloud Data
    fixedZ_total = []
    N = 30000
    # For each sensor image (cloud)
    for i in range(cloud_array.shape[0]):
        point_list = []
        zero_count = 0
        # For every point in the cloud
        for t in range(cloud_array.shape[1]):
            p = cloud_array[i, t, :]
            if p[1] > -0.2 and p[1] < 0.2:
                point_list.append(p)
                zero_count += 1
        # Save fixed-z pointcloud
        point_list = np.array(point_list)
        #print(point_list.shape)
        point_list = np.vstack((point_list.reshape((zero_count, 4)), np.zeros((N-zero_count,4))))
        #print(point_list.shape)
        fixedZ_total.append(point_list)
    fixedZ_total = np.array(fixedZ_total)
    print(fixedZ_total.shape)
    #"""

    """# Plot 2D "Maps"
    for i in range(fixedZ_total.shape[0]):
        plt.figure()
        ax = plt.axes()#projection='3d')
        ax.plot(fixedZ_total[i, :, 2], fixedZ_total[i, :,0], '.')
        plt.show()"""
    
    # ANIMATION
    """ 3D:
    ani = animation.FuncAnimation(fig, update_3Dplot, fargs=(
        scanplot, cloud_array), frames=cloud_array.shape[0], interval=100, blit=True)
    ani.save('abel.gif')
    """
    #"""
    ani = animation.FuncAnimation(fig, update_2Dplot, fargs=(
        scanplot, fixedZ_total), frames=fixedZ_total.shape[0], interval=100, blit=True)
    ani.save('abel2d.gif')
    #"""

plt.show()
