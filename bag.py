import bagpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bagpy import bagreader

b = bagreader('data/bags/pose+twist2.bag')
print(b.topic_table)

pose = b.message_by_topic('/loc/pose')
twist = b.message_by_topic('/loc/twist')


pose_array = np.genfromtxt(pose, delimiter=',')
twist_array = np.genfromtxt(twist, delimiter=',')

print(pose_array.shape, twist_array.shape)

#pose_data = pd.read_csv(pose)
# print(pose_data)
#fig1, ax1 = bagpy.create_fig(1)
#ax1[0].scatter(x='Time', y='data', data=pose_data)
plt.figure(1)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(pose_array[:, 5], pose_array[:, 6])
plt.figure(2)
plt.subplot(311)
plt.ylabel("$V_x$")
plt.plot(twist_array[:, 0], twist_array[:, 5])
plt.subplot(312)
plt.ylabel("$V_y$")
plt.plot(twist_array[:, 0], twist_array[:, 6])
plt.subplot(313)
plt.ylabel("$\omega$")
plt.plot(twist_array[:, 0], twist_array[:, 10])
plt.show()

vel_data = pd.read_csv(twist)
# print(vel_data)
fig2, ax2 = bagpy.create_fig(1)
ax2[0].scatter(x='Time', y='data', data=vel_data)
plt.show()
