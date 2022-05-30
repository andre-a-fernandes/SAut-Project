from simpleicp import PointCloud, SimpleICP
from dataclasses import fields
import numpy as np
import matplotlib.pyplot as plt


H_real = ([ 0, -1, 0,  2],
 [ 1 , 0,  0, 4],
 [ 0, 0,  1,  0],
 [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00,  1.00000000e+00])

# Read point clouds from xyz files into n-by-3 numpy arrays
N = 100000
X_fix = np.genfromtxt("/home/costa/Downloads/SAut-Project-main/dragon1.xyz")
X_mov = np.genfromtxt("/home/costa/Downloads/SAut-Project-main/dragon2.xyz")

# X_fix = np.delete(X_fix,2,1)
# zeros = np.ones((1,N))*np.random.rand(1,N)*0.0001
# X_fix = np.vstack((X_fix.T, zeros))
# X_fix = X_fix.T



ones = np.ones((1,N))
P = np.vstack((X_fix.T, ones))
X_mov = np.delete(H_real @ P,3,0).T

print("X_fix looks like this")
print(X_fix)
print("X_mov looks like this")
print(X_mov)


# Create point cloud objects
pc_fix = PointCloud(X_fix, columns=["x", "y", "z"])
pc_mov = PointCloud(X_mov, columns=["x", "y", "z"])


# print(P.shape)
# print(P)

# Create simpleICP object, add point clouds, and run algorithm!
icp = SimpleICP()
icp.add_point_clouds(pc_fix, pc_mov)
H, X_mov_transformed, rigid_body_transformation_params = icp.run(max_overlap_distance=1)

print("H looks like this")
print(H)
print(H.shape)


print(np.linalg.norm(H-H_real))

P_linha = np.delete(H @ P,3,0)


print(P_linha.T)

print(
f"{'parameter':>9s} | "
f"{'est.value':>15s} | "
f"{'est.uncertainty':>15s} | "
f"{'obs.value':>15s} | "
f"{'obs.weight':>15s}"
)
for parameter in fields(rigid_body_transformation_params):
        print(
            f"{parameter.name:>9s} | "
            f"{getattr(rigid_body_transformation_params, parameter.name).estimated_value_scaled:15.6f} | "
            f"{getattr(rigid_body_transformation_params, parameter.name).estimated_uncertainty_scaled:15.6f} | "
            f"{getattr(rigid_body_transformation_params, parameter.name).observed_value_scaled:15.6f} | "
         f"{getattr(rigid_body_transformation_params, parameter.name).observation_weight:15.3e}"
         )



# Display
X_mov = X_mov.T
    
ax = plt.axes(projection='3d')
plt.plot(P[0, :], P[1, :], P[2, :], '.')
#plt.title("$P$")
plt.xlabel("x")
plt.ylabel("y")
# plt.figure()
# ax = plt.axes(projection='3d')
plt.plot(X_mov[0, :], X_mov[1, :], X_mov[2, :],'.')
plt.show()
# plt.title("$P_(linha)$")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.figure()
# ax = plt.axes(projection='3d')
# plt.plot(P_linha[0, :], P_linha[1, :], P_linha[2, :], 'r.')
# plt.title("$mov$")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()