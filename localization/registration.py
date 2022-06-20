from simpleicp import PointCloud, SimpleICP
from dataclasses import fields
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


H = ([9.84803761e-01, -1.73670813e-01, -4.10710735e-05,  4.19684172e-04],
     [1.73670815e-01, 9.84803760e-01,  5.07245785e-05, -7.50290562e-04],
     [3.16375689e-05, -5.70866025e-05,  9.99999998e-01,  0],
     [0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  1.00000000e+00])


# Read point clouds from xyz files into n-by-3 numpy arrays
N = 20702
data_path = Path(__file__).parent
# X_fix = np.genfromtxt(data_path.joinpath(
#    Path("../data/bunny_part1.xyz")))
X_fix = np.genfromtxt(
    'data/bunny_part1.xyz')
X_mov = np.genfromtxt(
    'data/bunny_part2.xyz')

# X_fix = np.delete(X_fix,2,1)
# zeros = np.ones((1,N))*np.random.rand(1,N)*0.0001
# X_fix = np.vstack((X_fix.T, zeros))
# X_fix = X_fix.T


ones = np.ones((1, N))
P = np.vstack((X_fix.T, ones))
X_mov = np.delete(H @ P, 3, 0).T

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
H, X_mov_transformed, rigid_body_transformation_params = icp.run(
    max_overlap_distance=1)

print("H looks like this")
print(H)
print(H.shape)

P_linha = H @ P

# print(P_linha.shape)

P_linha = np.delete(P_linha, 3, 0)

# print(P_linha.T.shape)
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

# pc_moved = PointCloud(P_linha, columns=["x", "y", "z"])
# icd = SimpleICP()
# icd.add_point_clouds(pc_fix, pc_moved)
# H, X_mov_transformed, rigid_body_transformation_params = icd.run(max_overlap_distance=1)

# Display

ax = plt.axes(projection='3d')
plt.plot(P[0, :], P[1, :], P[2, :], '.')
plt.title("$P$")
plt.figure()
ax = plt.axes(projection='3d')
plt.plot(P_linha[0, :], P_linha[1, :], P_linha[2, :], '.')
plt.title("$P_linha$")
plt.show()
