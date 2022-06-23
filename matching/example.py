import numpy as np
import matplotlib.pyplot as plt

from icp import icp

import sys
sys.path.append('./')  
from map import choose_landmark_map

if __name__ == '__main__':
    plt.ion()
    
    # set seed for reproducible results
    np.random.seed(12345)

    # create a set of points to be the reference for ICP
    M, n = choose_landmark_map("iss_L")
    reference_points = M[:, :2]
    print(reference_points.shape)
    plt.plot(reference_points[:, 0], reference_points[:, 1], 'rs', label='reference points')

    # transform the set of reference points to create a new set of
    # points for testing the ICP implementation
    points_to_be_aligned = reference_points.copy()
    points_trans = reference_points.copy()

    # 2. apply translation to the new point set
    points_trans += np.array([-9.8, 9.8])
    plt.plot(points_trans[:, 0], points_trans[:, 1], 'y1', label='translated points')

    # 3. apply rotation to the new point set
    theta = np.deg2rad(0)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s],
                    [s, c]])
    points_to_be_aligned = points_trans @ rot
    plt.plot(points_to_be_aligned[:, 0], points_to_be_aligned[:, 1], 'bx', label='points to be aligned')

    # run icp
    transformation_history, aligned_points = icp(reference_points=reference_points, 
    points=points_to_be_aligned, distance_threshold=12, convergence_rotation_threshold=1e-20, 
    convergence_translation_threshold=1e-20, verbose=True)
    # 2nd icp
    transformation_history, aligned_points = icp(reference_points=reference_points,
     points=aligned_points, distance_threshold=2, verbose=True)

    # show results
    plt.plot(aligned_points[:, 0], aligned_points[:, 1], 'g+', label='aligned points')
    plt.legend()
    plt.show(block=True)
