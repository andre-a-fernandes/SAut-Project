# Simultaneous Localization and Mapping on a free-flying space robot using EKF-SLAM :robot:

Project done for the Autonomous Systems course @ IST, Portugal.

This project consists of both localizing an Astrobee robot (3 of which are, at the time of writing this, aboard the ISS) as well as mapping its environment using a finite number of landmarks, through an Extended Kalman Filter.

## File Structure

Install required dependencies with `pip install -r requirements.txt` # NOT YET DONE #

    - data
        - bags
            Raw data obtained through rosbag recordings in the Astrobee simulator.
        - processed
            Already-processed data into Numpy array format.
        - bag.py
    - main.py
        Performs EKF-SLAM and outputs plots for analysis/comparison.
    - ekf_slam.py
    - ekf_unknown_correpondences.py
        Both implement the SLAM algorithm, with the latter dealing with data association
        via a Maximum Likelihood estimator.
    - utils.py
    
Full breakdown:

### > Python EKF-SLAM implementation

    - main.py
        - Performs EKF-SLAM and outputs plots of real trajectory and landmark location vs. estimation
    - ekf_slam.py
        - Implements the EKF-SLAM with known data association
    - ekf_unknown_correpondences.py
        - Implements the EKF-SLAM with data association via a Maximum Likelihood estimator

### > Environment map definition (`map.py`)

### > Collected robot data

    - data
        - bags
            - Raw data obtained via rosbag recordings in the Astrobee simulator.
        - processed
            - Already-processed data into numpy array format.
    - bag.py
        - reads rosbags collected in the robot

### > Operate the EKF localization code

    - localization
        - localization.py
            - Performs localization on a given map using the EKF
        - ekf.py
            - Implements the Extended Kalman Filter (EKF)
        - registration.py
            - Matches two point-clouds using ICP

### > Misc. handy methods (`utils.py` and `plot.py`)
