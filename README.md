# Simultaneous Localization and Mapping on a free-flying space robot using EKF-SLAM :robot:

Project done for the Autonomous Systems course @ IST, Portugal.

This project consists of both localizing an Astrobee robot (3 of which are, at the time of writing this, aboard the ISS) as well as mapping its environment using a finite number of landmarks, through an Extended Kalman Filter.

## File Structure

Install required dependencies with `pip install -r requirements.txt` # NOT YET DONE #

### > Environment map definition (`map.py`)

### > Environment map definition (`map.py`)

### > Operate the EKF localization code

    - localization
        - localization.py
            - Performs localization on a given map using the EKF
        - ekf.py
            - Implements the Extended Kalman Filter (EKF)
        - registration.py
            - Matches two point-clouds using ICP

### > Misc. handy methods (`utils.py`)

    - utils.py
        - only contains functions used by the previous scripts.
