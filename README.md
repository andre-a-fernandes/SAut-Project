# Simultaneous Localization and Mapping on a free-flying space robot using EKF-SLAM :robot:

Project done for the Autonomous Systems course @ IST, Portugal.

This project consists of both localizing an Astrobee robot (3 of which are, at the time of writing this, aboard the ISS) as well as mapping its environment using a finite number of landmarks, through an Extended Kalman Filter.

## File Structure

Install required dependencies with `pip install -r requirements.txt` # NOT YET DONE #

< Operate the EKF localization code >
    - localization
        - imgs.py
            - Removes outliers and performs PCA
            - K-Means clustering 

### > Extended Kalman Filter (`ekf.py`)

### > Environment map definition (`map.py`)

### > Misc. handy methods (`utils.py`)

The remaining files only contain functions used by the previous scripts.
