# Capstone_2018
Eye tracking control Drone

Pupil Location -> CSV Feature
- The eyeris_detector.py can now generate a CSV file (np.csv) with left/right pupil W/H values
- To ensure the quality of data, please delete the "old" files, since the new data will just append at the end
- The data extracted will skip the blinking false data [0 0 0 0], however it may still record the data when only 1 eye is detected
- Need to add in filter to eliminate the 0 values when analysis results
