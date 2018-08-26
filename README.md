## Introduction

This is the accompanying source code for the below medium post

https://medium.com/softmind-engineering/real-time-activity-recognition-with-accelerometer-and-gyroscope-sensor-myo-663cda8536e6

## Caveats

Sorry the source code of this repository isn't very well organized. I didn't have too much time to tidy up the code to make it more generic to use. The purpose of this repo is mainly for reference for the above medium post, but you are welcome to modify and use the code to accomplish your tasks.

## Collect Data

collect_data.py is the sript to collect MYO data. 

1. update the line `activity = 'scratch'` to the acitivity you are about the carry out 

2. run python collect_data.py

3. press 'a' to start, then execute your action, then press 'b' to stop

4. the collected IMU and and EMG data will be stored in the file `data/ACTIVITY/imu1.txt` and `data/ACTIVITY/emg1.txt`

5. repeat pressing 'a' and 'b' to collect more trials of activities, and they will be stored in imu2.txt, imu3.txt, etc.

6. After finishing the trials of certain activity, update the line in step 1) again and repeat the steps


## Training

`train.ipynb` is jupyter notebook, providing sample training code using the collected data. It outputs `models/classifier.pkl` and `models/pca.pkl`. 

## Tracking

track.py provides a live activity tracking with the above trained model



