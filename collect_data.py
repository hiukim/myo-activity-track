from time import sleep
from myo import init, Hub, DeviceListener, StreamEmg
import os
import signal
import numpy as np
import keyboard

act_counter = 1
isRecording = False
acc_data = np.empty(shape=(0, 3))
gyro_data = np.empty(shape=(0, 3))
ori_data = np.empty(shape=(0, 4))
emg_data = np.empty(shape=(0, 8))

data_folder = 'data'
activity = 'scratch'

class Listener(DeviceListener):

    def on_connect(self, myo, timestamp, firmware_version):
        myo.set_stream_emg(StreamEmg.enabled)

    def on_pair(self, myo, timestamp, firmware_version):
        print("Hello, Myo!")

    def on_unpair(self, myo, timestamp):
        print("Goodbye, Myo!")

    def on_orientation_data(self, myo, timestamp, quat):
        global isRecording, ori_data
        if not isRecording:
            return

        print("Orientation:", quat.x, quat.y, quat.z, quat.w)
        ori_data = np.append(ori_data, np.expand_dims(np.array([quat.x, quat.y, quat.z, quat.w]), axis=0), axis=0)

    def on_gyroscope_data(self, event, timestamp, gy):
        global isRecording
        if not isRecording:
            return

        global gyro_data
        print ("on gy: ", gy)
        gyro_data = np.append(gyro_data, np.expand_dims(np.array(list(gy)), axis=0), axis=0)

    def on_accelerometor_data(self, event, timestamp, acc):
        global isRecording
        if not isRecording:
            return

        global acc_data
        print ("on acc: ", acc)
        acc_data = np.append(acc_data, np.expand_dims(np.array(list(acc)), axis=0), axis=0)

    def on_emg_data(self, myo, timestamp, data):
        global isRecording
        if not isRecording:
            return

        global emg_data
        print("EMG", data)
        emg_data = np.append(emg_data, np.expand_dims(np.array(list(data)), axis=0), axis=0)

init()
hub = Hub()
hub.run(1000, Listener())

def saveData():
    global act_counter, acc_data, gyro_data, ori_data, emg_data, activity

    imu_data = np.append(np.append(acc_data, gyro_data, axis=1), ori_data, axis=1)
    np.savetxt(data_folder + '/' + activity + '/imu' + str(act_counter) + '.txt', imu_data, delimiter=',')
    np.savetxt(data_folder + '/' + activity + '/emg' + str(act_counter) + '.txt', emg_data, delimiter=',')

    acc_data = np.empty(shape=(0, 3))
    gyro_data = np.empty(shape=(0, 3))
    ori_data = np.empty(shape=(0, 4))
    emg_data = np.empty(shape=(0, 8))
    act_counter += 1

def key_press(key):
    global isRecording
    if key.name == 'a':
        print ('start recording...')
        isRecording = True
    elif key.name == 'b':
        print ('stop recording...')
        isRecording = False
        saveData()

keyboard.on_press(key_press)

try:
    while True:
        sleep(0.5)
except KeyboardInterrupt:
    print('\nQuit')
finally:
    hub.shutdown()  # !! crucial
