from time import sleep
from myo import init, Hub, DeviceListener, StreamEmg
import numpy as np
import os
from sklearn.externals import joblib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
from drawnow import drawnow, figure

display_size = 100
window_size = 40
classify_interval = 20 # run classification every 20 data points

pca = joblib.load('models/pca.pkl')
clf = joblib.load('models/classifier.pkl')

classes = ['move', 'scratch', 'work', 'relax']
currentClassifyLabel = ''

running_acc_data = np.empty(shape=(0, 3))
running_gyro_data = np.empty(shape=(0, 3))
classifyCounter = 0

class Listener(DeviceListener):
    runningClusters = np.empty(shape=(0), dtype=int)
    runningData = np.empty(shape=(0, 8))

    def on_connect(self, myo, timestamp, firmware_version):
        myo.set_stream_emg(StreamEmg.enabled)

    def on_pair(self, myo, timestamp, firmware_version):
        print("Hello, Myo!")

    def on_unpair(self, myo, timestamp):
        print("Goodbye, Myo!")

    def on_gyroscope_data(self, event, timestamp, gy):
        global running_gyro_data
        #print ("on gy: ", gy)
        running_gyro_data = np.append(running_gyro_data, np.expand_dims(np.array(list(gy)), axis=0), axis=0)
        running_gyro_data = running_gyro_data[-1*100:, :]

    def on_accelerometor_data(self, event, timestamp, acc):
        global running_acc_data, running_gyro_data, classifyCounter, currentClassifyLabel, classes
        #print ("on acc: ", acc)
        running_acc_data = np.append(running_acc_data, np.expand_dims(np.array(list(acc)), axis=0), axis=0)
        running_acc_data = running_acc_data[-1*100:, :]

        classifyCounter += 1
        if running_acc_data.shape[0] >= window_size and running_gyro_data.shape[0] >= window_size and classifyCounter % classify_interval == 0:

            acc_data = running_acc_data[-1*window_size:,:]
            gyro_data = running_gyro_data[-1*window_size:,:]
            imu_data = np.append(acc_data, gyro_data, axis=1)

            print (imu_data)

            fft = np.abs(np.fft.fft(imu_data, axis=0))
            fft = fft.reshape(-1)

            reducedX = pca.transform(np.expand_dims(fft, axis=0))
            cls = clf.predict(reducedX)

            currentClassifyLabel = classes[int(cls[0])]

def update_line(lines, classifyLabel):
    global running_acc_data, running_gyro_data, currentClassifyLabel
    minSize = min(running_acc_data.shape[0], running_gyro_data.shape[0])
    imu_data = np.append(running_acc_data[:minSize], running_gyro_data[:minSize], axis=1)

    classifyLabel.set_text(currentClassifyLabel)
    if currentClassifyLabel == 'scratch':
        classifyLabel.set_bbox({'facecolor':'red', 'alpha':0.5, 'pad':10})
    else:
        classifyLabel.set_bbox({'facecolor':'green', 'alpha':0.5, 'pad':10})

    for i in range(6):
        d1 = np.transpose(imu_data[:, i])
        dd = np.append(np.expand_dims(np.array(range(imu_data.shape[0])), axis=0), np.expand_dims(d1, axis=0), axis=0)
        lines[i].set_data(dd)
    return lines[0], lines[1], lines[2], lines[3], lines[4], lines[5],

fig = plt.figure()
plt.rc('font', size=8)
lines = []
for i in range(6):
    ax = fig.add_subplot(7, 1, (i+2))
    ax.set_xlim([0, 100])
    if i < 3:
        ax.set_ylim([-2, 2])
    else:
        ax.set_ylim([-100, 100])
    l, = ax.plot([])
    lines.append(l)

classifyLabel = plt.text(30, 1500, '--', style='italic', fontsize=30, bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

plt.ion()
plt.show()

init()
hub = Hub()
hub.run(1000, Listener())
try:
    while True:
        sleep(0.01)
        update_line(lines, classifyLabel)
        plt.draw()
        plt.pause(0.001)
except KeyboardInterrupt:
    print('\nQuit')
finally:
    hub.shutdown()  # !! crucial
