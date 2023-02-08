import os
import sys
print("Hello World!")
print(os.getcwd())

#针对Visual Studio Code，如果工作目录是pythonTest
sys.path.append("../Kalman-and-Bayesian-Filters-in-Python")
sys.path.append("../filterpy")

'''
#针对Visual Studio Code，如果工作目录是kalman
sys.path.append("../../Kalman-and-Bayesian-Filters-in-Python")
sys.path.append("../../filterpy")
'''
from numpy.random import randn

class PosSensor(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]
        
    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        
        return [self.pos[0] + randn() * self.noise_std,
                self.pos[1] + randn() * self.noise_std]

import matplotlib.pyplot as plt
import numpy as np

from kf_book.book_plots import plot_measurements

pos, vel = (4, 3), (2, 1)
sensor = PosSensor(pos, vel, noise_std=1)
ps = np.array([sensor.read() for _ in range(50)])

#plot_measurements(ps[:, 0], ps[:, 1]);
#plt.show();
#print(ps[:,0])

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from kf_book.book_plots import plot_filter

#implement the filter
from filterpy.stats import plot_covariance_ellipse
from scipy.linalg import block_diag

R_std = 0.35
Q_std = 0.04

def tracker1():
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0   # time step

    tracker.F = np.array([[1, dt, 0,  0],
                          [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]])
    tracker.u = 0.
    tracker.H = np.array([[1/0.3048, 0, 0, 0],
                          [0, 0, 1/0.3048, 0]])

    tracker.R = np.eye(2) * R_std**2
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
    tracker.Q = block_diag(q, q)
    tracker.x = np.array([[0, 0, 0, 0]]).T
    tracker.P = np.eye(4) * 500.
    return tracker

# simulate robot movement
N = 30
sensor = PosSensor((0, 0), (2, .2), noise_std=R_std)

zs = np.array([sensor.read() for _ in range(N)])

# run filter
robot_tracker = tracker1()
mu=np.zeros((N, 4, 1))

#muu,cov, _, _ = robot_tracker.batch_filter(zs)
'''
for x, P in zip(mu, cov):
    # covariance of x and y
    cov = np.array([[P[0, 0], P[2, 0]], 
                    [P[0, 2], P[2, 2]]])
    mean = (x[0, 0], x[2, 0])
    plot_covariance_ellipse(mean, cov=cov, fc='g', std=3, alpha=0.5)
'''

for i in range(N):
    robot_tracker.predict()
    robot_tracker.update(zs[i])
    mean=(robot_tracker.x[0,0],robot_tracker.x[2,0])
    cov=np.array([[robot_tracker.P_post[0,0],robot_tracker.P_post[2,0]],
    [robot_tracker.P_post[0,2],robot_tracker.P_post[2,2]]])
    plot_covariance_ellipse(mean, cov, fc='g', std=3, alpha=0.5)
    mu[i,:]=robot_tracker.x
#plot results
zs *= .3048 # convert to meters
plot_filter(mu[:, 0], mu[:, 2])
plot_measurements(zs[:, 0], zs[:, 1])
plt.legend(loc=2)
plt.xlim(0, 20);
plt.show();
