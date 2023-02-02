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
import sys
sys.path.append("C:/Users/westshell_ASUS/Documents/pythonWork/Kalman-and-Bayesian-Filters-in-Python")
from kf_book.book_plots import plot_measurements

pos, vel = (4, 3), (2, 1)
sensor = PosSensor(pos, vel, noise_std=1)
ps = np.array([sensor.read() for _ in range(50)])

plot_measurements(ps[:, 0], ps[:, 1]);
plt.show();
print(ps)


