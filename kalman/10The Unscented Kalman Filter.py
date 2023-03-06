import os
import sys

#针对Visual Studio Code，如果工作目录是pythonTest
sys.path.append("../Kalman-and-Bayesian-Filters-in-Python")
sys.path.append("../filterpy")

'''
#针对Visual Studio Code，如果工作目录是kalman
sys.path.append("../../Kalman-and-Bayesian-Filters-in-Python")
sys.path.append("../../filterpy")
'''

from kf_book.book_plots import set_figsize, figsize
import matplotlib.pyplot as plt
from kf_book.nonlinear_plots import plot_nonlinear_func
from numpy.random import normal
import numpy as np

# create 500,000 samples with mean 0, std 1
gaussian = (0., 1.)
data = normal(loc=gaussian[0], scale=gaussian[1], size=500000)

def f(x):
    return (np.cos(4*(x/2 + 0.7))) - 1.3*x

plot_nonlinear_func(data, f)