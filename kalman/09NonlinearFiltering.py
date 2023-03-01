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

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

N = 5000
a = np.pi/2. + (randn(N) * 0.35)
r = 50.0     + (randn(N) * 0.4)
xs = r * np.cos(a)
ys = r * np.sin(a)

plt.scatter(xs, ys, label='Sensor', color='k', 
            alpha=0.4, marker='.', s=1)
xmean, ymean = sum(xs) / N, sum(ys) / N
plt.scatter(0, 50, c='k', marker='o', s=200, label='Intuition')
plt.scatter(xmean, ymean, c='r', marker='*', s=200, label='Mean')
plt.axis('equal')
plt.legend();
plt.show();

from numpy.random import normal

data = normal(loc=0., scale=1., size=500000)
plt.hist(2*data + 1, 1000);
plt.show();

from kf_book.book_plots import set_figsize, figsize
from kf_book.nonlinear_plots import plot_nonlinear_func

def g1(x):
    return 2*x+1

plot_nonlinear_func(data, g1)
plt.show();

def g2(x):
    return (np.cos(3*(x/2 + 0.7))) * np.sin(0.3*x) - 1.6*x

plot_nonlinear_func(data, g2)
plt.show();

N = 30000
plt.subplot(121)
plt.scatter(data[:N], range(N), alpha=.1, s=1.5)
plt.title('Input')
plt.subplot(122)
plt.title('Output')
plt.scatter(g2(data[:N]), range(N), alpha=.1, s=1.5);
plt.show();

y = g2(data)
plot_nonlinear_func(y, g2)
plt.show();

print('input  mean, variance: %.4f, %.4f' % 
      (np.mean(data), np.var(data)))
print('output mean, variance: %.4f, %.4f' % 
      (np.mean(y), np.var(y)))

def g3(x): 
    return -1.5 * x

plot_nonlinear_func(data, g3)
plt.show();
out = g3(data)
print('output mean, variance: %.4f, %.4f' % 
      (np.mean(out), np.var(out)))

out = g3(data)
out2 = g2(data)

for i in range(10):
    out = g3(out)
    out2 = g2(out2)
print('linear    output mean, variance: %.4f, %.4f' % 
      (np.average(out), np.std(out)**2))
print('nonlinear output mean, variance: %.4f, %.4f' % 
      (np.average(out2), np.std(out2)**2))

def g3(x): 
    return -x*x

data = normal(loc=1, scale=1, size=500000)
plot_nonlinear_func(data, g3);
plt.show();

import kf_book.nonlinear_internal as nonlinear_internal

nonlinear_internal.plot1();
plt.show();
nonlinear_internal.plot2()
plt.show();
nonlinear_internal.plot3()
plt.show();
nonlinear_internal.plot4()
plt.show();