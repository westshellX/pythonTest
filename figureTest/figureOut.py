import matplotlib.pyplot as plt
from pyparsing import alphas
from numpy.random import randn
import numpy as np

from VRCom import *
def figureTest():
    fig=plt.figure()
    ax1=fig.add_subplot(2,2,1)
    ax2=fig.add_subplot(2,2,2)
    ax3=fig.add_subplot(2,2,3)
    ax4=fig.add_subplot(2,2,4)
    
    ax4.plot(np.arange(10))
    ax3.plot(randn(50).cumsum(),'k--')
    ax1.hist(randn(100),bins=20,color='k',alpha=0.3)
    ax2.scatter(np.arange(30),np.arange(30)+3*randn(30))
    
    plt.show()

if __name__ == "__main__":
    #运行测试
    InitVRCom()
    CreateClient()
    

    ClearVRCom()