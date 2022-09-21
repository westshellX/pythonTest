from re import T
import matplotlib.pyplot as plt
from pyparsing import alphas
from numpy.random import randn
import numpy as np
import time
from VRCom import *
import ctypes
def showShipDataFigure(figurePlot,shipData):
    figurePlot.plot(shipData.x,shipData.y)
def on_close(event):
    print('Closed Figure!')
    ClearVRCom()
    
def figureTest():
    # fig=plt.figure()
    # ax1=fig.add_subplot(2,2,1)
    # ax2=fig.add_subplot(2,2,2)
    # ax3=fig.add_subplot(2,2,3)
    # ax4=fig.add_subplot(2,2,4)
    
    # ax4.plot(np.arange(10))
    # ax3.plot(randn(50).cumsum(),'k--')
    # ax1.hist(randn(100),bins=20,color='k',alpha=0.3)
    # ax2.scatter(np.arange(30),np.arange(30)+3*randn(30))
    
    # fig.canvas.mpl_connect('close_event',on_close)
    # plt.show()

    fig,ax=plt.subplots()
    th=np.linspace(0,2*np.pi,512)
    ln,=ax.plot(th,np.sin(th))
    t=0
    while True:
        x,y=t,np.sin(t)
        # ln.set_data(x,y)
        plt.pause(1)
        t=t+1
    plt.show()
if __name__ == "__main__":

    figureTest()
    
    #运行测试
    InitVRCom()
    CreateClient()

    
    #最大船舶数量
    SHIPMAX=200
    osShipiID=1
    while True:
        LockDynamShipList()

        m_nVSLCnt=ctypes.c_ulong(0)
        lpElapsedTime=ctypes.c_ulonglong(0)
        m_pVSL=GetDynamShipList(ctypes.byref(m_nVSLCnt),ctypes.byref(lpElapsedTime))
                
                #转化成秒
        timeSecond=float(lpElapsedTime.value/1000.0)
        if(timeSecond<100.0 and m_nVSLCnt.value>0):
            print(m_nVSLCnt.value)
            for shipIndex in range(m_nVSLCnt.value):
                if(m_pVSL[shipIndex].nMMSI>0 and m_pVSL[shipIndex].nMMSI<SHIPMAX):
                    isOS=False
                    #print('shipIndex={} nMMSI={}'.format(shipIndex,m_pVSL[shipIndex].nMMSI))
                    if(m_pVSL[shipIndex].nMMSI==osShipID):
                        isOS=True
                        showShipDataFigure(m_pVSL[shipIndex])
        UnlockDynamShipList()
        time.sleep(0.5)



    ClearVRCom()