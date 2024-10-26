import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyparsing import alphas
from numpy.random import randn
import numpy as np
import time
from VRCom import *
import ctypes
import itertools

#最大船舶数量
SHIPMAX=200
osShipID=1
isOS=False
isShipMachineRun=False

def showShipDataFigure(figurePlot,shipData):
    figurePlot.plot(shipData.x,shipData.y)
def on_close(event):
    print('Closed Figure!')
    ClearVRCom()
    
def init():
    # fig=plt.figure()
    # ax1=fig.add_subplot(2,2,1)
    # ax2=fig.add_subplot(2,2,2)
    # ax3=fig.add_subplot(2,2,3)
    # ax4=fig.add_subplot(2,2,4)
    
    # ax4.plot(np.arange(10))
    # ax3.plot(randn(50).cumsum(),'k--')
    # ax1.hist(randn(100),bins=20,color='k',alpha=0.3)
    # ax2.scatter(np.arange(30),np.arange(30)+3*randn(30))
    

    # plt.show()
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata,ydata)
    return line,
    
def DoNetStrFromShipMachine(szCmd):
        #改变编码方式，转换成便于理解的字符串
        cmdStr=szCmd.decode('utf-8')
        #python中没有switch 语句的用法
        cmdFirstChar=cmdStr[0]
        print(cmdStr)
        #需要使用正则表达式来提取数字
        validNum=re.findall(r"-?\d+\.?\d*",cmdStr)

        if(cmdFirstChar=='A'):
		#if (sscanf(str, "A%d,%lf,%lf,%lf,%lf,%d,%d,%d\n", &ship_id, &yy, &xx, &cc, &vv, &ship_type, &mm, &ss) != 8)
            if(len(validNum)!=8):
                return
            ship_id=int(validNum[0])
            yy=float(validNum[1])
            xx=float(validNum[2])
            cc=float(validNum[3])
            vv=float(validNum[4])
            ship_type=int(validNum[5])
            mm=int(validNum[6])
            ss=int(validNum[7])
            shipData=DynamicShipBase
            shipData.x=xx
            shipData.y=yy
            shipData.c=cc
            shipData.nMMSI=ship_id
            # if(ship_id==self.osShipID):
            #     isOS=True
            # self.ProcessShipDataToAIS(shipData,isOS)

        elif(cmdFirstChar=='G'):
            isShipMachineRun=True
        elif(cmdFirstChar=='H'):
            isShipMachineRun=False
        elif(cmdFirstChar=='I'):
            #港口初始化 
            #匹配的形式 整数，小数，整数加小数
            #if(sscanf(str, "I%d,%d,%lf,%lf,%lf,%lf,%d\n", &port_id, &ship_type, &yy, &xx, &cc, &vv, &test_no)!=7)
            if(len(validNum)!=7):
                return
            port_id=int(validNum[0])
            ship_type=int(validNum[1])
            yy=float(validNum[2])
            xx=float(validNum[3])
            cc=float(validNum[4])
            vv=float(validNum[5])
            test_no=int(validNum[6])
            portNo=port_id

            shipData=DynamicShipBase
            shipData.x=xx
            shipData.y=yy
            shipData.c=cc
            #主本船从0开始
            # shipDatnMMSI=osShipID
            # self.ProcessShipDataToAIS(shipData,True)

#数据产生方式
def data_gen():
    for cnt in itertools.count():
        t=cnt/10
        yield t,np.sin(2*np.pi*t)*np.exp(-t/10.)
    yield 1,1
    
def run(frame):
    t,y=frame
    xdata.append(t)
    ydata.append(y)

    xmin,xmax=ax.get_xlim()
    if t>=xmax:
        ax.set_xlim(xmin,2*xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata,ydata)
    cmdStr=VR_CUSTOM_CMDDATA()
    while PopupCommandStrSvr2Clt(cmdStr):
        #显示出来便于观察
        print(cmdStr.szCmd)
        DoNetStrFromShipMachine(cmdStr.szCmd)  


    m_nVSLCnt=ctypes.c_ulong(0)
    lpElapsedTime=ctypes.c_ulonglong(0)
        #m_pVSL=ctypes.POINTER(DynamicShipBase)
    LockDynamShipList()

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

                        #添加位置坐标值
                        xdata.append(m_pVSL[shipIndex].x)
                        ydata.append(m_pVSL[shipIndex].y)
                                
                        shipStateInfoStr='%d,%f,%f,%f'%(m_pVSL[shipIndex].nMMSI,m_pVSL[shipIndex].x,m_pVSL[shipIndex].y,m_pVSL[shipIndex].c)
                            # self.text_ctrl_dynamicShipData.AppendText(shipStateInfoStr)
                        print(shipStateInfoStr)
    UnlockDynamShipList()
    
    # elif(isShipMachineRun==False):
    #     #print('No shipData received!')
    #     i=0

    #line.set_data(xdata,ydata)
    return line,

'''
初衷：接受船舶的动态数据，在二维平面输出，便于插值
'''  
if __name__ == "__main__":

    fig,ax=plt.subplots()
    fig.canvas.mpl_connect('close_event',on_close)
    line,=ax.plot([],[],'ro')
    ax.grid()
    xdata=[]
    ydata=[]
    
    #运行测试
    InitVRCom()
    CreateClient()

    ani=animation.FuncAnimation(fig,run,data_gen,init_func=init,interval=100,blit=True)
    plt.show()