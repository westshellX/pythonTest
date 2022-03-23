from VRCom import *
#from YNet import *
import ctypes

#运行测试
InitVRCom()
CreateClient()

#cmdStr=ctypes.create_string_buffer(256)
cmdStr=VR_CUSTOM_CMDDATA()
isShipMachineRun=False
while 1:
    while PopupCommandStrSvr2Clt(cmdStr):
        print(cmdStr.szCmd)
        print(cmdStr.szCmd[0:255])
        print(cmdStr.szCmd[0:1])
        if(cmdStr.szCmd==b'G\n'):
            isShipMachineRun=True
        if(cmdStr.szCmd[0:1]==b'H'):
            isShipMachineRun=False
    if isShipMachineRun:
        m_nVSLCnt=ctypes.c_ulong(0)
        lpElapsedTime=ctypes.c_ulonglong(0)
        m_pVSL=ctypes.POINTER(DynamicShipBase)
        LockDynamShipList()
        
        m_pVSL=GetDynamShipList(ctypes.byref(m_nVSLCnt),ctypes.byref(lpElapsedTime))
        if(m_nVSLCnt.value>0):
            shipIndex=0
            print(m_pVSL[shipIndex].nMMSI,m_pVSL[shipIndex].x,m_pVSL[shipIndex].y,m_pVSL[shipIndex].c)
            
        UnlockDynamShipList()
        #print('*****************************************************************')
ClearVRCom()