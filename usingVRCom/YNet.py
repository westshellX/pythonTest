import ctypes
#匹配数据结构
'''typedef struct VRCustomCmdData
{
	CHAR	szCmd[256];
}VR_CUSTOM_CMDDATA,*LPVR_CUSTOM_CMDDATA;'''


class VR_CUSTOM_CMDDATA(ctypes.Structure):
    _fields_=[
        ('szCmd',ctypes.c_char*256),
    ]
'''struct DynamicShipBase{
	long	nMMSI;
	double	x,y,z,p,r;			//船舶六自由度姿态信息(世界空间坐标) 单位为米
	double	c,v,gc,gv,rot; 		//分别表示航向(度)、航速（节＝海里/小时）、对地航向、对地航速、转向率(转/分钟)
	double  tc,mroll, proll, mpitch, ppitch;
	double  vroll, vpitch;// vz;
};'''
class DynamicShipBase(ctypes.Structure):
    _fields_=[
        ('nMMSI',ctypes.c_long),
        ('x',ctypes.c_double),
        ('y',ctypes.c_double),
        ('z',ctypes.c_double),
        ('p',ctypes.c_double),
        ('r',ctypes.c_double),
        ('c',ctypes.c_double),
        ('v',ctypes.c_double),
        ('gc',ctypes.c_double),
        ('gv',ctypes.c_double),
        ('rot',ctypes.c_double),
        ('tc',ctypes.c_double),
        ('mroll',ctypes.c_double),
        ('proll',ctypes.c_double),
        ('mpitch',ctypes.c_double),
        ('ppitch',ctypes.c_double),
        ('vroll',ctypes.c_double),
        ('vpitch',ctypes.c_double)
    ]
#加载YNet.dll
YNet=ctypes.cdll.LoadLibrary('.\YNet.dll')

#匹配常用函数名
AutoAll_RunDll=YNet.AutoAll_RunDll

InitYNet=getattr(YNet,"?InitYNet@@YA_NXZ")

ClearYNet=getattr(YNet,"?ClearYNet@@YAXXZ")

PopupCommandStrSvr2Clt=getattr(YNet,"?PopupCommandStrSvr2Clt@@YA_NAAUVRCustomCmdData@@@Z")
PopupCommandStrSvr2Clt.argtypes=[ctypes.POINTER(VR_CUSTOM_CMDDATA)]

GetDynamShipList=getattr(YNet,"?GetDynamShipList@@YAPAUDynamicShipBase@@AAKPA_K@Z")
GetDynamShipList.argtypes=[ctypes.POINTER(ctypes.c_ulong),ctypes.POINTER(ctypes.c_ulonglong)]
GetDynamShipList.restype =ctypes.POINTER(DynamicShipBase)

LockDynamShipList=getattr(YNet,"?LockDynamShipList@@YAXXZ")

UnlockDynamShipList=getattr(YNet,"?UnlockDynamShipList@@YAXXZ")

CreateClient=getattr(YNet,"?CreateClient@@YAPAXXZ")

CreateSever=getattr(YNet,"?CreateServer@@YAPAXXZ")

print('------------------------------------------------')
'''
YNet=cdll.LoadLibrary('D:/cppProjects/YNet/YNet.dll')
print('------------------------------------------------')
print(YNet)
print('------------------------------------------------')
'''

#运行测试
InitYNet()
CreateClient()

#cmdStr=ctypes.create_string_buffer(256)
cmdStr=VR_CUSTOM_CMDDATA()
while 1:
    while PopupCommandStrSvr2Clt(cmdStr):
        print(cmdStr.szCmd)
        #print(cmdStr[0:255])
    m_nVSLCnt=ctypes.c_ulong(0)
    #print(m_nVSLCnt.value)
    #print('*****************************************************************')
    if(cmdStr.szCmd==b'G\n'):
        lpElapsedTime=ctypes.c_ulonglong(0)
        m_pVSL=ctypes.POINTER(DynamicShipBase)
        LockDynamShipList()
        m_pVSL=GetDynamShipList(ctypes.byref(m_nVSLCnt),ctypes.byref(lpElapsedTime))
        if(m_nVSLCnt.value>0):
            shipIndex=0
            print(m_pVSL[shipIndex].nMMSI,m_pVSL[shipIndex].x,m_pVSL[shipIndex].y,m_pVSL[shipIndex].c)
        UnlockDynamShipList()
    #print('*****************************************************************')
    elif(cmdStr.szCmd==b'H\n'):
        index=0
ClearYNet()