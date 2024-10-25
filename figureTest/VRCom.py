import ctypes
#匹配数据结构
'''
typedef struct VRCustomCmdData
{
	CHAR	szCmd[256];
}VR_CUSTOM_CMDDATA,*LPVR_CUSTOM_CMDDATA;
'''


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
#加载VRCom.dll
VRCom=ctypes.cdll.LoadLibrary('..\VRCom.dll')

#匹配常用函数名
AutoAll_RunDll=VRCom.AutoAll_RunDll

InitVRCom=getattr(VRCom,"?InitVRCom@@YA_NXZ")

ClearVRCom=getattr(VRCom,"?ClearVRCom@@YAXXZ")

PopupCommandStrSvr2Clt=getattr(VRCom,"?PopupCommandStrSvr2Clt@@YA_NAAUVRCustomCmdData@@@Z")
PopupCommandStrSvr2Clt.argtypes=[ctypes.POINTER(VR_CUSTOM_CMDDATA)]

GetDynamShipList=getattr(VRCom,"?GetDynamShipList@@YAPAUDynamicShipBase@@AAKPA_K@Z")
GetDynamShipList.argtypes=[ctypes.POINTER(ctypes.c_ulong),ctypes.POINTER(ctypes.c_ulonglong)]
GetDynamShipList.restype =ctypes.POINTER(DynamicShipBase)

LockDynamShipList=getattr(VRCom,"?LockDynamShipList@@YAXXZ")

UnlockDynamShipList=getattr(VRCom,"?UnlockDynamShipList@@YAXXZ")

CreateClient=getattr(VRCom,"?CreateClient@@YAPAXXZ")

CreateSever=getattr(VRCom,"?CreateServer@@YAPAXXZ")

__all__=[
    "InitVRCom","CreateClient",'PopupCommandStrSvr2Clt','LockDynamShipList',
    'UnlockDynamShipList','GetDynamShipList','VR_CUSTOM_CMDDATA','ClearVRCom','DynamicShipBase'
    ]