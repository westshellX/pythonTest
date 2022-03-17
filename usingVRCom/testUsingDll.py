from ctypes import *
import ctypes
from ctypes.wintypes import POINT
from operator import truediv
from pickle import TRUE
from platform import win32_edition

from pyparsing import Char
print(windll.kernel32)
print(cdll.kernel32)
#兼容的结构
'''
class VR_CUSTOM_CMDDATA(ctypes.Structure):
    _fields_=[
        ('szCmd',ctypes.c_char_p),
    ]
cd=VR_CUSTOM_CMDDATA()
cd.szCmd=ctypes.create_string_buffer(256)
print(cd.szCmd)
print(len(cd.szCmd))
print(cd.szCmd)
'''
print(__doc__)
VRCom=cdll.LoadLibrary('.\VRCom.dll')
print(VRCom)
print('------------------------------------------------')
#匹配函数名
AutoAll_RunDll=VRCom.AutoAll_RunDll
InitVRCom=getattr(VRCom,"?InitVRCom@@YA_NXZ")
ClearVRCom=getattr(VRCom,"?ClearVRCom@@YAXXZ")
PopupCommandStrSvr2Clt=getattr(VRCom,"?PopupCommandStrSvr2Clt@@YA_NAAUVRCustomCmdData@@@Z")
PopupCommandStrSvr2Clt.argtypes=[ctypes.c_char_p]
cmdStr=ctypes.create_string_buffer(256)
CreateClient=getattr(VRCom,"?CreateClient@@YAPAXXZ")
print(InitVRCom)
print('------------------------------------------------')
'''
YNet=cdll.LoadLibrary('D:/cppProjects/YNet/YNet.dll')
print('------------------------------------------------')
print(YNet)
print('------------------------------------------------')
'''
InitVRCom()
CreateClient()
index=0
while TRUE:
    while PopupCommandStrSvr2Clt(cmdStr):
        print(cmdStr)
        print(cmdStr[0:255])
        index+=1
ClearVRCom()