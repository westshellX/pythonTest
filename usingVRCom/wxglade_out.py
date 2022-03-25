#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#
# generated by wxGlade 1.0.4 on Mon Mar 21 11:48:35 2022
#
import socket
import threading
from bitarray import test
import wx
import wx.lib.newevent
import serial
import serial.tools.list_ports
from VRCom import *
import ctypes
import shipDataProcess
import encodeDictTest
import re
import time

# begin wxGlade: dependencies
# end wxGlade

# begin wxGlade: extracode
# end wxGlade

#接受数据事件
SerialRxEvent, EVT_SERIALRX = wx.lib.newevent.NewEvent()
#SERIALRX = wx.NewEventType()


class ComDialog(wx.Dialog):
    def __init__(self, *args, **kwds):

        self.isShipMachineRun=False

        #线程开始标志
        self.alive = threading.Event()
        self.thread = None

        #VR运行线程
        self.vrAlive=threading.Event()
        self.vrThread= threading.Thread(target=self.vrRunThread)
        self.vrThread.setDaemon(1)
        self.vrAlive.set()
        self.vrThread.start()

        #Socket
        self.aisSocket=None

        # Content of this block not found. Did you rename this class?
        # begin wxGlade: ComDialog.__init__
        kwds["style"] = kwds.get("style", 0) | wx.DEFAULT_DIALOG_STYLE
        wx.Dialog.__init__(self, *args, **kwds)
        self.SetSize((644, 461))
        self.SetTitle("Com")

        sizer_1 = wx.BoxSizer(wx.HORIZONTAL)

        sizer_4 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, u"串口设置"), wx.VERTICAL)
        sizer_1.Add(sizer_4, 1, wx.FIXED_MINSIZE, 0)

        sizer_2 = wx.FlexGridSizer(5, 2, 2, 3)
        sizer_4.Add(sizer_2, 1, wx.ALL | wx.EXPAND, 1)

        label_2 = wx.StaticText(self, wx.ID_ANY, "Port")
        sizer_2.Add(label_2, 0, wx.EXPAND, 0)

        self.choice_port = wx.Choice(self, wx.ID_ANY, choices=[])
        self.choice_port.SetMinSize((108, 25))
        sizer_2.Add(self.choice_port, 0, wx.EXPAND, 0)

        label_4 = wx.StaticText(self, wx.ID_ANY, "Baudrate")
        sizer_2.Add(label_4, 0, wx.EXPAND, 0)

        self.combo_box_baudrate = wx.ComboBox(self, wx.ID_ANY, choices=[], style=wx.CB_DROPDOWN)
        self.combo_box_baudrate.SetMinSize((109, 25))
        sizer_2.Add(self.combo_box_baudrate, 0, wx.EXPAND, 0)

        label_3 = wx.StaticText(self, wx.ID_ANY, "DataBytes")
        sizer_2.Add(label_3, 0, wx.EXPAND, 0)

        self.choice_databytes = wx.Choice(self, wx.ID_ANY, choices=[])
        self.choice_databytes.SetMinSize((108, 25))
        sizer_2.Add(self.choice_databytes, 0, wx.EXPAND, 0)

        label_5 = wx.StaticText(self, wx.ID_ANY, "Stopbits")
        sizer_2.Add(label_5, 0, wx.EXPAND, 0)

        self.choice_stopbits = wx.Choice(self, wx.ID_ANY, choices=[])
        self.choice_stopbits.SetMinSize((108, 25))
        sizer_2.Add(self.choice_stopbits, 0, wx.EXPAND, 0)

        label_6 = wx.StaticText(self, wx.ID_ANY, "Parity")
        sizer_2.Add(label_6, 0, wx.EXPAND, 0)

        self.choice_parity = wx.Choice(self, wx.ID_ANY, choices=[])
        self.choice_parity.SetMinSize((108, 25))
        sizer_2.Add(self.choice_parity, 0, wx.EXPAND, 0)

        self.button_open = wx.Button(self, wx.ID_ANY, "OPEN")
        self.button_open.SetMinSize((240, 25))
        sizer_4.Add(self.button_open, 0, wx.EXPAND, 0)

        sizer_4.Add((20, 20), 0, wx.EXPAND, 0)

        self.text_ctrl_receive = wx.TextCtrl(self, wx.ID_ANY, "", style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.text_ctrl_receive.SetMinSize((206, 100))
        sizer_4.Add(self.text_ctrl_receive, 0, wx.EXPAND, 0)

        self.button_clear = wx.Button(self, wx.ID_ANY, "Clear")
        sizer_4.Add(self.button_clear, 0, wx.ALIGN_RIGHT, 0)

        self.text_ctrl_send = wx.TextCtrl(self, wx.ID_ANY, "")
        sizer_4.Add(self.text_ctrl_send, 0, wx.EXPAND, 0)

        self.button_send = wx.Button(self, wx.ID_ANY, "Send")
        sizer_4.Add(self.button_send, 0, wx.ALIGN_RIGHT, 0)

        sizer_3 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(sizer_3, 1, 0, 0)

        sizer_8 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, "VRComData"), wx.VERTICAL)
        sizer_3.Add(sizer_8, 1, wx.EXPAND, 0)

        sizer_9 = wx.FlexGridSizer(2, 2, 5, 0)
        sizer_8.Add(sizer_9, 1, wx.EXPAND, 0)

        label_9 = wx.StaticText(self, wx.ID_ANY, "CmdStr")
        sizer_9.Add(label_9, 0, wx.ALIGN_CENTER, 0)

        self.text_ctrl_cmdStr = wx.TextCtrl(self, wx.ID_ANY, "", style=wx.TE_MULTILINE)
        sizer_9.Add(self.text_ctrl_cmdStr, 0, wx.EXPAND, 0)

        label_10 = wx.StaticText(self, wx.ID_ANY, "DynamicShipData")
        sizer_9.Add(label_10, 0, wx.ALIGN_CENTER_VERTICAL, 0)

        self.text_ctrl_dynamicShipData = wx.TextCtrl(self, wx.ID_ANY, "", style=wx.TE_MULTILINE)
        sizer_9.Add(self.text_ctrl_dynamicShipData, 0, wx.EXPAND, 0)

        sizer_6 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, u"AIS："), wx.VERTICAL)
        sizer_3.Add(sizer_6, 1, wx.EXPAND, 0)

        sizer_7 = wx.BoxSizer(wx.VERTICAL)
        sizer_6.Add(sizer_7, 1, wx.EXPAND, 0)

        self.text_ctrl_aisInfo = wx.TextCtrl(self, wx.ID_ANY, "", style=wx.TE_MULTILINE | wx.TE_READONLY)
        sizer_7.Add(self.text_ctrl_aisInfo, 0, wx.EXPAND, 0)

        grid_sizer_1 = wx.GridSizer(2, 3, 0, 0)
        sizer_7.Add(grid_sizer_1, 1, wx.EXPAND, 0)

        label_1 = wx.StaticText(self, wx.ID_ANY, "Protocol")
        grid_sizer_1.Add(label_1, 0, wx.ALIGN_CENTER, 0)

        self.radio_btn_tcp = wx.RadioButton(self, wx.ID_ANY, "TCP")
        self.radio_btn_tcp.SetValue(1)
        grid_sizer_1.Add(self.radio_btn_tcp, 0, 0, 0)

        self.radio_btn_udp = wx.RadioButton(self, wx.ID_ANY, "UDP")
        self.radio_btn_udp.SetValue(1)
        grid_sizer_1.Add(self.radio_btn_udp, 0, 0, 0)

        label_8 = wx.StaticText(self, wx.ID_ANY, "Port:")
        grid_sizer_1.Add(label_8, 0, wx.ALIGN_CENTER, 0)

        self.text_ctrl_sockPort = wx.TextCtrl(self, wx.ID_ANY, "3333")
        grid_sizer_1.Add(self.text_ctrl_sockPort, 0, 0, 0)

        self.button_socket = wx.Button(self, wx.ID_ANY, "StartSocket\n")
        grid_sizer_1.Add(self.button_socket, 0, wx.ALIGN_RIGHT, 0)

        sizer_6.Add((20, 20), 0, wx.EXPAND, 0)

        sizer_5 = wx.GridSizer(1, 2, 2, 0)
        sizer_3.Add(sizer_5, 1, wx.EXPAND, 0)

        label_7 = wx.StaticText(self, wx.ID_ANY, "GPS:")
        sizer_5.Add(label_7, 0, wx.ALIGN_CENTER, 0)

        self.text_ctrl_gpsInfo = wx.TextCtrl(self, wx.ID_ANY, "", style=wx.TE_MULTILINE | wx.TE_READONLY)
        sizer_5.Add(self.text_ctrl_gpsInfo, 0, wx.EXPAND, 0)

        sizer_9.AddGrowableCol(1)

        sizer_2.AddGrowableCol(1)

        self.SetSizer(sizer_1)

        self.Layout()

        self.Bind(wx.EVT_BUTTON, self.OnButtonOpenFuction, self.button_open)
        self.Bind(wx.EVT_BUTTON, self.OnClear, self.button_clear)
        self.Bind(wx.EVT_BUTTON, self.OnSend, self.button_send)
        self.Bind(wx.EVT_BUTTON, self.SocketButton, self.button_socket)
        # end wxGlade
        
        #绑定接收数据事件与处理方法
        self.Bind(EVT_SERIALRX, self.OnSerialRead)

        self.serial=serial.Serial()
        # fill in ports and select current setting
        preferred_index = 0
        self.choice_port.Clear()
        self.ports = []
        for n, (portname, desc, hwid) in enumerate(sorted(serial.tools.list_ports.comports())):
            self.choice_port.Append(u'{} - {}'.format(portname, desc))
            self.ports.append(portname)
            if self.serial.name == portname:
                preferred_index = n
        self.choice_port.SetSelection(preferred_index)

        preferred_index = None
        # fill in baud rates and select current setting
        self.combo_box_baudrate.Clear()
        for n, baudrate in enumerate(self.serial.BAUDRATES):
            self.combo_box_baudrate.Append(str(baudrate))
            if self.serial.baudrate == baudrate:
                preferred_index = n
            if preferred_index is not None:
                self.combo_box_baudrate.SetSelection(preferred_index)
            else:
                self.combo_box_baudrate.SetValue(u'{}'.format(self.serial.baudrate))
        
        # fill in data bits and select current setting
        self.choice_databytes.Clear()
        for n, bytesize in enumerate(self.serial.BYTESIZES):
            self.choice_databytes.Append(str(bytesize))
            if self.serial.bytesize == bytesize:
                index = n
        self.choice_databytes.SetSelection(index)
        
        # fill in stop bits and select current setting
        self.choice_stopbits.Clear()
        for n, stopbits in enumerate(self.serial.STOPBITS):
            self.choice_stopbits.Append(str(stopbits))
            if self.serial.stopbits == stopbits:
                index = n
        self.choice_stopbits.SetSelection(index)

        # fill in parities and select current setting
        self.choice_parity.Clear()
        for n, parity in enumerate(self.serial.PARITIES):
            self.choice_parity.Append(str(serial.PARITY_NAMES[parity]))
            if self.serial.parity == parity:
                index = n
        self.choice_parity.SetSelection(index)

    def openPort(self):
        #设置参数
        self.serial.port = self.ports[self.choice_port.GetSelection()]

        try:
            b = int(self.combo_box_baudrate.GetValue())
        except ValueError:
            with wx.MessageDialog(
                        self,
                        'Baudrate must be a numeric value',
                        'Value Error',
                        wx.OK | wx.ICON_ERROR) as dlg:
                    dlg.ShowModal()
            success = False
        else:
            self.serial.baudrate = b

        self.serial.bytesize = self.serial.BYTESIZES[self.choice_databytes.GetSelection()]
        self.serial.stopbits = self.serial.STOPBITS[self.choice_stopbits.GetSelection()]
        self.serial.parity = self.serial.PARITIES[self.choice_parity.GetSelection()]
        #打开串口
        try:
            self.serial.open()
        except serial.SerialException as e:
            with wx.MessageDialog(self, str(e), "Serial Port Error", wx.OK | wx.ICON_ERROR)as dlg:
                dlg.ShowModal()
        else:
            #显示串口状态
            self.SetTitle("Serial Terminal on {} [{},{},{},{}{}{}]".format(
                        self.serial.portstr,
                        self.serial.baudrate,
                        self.serial.bytesize,
                        self.serial.parity,
                        self.serial.stopbits,
                        ' RTS/CTS' if self.serial.rtscts else '',
                        ' Xon/Xoff' if self.serial.xonxoff else '',
                        ))
            ok = True

            self.StartThread()
    
    def closePort(self):
        """Called on application shutdown."""

        #注意：需要先关闭串口，后关闭线程
        self.serial.close()             # cleanup
        self.StopThread()               # stop reader thread
        #self.Destroy()      

    def ComPortThread(self):
        """\
        Thread that handles the incoming traffic. Does the basic input
        transformation (newlines) and generates an SerialRxEvent
        """
        while self.alive.isSet():
            b = self.serial.read(self.serial.in_waiting or 1)
            if b:
                # newline transformation
               ''' if self.settings.newline == NEWLINE_CR:
                    b = b.replace(b'\r', b'\n')
                elif self.settings.newline == NEWLINE_LF:
                    pass
                elif self.settings.newline == NEWLINE_CRLF:
                    b = b.replace(b'\r\n', b'\n')
                '''
            wx.PostEvent(self, SerialRxEvent(data=b))
            print(b)

    #处理ship机通过VRCom发送过来的字符串命令
    def DoNetStrFromShipMachine(self,szCmd):            

        #改变编码方式，转换成便于理解的字符串
        cmdStr=szCmd.decode('utf-8')
        #python中没有switch 语句的用法
        cmdFirstChar=cmdStr[0]
        #需要使用正则表达式来提取数字
        validNum=re.findall(r"\d+\.?\d*",cmdStr)
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
            self.ProcessShipDataToAIS(shipData)

        elif(cmdFirstChar=='G'):
            self.isShipMachineRun=True
        elif(cmdFirstChar=='H'):
            self.isShipMachineRun=False
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
            shipDataProcess.setCurrentPortNo(port_id)
    def vrRunThread(self):

        print('VRThread running')
        cmdStr=VR_CUSTOM_CMDDATA()
        #线程需要一直运行
        while self.vrAlive.isSet():
            while PopupCommandStrSvr2Clt(cmdStr):
                #显示出来便于观察
                self.text_ctrl_cmdStr.AppendText(cmdStr.szCmd)
                self.DoNetStrFromShipMachine(cmdStr.szCmd)                
            if self.isShipMachineRun:
                m_nVSLCnt=ctypes.c_ulong(0)
                lpElapsedTime=ctypes.c_ulonglong(0)
                m_pVSL=ctypes.POINTER(DynamicShipBase)
                LockDynamShipList()

                m_pVSL=GetDynamShipList(ctypes.byref(m_nVSLCnt),ctypes.byref(lpElapsedTime))
                print(lpElapsedTime.value)
                
                #转化成秒
                timeSecond=float(lpElapsedTime.value/1000.0)
                if(timeSecond<100.0 and m_nVSLCnt.value>0):
                    for shipIndex in range(m_nVSLCnt.value):
                        #shipIndex=0
                        shipStateInfoStr='%d,%f,%f,%f'%(m_pVSL[shipIndex].nMMSI,m_pVSL[shipIndex].x,m_pVSL[shipIndex].y,m_pVSL[shipIndex].c)
                        self.text_ctrl_dynamicShipData.AppendText(shipStateInfoStr)
                        self.ProcessShipDataToAIS(m_pVSL[shipIndex])
                
                UnlockDynamShipList()

                #控制下速度，没必要太快
                time.sleep(0.5)

    #串口接收数据线程
    def StartThread(self):
        """Start the receiver thread"""
        self.thread = threading.Thread(target=self.ComPortThread)
        self.thread.setDaemon(1)
        self.alive.set()
        self.thread.start()
        self.serial.rts = True
        self.serial.dtr = True
        #self.frame_terminal_menubar.Check(ID_RTS, self.serial.rts)
        #self.frame_terminal_menubar.Check(ID_DTR, self.serial.dtr)

    def OnSerialRead(self, event):
        """Handle input from the serial port."""
        print(event.data)
        #self.WriteText(event.data.decode('UTF-8', 'replace'))

        #编码方式
        text=event.data.decode('UTF-8','replace')
        #显示接收到的数据
        self.text_ctrl_receive.AppendText(text)

    def StopThread(self):
        """Stop the receiver thread, wait until it's finished."""
        print(self.thread)
        if self.thread is not None:
            self.alive.clear()          # clear alive event for thread
            self.thread.join()          # wait until thread has finished
            self.thread = None        
    
    def ProcessShipDataToAIS(self,shipData):
        #还需要判断shipData数据的有效性
        print(shipData.nMMSI,shipData.x)
        shipLat=shipDataProcess.shipPosYToLattitude(shipData.y)
        shipLong=shipDataProcess.shipPosXToLongitue(shipData.x)
        aisInfoStr=encodeDictTest.encodeDict(shipData.c,shipLat,shipLong,shipData.nMMSI,1)
        print(aisInfoStr)
        self.text_ctrl_aisInfo.AppendText(aisInfoStr)

        #UDPSocket发送
        if(self.aisSocket!=None):
            host = "127.0.0.1"
            port = int(self.text_ctrl_sockPort.GetLineText(0))
            self.aisSocket.sendto(aisInfoStr.encode(), (host, port))
    def OnButtonOpenFuction(self, event):  # wxGlade: ComDialog.<event_handler>
        if(self.button_open.GetLabel()=='OPEN'):
            self.button_open.SetLabel('Close')
            self.openPort()
        else:
            self.button_open.SetLabel('OPEN')
            self.closePort()
    def OnClear(self, event):  # wxGlade: ComDialog.<event_handler>
        self.text_ctrl_receive.Clear()

    def OnSend(self, event):  # wxGlade: ComDialog.<event_handler>
        if(self.text_ctrl_send.IsEmpty()):
            return

        print(self.text_ctrl_send.GetValue())
        if(self.serial.is_open):
            self.serial.write(self.text_ctrl_send.GetValue().encode('UTF-8', 'replace'))

    def SocketButton(self, event):  # wxGlade: ComDialog.<event_handler>
        if(self.button_socket.GetLabel()=='StartSocket'):
            self.button_socket.SetLabel('CloseSocket')

            if(self.text_ctrl_sockPort.IsEmpty()):
                messageDlg=wx.MessageDialog(self,"No port setted!",'Warning')
                messageDlg.ShowModal()
                return
        
            port=int(self.text_ctrl_sockPort.GetLineText(0))
    
            if(port<0 or port==0):
              wx.MessageDialog(self,"Port number must greater than 0 !",'Warning').ShowModal()
              return

            if(self.radio_btn_tcp.GetValue()==True):
                self.StartTCPSocket()
            if(self.radio_btn_udp.GetValue()==True):
                self.StartUDPSocket()
            else:
                print("No socket!")
        else:
            self.button_socket.SetLabel('StartSocket')
            if(self.aisSocket!=None and self.aisSocket):
                print("Close aisSocket!")
    def StartUDPSocket(self):
        print("Start UDP Server!")        
        self.aisSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        '''try:
            while True:
                while True:
                    print(f"Sending {len(MESSAGES)} messages all at once.")
                # send all at once and then close
                    for msg in MESSAGES:
                        self.aisSocket.sendto(msg + b"\r\n", (host, port))
                        time.sleep(2)
        finally:
            self.aisSocket.close()'''
        #self.aisSocket.isOpen()
    def StartTCPSocket(self):
        print("Start TCP Server!")
# end of class MyDialog

class MyApp(wx.App):
    def OnInit(self):
        self.dialog = ComDialog(None, wx.ID_ANY, "")
        self.SetTopWindow(self.dialog)
        self.dialog.ShowModal()
        self.dialog.Destroy()
        return True

# end of class MyApp

if __name__ == "__main__":
    #运行测试
    InitVRCom()
    CreateClient()

    app = MyApp(0)
    app.MainLoop()

    ClearVRCom()