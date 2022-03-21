#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#
# generated by wxGlade 1.0.4 on Mon Mar 21 11:48:35 2022
#

import threading
import wx
import wx.lib.newevent
import serial
import serial.tools.list_ports

# begin wxGlade: dependencies
# end wxGlade

# begin wxGlade: extracode
# end wxGlade

#接受数据事件
SerialRxEvent, EVT_SERIALRX = wx.lib.newevent.NewEvent()
#SERIALRX = wx.NewEventType()

class MyDialog(wx.Dialog):
    def __init__(self, *args, **kwds):

        #线程开始标志
        self.alive = threading.Event()
        self.thread = None

        # begin wxGlade: MyDialog.__init__
        kwds["style"] = kwds.get("style", 0) | wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER
        wx.Dialog.__init__(self, *args, **kwds)
        self.SetTitle("Com")

        sizer_1 = wx.BoxSizer(wx.HORIZONTAL)

        sizer_4 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, u"串口设置"), wx.VERTICAL)
        sizer_1.Add(sizer_4, 1, wx.FIXED_MINSIZE, 0)

        sizer_2 = wx.GridSizer(5, 2, 20, 20)
        sizer_4.Add(sizer_2, 1, wx.ALL, 1)

        label_2 = wx.StaticText(self, wx.ID_ANY, "Port")
        sizer_2.Add(label_2, 0, wx.EXPAND, 0)

        self.choice_port = wx.Choice(self, wx.ID_ANY, choices=[])
        self.choice_port.SetMinSize((108, 25))
        sizer_2.Add(self.choice_port, 0, 0, 0)

        label_4 = wx.StaticText(self, wx.ID_ANY, "Baudrate")
        sizer_2.Add(label_4, 0, wx.EXPAND, 0)

        self.combo_box_baudrate = wx.ComboBox(self, wx.ID_ANY, choices=[], style=wx.CB_DROPDOWN)
        self.combo_box_baudrate.SetMinSize((109, 25))
        sizer_2.Add(self.combo_box_baudrate, 0, 0, 0)

        label_3 = wx.StaticText(self, wx.ID_ANY, "DataBytes")
        sizer_2.Add(label_3, 0, wx.EXPAND, 0)

        self.choice_databytes = wx.Choice(self, wx.ID_ANY, choices=[])
        self.choice_databytes.SetMinSize((108, 25))
        sizer_2.Add(self.choice_databytes, 0, 0, 0)

        label_5 = wx.StaticText(self, wx.ID_ANY, "Stopbits")
        sizer_2.Add(label_5, 0, wx.EXPAND, 0)

        self.choice_stopbits = wx.Choice(self, wx.ID_ANY, choices=[])
        self.choice_stopbits.SetMinSize((108, 25))
        sizer_2.Add(self.choice_stopbits, 0, 0, 0)

        label_6 = wx.StaticText(self, wx.ID_ANY, "Parity")
        sizer_2.Add(label_6, 0, wx.EXPAND, 0)

        self.choice_parity = wx.Choice(self, wx.ID_ANY, choices=[])
        self.choice_parity.SetMinSize((108, 25))
        sizer_2.Add(self.choice_parity, 0, 0, 0)

        self.button_open = wx.Button(self, wx.ID_ANY, "OPEN")
        self.button_open.SetMinSize((240, 25))
        sizer_4.Add(self.button_open, 0, 0, 0)

        sizer_3 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(sizer_3, 1, 0, 0)

        self.text_ctrl_receive = wx.TextCtrl(self, wx.ID_ANY, "", style=wx.TE_MULTILINE)
        self.text_ctrl_receive.SetMinSize((206, 100))
        sizer_3.Add(self.text_ctrl_receive, 0, wx.EXPAND, 0)

        self.Clear = wx.Button(self, wx.ID_CANCEL, "")
        sizer_3.Add(self.Clear, 0, wx.ALIGN_RIGHT, 0)

        self.text_ctrl_send = wx.TextCtrl(self, wx.ID_ANY, "")
        sizer_3.Add(self.text_ctrl_send, 0, wx.EXPAND, 0)

        self.Send = wx.Button(self, wx.ID_OK, "")
        self.Send.SetDefault()
        sizer_3.Add(self.Send, 0, wx.ALIGN_RIGHT, 0)

        self.SetSizer(sizer_1)
        sizer_1.Fit(self)

        self.Layout()

        self.Bind(wx.EVT_BUTTON, self.OnButtonOpenFuction, self.button_open)
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

    def OnButtonOpenFuction(self, event):  # wxGlade: MyDialog.<event_handler>
        if(self.button_open.GetLabel()=='OPEN'):
            self.button_open.SetLabel('Close')
            self.openPort()
        else:
            self.button_open.SetLabel('OPEN')
            self.closePort()
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
# end of class MyDialog

class MyApp(wx.App):
    def OnInit(self):
        ser = serial.Serial()
        print(ser)

        self.dialog = MyDialog(None, wx.ID_ANY, "")
        self.SetTopWindow(self.dialog)
        self.dialog.ShowModal()
        self.dialog.Destroy()
        return True

# end of class MyApp

if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()
