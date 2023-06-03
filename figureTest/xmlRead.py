#打算读取shipData.xml的数据，然后进行分析比较
from xml.dom.minidom import parse
import xml.dom.minidom

DOMTree=xml.dom.minidom.parse("C:\SMU\shipDynamicInfo.xml")

collection=DOMTree.documentElement
shipDatas=collection.getElementsByTagName("DynamicShipBase")
for shipData in shipDatas:
    #print(" %s" %shipData.getAttribute("x"))
    print("hello")