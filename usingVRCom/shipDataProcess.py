import math


'''
double CentLL[45][2]={
		28.000000,122.200000,                     //0 公海
		31.226737,121.495750,                     //1.黄浦江
//		31.242743,121.496339,                     //1.黄浦江 _new
		35.080000,129.060000,                     //2.韩国釜山港
		1.2000000,103.800000,                     //3.新加坡港         
		51.000000,1.40000000,                     //4.多弗尔海峡   
		30.560570,122.176770,                     //5.洋山深水港 
		38.950000,121.800000,                     //6.大连港       
		30.000000,126.000000,                     //7.模拟狭水道
		35.350000,139.750000,                     //8.日本东京湾      
		30.769999,120.760002,                     //9.嘉兴环城河南湖 
		30.800000,120.700000,                     //10.嘉兴西塘乌镇          
		30.750000,120.700000,                     //11 嘉兴京杭运河             
		30.333333,113.916667,                     //12 长江武汉段金口水道
		31.100000,122.000000,                     //13 长江上海段 
		29.900000,122.200000,                     //14 宁波舟山水域           
		24.000000,118.000000,                     //15 台湾海峡                
		32.230000,119.200000,			 		  //16 长江镇江南京段    
		31.400000,118.350000,					  //17 长江芜湖段
		29.600000,106.600000,		  			  //18 长江重庆段
		26.050000,119.550000,					  //19 福建福州港 
		24.400000,118.200000,                     //20 福建厦门港
		27.975000,120.800000,                     //21 浙江温州港
		22.000000,114.000000,                     //22 广东珠江口                       
		22.300000,114.150000,                     //23 香港维多利亚湾                           
		38.970000,117.800000,                     //24 天津港
		34.750000,119.500000,                     //25 江苏连云港
		29.600000,121.750000,                     //26 浙江象山港                          
		47.650000,132.500000,                     //27 黑龙江佳木斯                            
		18.200000,109.550000,                     //28 海南三亚港                            
		30.664900,121.331917,                     //29 上海化工区水域                       
		30.650000,121.500000,                     //30 浙江嘉兴港
		21.100000,110.500000,                     //31 广东湛江港                           
		28.100000,121.100000,                     //32 浙江台州湾           
		32.000000,120.400000,                     //33 长江张家港                          
		25.133333,119.000000,                     //34 福建湄洲湾                     
		39.900000,119.600000,                     //35 河北秦皇岛                         
 		19.750000,109.100000,                     //36 海南洋浦港                           
    	32.500000,121.500000,                     //37 江苏洋口港                          
		23.398000,113.777498,                     //38 广东东莞华南MALL                    
		37.600000,121.450000,                     //39 山东烟台港  
		36.000000,120.220000,                     //40 山东青岛港        //原来23号  
		23.300000,116.800000,                     //41 广东汕头港        //原来28号	
		28.300000,51.2000000,      	              //42 伊朗北帕斯
		40.300000,124.800000,                     //43 吉林丹东港         
		51.000000,123.000000                     //44 黑龙江黑河港       	
};'''



AllPortCentLL={
    0:(28.000000,122.200000,'0 公海'),                    
    1:(31.226737,121.495750,'1.黄浦江'),
	2:(35.080000,129.060000,'2.韩国釜山港'),
	3:(1.2000000,103.800000,'3.新加坡港'),         
	4:(51.000000,1.40000000,'4.多弗尔海峡'),   
	5:(30.560570,122.176770,'5.洋山深水港'), 
	6:(38.950000,121.800000,'6.大连港'),     
	7:(30.000000,126.000000,'7.模拟狭水道'),
	8:(35.350000,139.750000,'8.日本东京湾'),     
	9:(30.769999,120.760002,'9.嘉兴环城河南湖'), 
	10:(30.800000,120.700000,'10.嘉兴西塘乌镇'),         
	11:(30.750000,120.700000,'11 嘉兴京杭运河'),             
	12:(30.333333,113.916667,'12 长江武汉段金口水道'),
	13:(31.100000,122.000000,'13 长江上海段'),
	14:(29.900000,122.200000,'14 宁波舟山水域'),          
	15:(24.000000,118.000000,'15 台湾海峡'),                
	16:(32.230000,119.200000,'16 长江镇江南京段'),    
	17:(31.400000,118.350000,'17 长江芜湖段'),
	18:(29.600000,106.600000,'18 长江重庆段'),
	19:(26.050000,119.550000,'19 福建福州港'),
	20:(24.400000,118.200000, '20 福建厦门港'),
	21:(27.975000,120.800000,'21 浙江温州港'),
	22:(22.000000,114.000000,'22 广东珠江口'),                       
	23:(22.300000,114.150000,'23 香港维多利亚湾'),                           
	24:(38.970000,117.800000,'24 天津港'),
	25:(34.750000,119.500000,'25 江苏连云港'),
	26:(29.600000,121.750000,'26 浙江象山港'),                          
	27:(47.650000,132.500000,'27 黑龙江佳木斯'),                            
	28:(18.200000,109.550000,'28 海南三亚港'),                       
	29:(30.664900,121.331917,'29 上海化工区水域'),                       
	30:(30.650000,121.500000,'30 浙江嘉兴港'),
	31:(21.100000,110.500000,'31 广东湛江港'),                        
	32:(28.100000,121.100000,'32 浙江台州湾'),         
	33:(32.000000,120.400000,'33 长江张家港'),                          
	34:(25.133333,119.000000,'34 福建湄洲湾'),                     
	35:(39.900000,119.600000,'35 河北秦皇岛'),                         
 	36:(19.750000,109.100000,'36 海南洋浦港'),                           
    37:(32.500000,121.500000,'37 江苏洋口港'),                          
	38:(23.398000,113.777498,'38 广东东莞华南MALL'),                    
	39:(37.600000,121.450000,'39 山东烟台港'),  
	40:(36.000000,120.220000,'40 山东青岛港'),        #原来23号  
	41:(23.300000,116.800000,'41 广东汕头港'),        #原来28号	
	42:(28.300000,51.2000000,'42 伊朗北帕斯'),
	43:(40.300000,124.800000,'43 吉林丹东港'),         
	44:(51.000000,123.000000,'44 黑龙江黑河港')
    }

''' test using AllPortCentLL
print(AllPortCentLL[0][0])
print(AllPortCentLL[0][1])
print(AllPortCentLL[0][2])
print(len(AllPortCentLL))
for portKey in AllPortCentLL.keys():
    print(AllPortCentLL[portKey])
'''
'''currentPortNo=0
def getCurrentPortNo()->int:
    return currentPortNo
def setCurrentPortNo(portNo):
	currentPortNo=portNo
	portNoStr='PortNo: {0}'.format(currentPortNo)
	print(portNoStr)
'''
#角度转换成弧度
RAD=0.01745329251994  #3.141592654/180.0

#转化精确度已经和控制台（Instructor)基本保持一致了
def shipPosXToLongitue(xValue,currentPortNo=0)->float:
	longitue=xValue/60.0/math.cos(AllPortCentLL[currentPortNo][0]*RAD)+AllPortCentLL[currentPortNo][1]
	#longitue=xValue/60.0+AllPortCentLL[currentPortNo][1]
	return longitue
#转化精确度已经和控制台（Instructor)基本保持一致了
def shipPosYToLattitude(yValue,currentPortNo=0)->float:
	#lattitude=LLatLat(yValue/60.0+AllPortCentLL[currentPortNo][0])
	#return lattitude
	return yValue/60.0+AllPortCentLL[currentPortNo][0]

E=0.081813369       #sqrt((1-6356863.0/6378245.0)*(1+6356863.0/6378245.0)) //地球扁率
PI=3.14159265359    #//(atan(1.0)*4)
DEG=57.29577951308  #//(180/pi)
#由纬度渐长率求纬度 2004-02-11 gkp
def LLatLat(llat)->float:
	i=0
	if (llat>0):
		i=1
	else:
		i=-1
	u=0.0
	p=0.0
	Lat0=0.0
	Lat=10.0  #可以保证先执行一遍循环
	llat0=math.fabs(llat)
	df=math.exp(llat0/60*RAD)
	while(math.fabs(Lat-Lat0)>0.000001):
		Lat0 = Lat
		u = E * math.sin(Lat0)
		p = (1 + u) / (1 - u)
		p = pow(p,E/2)
		p = p * df
		Lat = (math.atan(p) - PI / 4) * 2
	return Lat*DEG *i


#转化精确度已经和控制台（Instructor)基本保持一致了
def shipPosXToLattitue(xValue,currentPortNo=0)->float:
	lattitude=xValue/60.0+AllPortCentLL[currentPortNo][0]
	return lattitude

	longitue=xValue/60.0/math.cos(AllPortCentLL[currentPortNo][0]*RAD)+AllPortCentLL[currentPortNo][1]
	#longitue=xValue/60.0+AllPortCentLL[currentPortNo][1]
	return longitue
#转化精确度已经和控制台（Instructor)基本保持一致了
def shipPosYToLongitue(yValue,currentPortNo=0)->float:
	longitue=yValue/60.0/math.cos(AllPortCentLL[currentPortNo][0]*RAD)+AllPortCentLL[currentPortNo][1]
	return longitue

	lattitude=yValue/60.0+AllPortCentLL[currentPortNo][0]
	return lattitude

def isShipBaseInfoValid(shipData)->bool:
	return False

'''	
	#/小心驶得万年船
	if (shipData.c <= DBL_MAX or shipData.c >= -DBL_MAX)
		return False
	if (osg::isNaN(shipData.p) || !(shipData.p <= DBL_MAX && shipData.p >= -DBL_MAX))
		shipData.p = 0;
	if (osg::isNaN(shipData.r) || !(shipData.r <= DBL_MAX && shipData.r >= -DBL_MAX))
		shipData.r = 0;

	if (osg::isNaN(shipData.mroll) || !(shipData.mroll <= DBL_MAX && shipData.mroll >= -DBL_MAX))
		shipData.mroll = 0;

	if (osg::isNaN(shipData.proll) || !(shipData.proll <= DBL_MAX && shipData.proll >= -DBL_MAX))
		shipData.proll = 0;

	if (osg::isNaN(shipData.mpitch) || !(shipData.mpitch <= DBL_MAX && shipData.mpitch >= -DBL_MAX))
		shipData.mpitch = 0;

	if (osg::isNaN(shipData.ppitch) || !(shipData.ppitch <= DBL_MAX && shipData.ppitch >= -DBL_MAX))
		shipData.ppitch = 0;
'''