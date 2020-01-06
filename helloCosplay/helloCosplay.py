#!/usr/bin/env python
#-*- coding: utf-8 -*-

__author__ = 'ZYSzys'

import requests
from bs4 import BeautifulSoup
import os
import bs4
import sys

class Mz:
    def __init__(self):
        self.url = 'https://aoishigure.com'
        self.headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36', 
        'Referer': 'https://aoishigure.com/'
        }
        self.req = requests.session()
        self.all_a = []
        self.all_a_title = []
        imagePath=os.path.join(os.getcwd(),'imagPath')
        print(imagePath)
        if (os.path.isdir(imagePath)!=True):
           os.makedirs(imagePath)
        if(os.getcwd()!=imagePath):
           os.chdir(imagePath)
        self.initpwd = os.getcwd()
    #确定范围
    def Domainhtml(self):
        html = self.req.get(self.url, headers=self.headers)
        lisTemp=BeautifulSoup(html.text, 'lxml').find('div', class_='post-feed')
        if isinstance(lisTemp,bs4.element.Tag):
            lis = lisTemp.find_all('article')
            for a in lis:
                imgLink = a.find('a',class_='post-card-image-link')['href']
                #取出网址
                imgurl=self.url+imgLink
                #print(imgurl)
                self.all_a.append(imgurl)

                #取出标题
                imgTile=a.find('h2',class_='post-card-title').string
                #过滤乱码
                non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
                titleTmp=imgTile.translate(non_bmp_map)
                #只保留数字和字母
                getVals=list([val for val in titleTmp if val.isalpha() or val.isnumeric()])
                title=''.join(getVals)
##                print(title)
                self.all_a_title.append(title)

    #下载
    def Downloadimg(self):
        cnt = 0
        print('total: %s' % len(self.all_a))
        for a in self.all_a:
            print('[%s] downloading start...' % self.all_a_title[cnt])
            childImagePath=os.path.join(os.getcwd(), self.all_a_title[cnt])
            if(os.path.isdir(childImagePath)!=True):
                os.makedirs(childImagePath)
            os.chdir(childImagePath)
            imghtml = self.req.get(a, headers=self.headers)
            contentList= BeautifulSoup(imghtml.text, 'lxml').find_all('div', class_='kg-gallery-image')
            for contentItem in contentList:
                if isinstance(contentItem,bs4.element.Tag):
                    aaa=contentItem.find('img')['src']
                    fileNameStart=aaa.rfind('/')
                    fileName=aaa[fileNameStart+1:len(aaa)]
                    print(fileName)
                    img = self.req.get(aaa, headers=self.headers)
                    f = open(fileName, 'ab')
                    f.write(img.content)
                    f.close()
            print('[%s] downloading end...' % self.all_a_title[cnt])
            cnt += 1
            os.chdir(self.initpwd)

        print('Dowmload completed!')

if __name__ == '__main__':
    test = Mz()
    test.Domainhtml()
    test.Downloadimg()
