#!/usr/bin/env python
#-*- coding: utf-8 -*-

__author__ = 'ZYSzys'

import requests
from bs4 import BeautifulSoup
import os
import bs4

class Mz:

    #初始化
    def __init__(self):
        self.url = 'http://www.mzitu.com'
        self.headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36', 
        'Referer': 'http://www.mzitu.com/'
        }
        self.req = requests.session()
        self.all_a = []
        self.all_a_title = []
        self.all_a_max = []
        imagePath=os.path.join(os.getcwd(),'Mzitu')
        print(imagePath)
        if (os.path.isdir(imagePath)!=True):
           os.makedirs(imagePath)
        if(os.getcwd()!=imagePath):
           os.chdir(imagePath)
        self.initpwd = os.getcwd()

    def Domainhtml(self):
        html = self.req.get(self.url, headers=self.headers)
        lisTemp = BeautifulSoup(html.text, 'lxml').find('div', id_='postlist')
        if isinstance(lisTemp,bs4.element.Tag):
            lis=lisTemp.find_all('li')
            for a in lis:
                imgurl = a.find('a')['href']
                self.all_a.append(imgurl)

    def Getmaxpage(self):
        for a in self.all_a:
            imghtml = self.req.get(a, headers=self.headers)
            title = BeautifulSoup(imghtml.text, 'lxml').find('div', class_='post-context').find('p').string
            #print title
            last = BeautifulSoup(imghtml.text, 'lxml').find('div', class_='pagenavi').find_all('span')
            last = int(last[-2].string)
            self.all_a_title.append(title)
            self.all_a_max.append(last)

    def Downloadimg(self):
        cnt = 0
        print('total: %s' % len(self.all_a))
        for a in self.all_a:
            print('Downloading %s now...' % (cnt+1))
            childImagePath=os.path.join(os.getcwd(), self.all_a_title[cnt])
            if(os.path.isdir(childImagePath)!=True):
                os.makedirs(childImagePath)
            os.chdir(childImagePath)
            for i in range(1, self.all_a_max[cnt]+1):
                nurl = a+'/'+str(i)
                imghtml = self.req.get(nurl, headers=self.headers)
                tempAaa = BeautifulSoup(imghtml.text, 'lxml').find('div', class_='main-image')
                if isinstance(tempAaa,bs4.element.Tag):
                    aaa=tempAaa.find('img')['src']
                else:
                    print('NoneType found!')
                img = self.req.get(aaa, headers=self.headers)
                f = open(str(i)+'.jpg', 'ab')
                f.write(img.content)
                f.close()
            cnt += 1
            os.chdir(self.initpwd)

        print('Dowmload completed!')

if __name__ == '__main__':
    test = Mz()
    test.Domainhtml()
    test.Getmaxpage()
    test.Downloadimg()
