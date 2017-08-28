# -*- coding: utf-8 -*-
"""
Created on Fri Jun 02 14:46:36 2017

@author: Sean Chang
"""

import requests
import re
from bs4 import BeautifulSoup



def getHTMLText(url):
    #返回url对应的网页源代码
    try:
        r = requests.get(url)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""

def vedioaddress(url,num):
    n1 = 68629 #第一集的编号-1
    url = url + str(n1+num)
    return url

def getVedioAddress(url,num):
    #html = getHTMLText(vedioaddress(url,num))
    html = getHTMLText(url)
    soup = BeautifulSoup(html, 'html.parser') 
    a = soup.find_all('script')
    address= ''
    mstr= ''
    for i in a:
        if (i.string != None):
            mstr = mstr+str(i.string)
    address = re.search(r'((http|ftp|https)://).+\.mp4',mstr).group(0).split("'")[0]
    return address
        
        
def main():
    vediourl = getVedioAddress('http://10.22.1.18/show/index?id=',1)
    print(vediourl)
    r = requests.get(vediourl) 
    with open(vediourl.split('/')[-1], "wb") as code:
        code.write(r.content)

main()

