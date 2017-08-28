# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

##---(Thu May 18 09:34:16 2017)---
runfile('C:/Users/Sean Chang/.spyder/temp.py', wdir='C:/Users/Sean Chang/.spyder')
import numpy as np
runfile('C:/Users/Sean Chang/.spyder/temp.py', wdir='C:/Users/Sean Chang/.spyder')

##---(Sat May 20 09:06:00 2017)---
runfile('C:/Users/Sean Chang/.spyder/temp.py', wdir='C:/Users/Sean Chang/.spyder')

##---(Sat May 20 10:51:32 2017)---
runfile('C:/Users/Sean Chang/.spyder/temp.py', wdir='C:/Users/Sean Chang/.spyder')
print '\xe8\xbe\xbd\xe5\xae\x81'
import numpy as np
from sklearn.cluster import KMeans
def loadData(filePath):
    fr = open(filePath,'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1,len(items))])
    return retData,retCityName
if __name__ == '__main__':
    data,cityName = loadData('city.txt')
    km = KMeans(n_clusters = 3)
    label = km.fit_predict(data)
    expenses = np.sum(km.cluster_centers_,axis=1)
    CityCluster = [[],[],[]]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print "Expenses:%.2f" % expenses[i]
        print CityCluster[i]
import numpy as np
from sklearn.cluster import KMeans
def loadData(filePath):
    fr = open(filePath,'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1,len(items))])
    return retData,retCityName
import numpy as np
from sklearn.cluster import KMeans
def loadData(filePath):
    fr = open(filePath,'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1,len(items))])
    return retData,retCityName
print
print(3+2)
import numpy as np
from sklearn.cluster import KMeans
def loadData(filePath):
    fr = open(filePath,'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1,len(items))])
    return retData,retCityName
runfile('C:/Users/Sean Chang/.spyder/temp.py', wdir='C:/Users/Sean Chang/.spyder')
%clear
runfile('C:/Users/Sean Chang/.spyder/temp.py', wdir='C:/Users/Sean Chang/.spyder')
%clear
runfile('C:/Users/Sean Chang/.spyder/temp.py', wdir='C:/Users/Sean Chang/.spyder')
%clear
runfile('C:/Users/Sean Chang/.spyder/temp.py', wdir='C:/Users/Sean Chang/.spyder')
%clear
runfile('C:/Users/Sean Chang/.spyder/temp.py', wdir='C:/Users/Sean Chang/.spyder')
%clear
runfile('C:/Users/Sean Chang/.spyder/temp.py', wdir='C:/Users/Sean Chang/.spyder')

##---(Sat May 20 20:47:14 2017)---
runfile('C:/Users/Sean Chang/.spyder/iris.py', wdir='C:/Users/Sean Chang/.spyder')

##---(Tue May 23 18:02:35 2017)---
runfile('C:/Users/Sean Chang/.spyder/nmf.py', wdir='C:/Users/Sean Chang/.spyder')
import PIL

##---(Sat May 27 18:19:13 2017)---
runfile('C:/Users/Sean Chang/.spyder/taobaoprice.py', wdir='C:/Users/Sean Chang/.spyder')

##---(Wed May 31 19:28:02 2017)---
runfile('C:/Users/Sean Chang/.spyder/stock.py', wdir='C:/Users/Sean Chang/.spyder')

##---(Wed May 31 21:37:31 2017)---
runfile('C:/Users/Sean Chang/.spyder/stock.py', wdir='C:/Users/Sean Chang/.spyder')
with open('E:\\','a',encoding='utf-8')as f:    f.write(10)    
runfile('C:/Users/Sean Chang/.spyder/stock.py', wdir='C:/Users/Sean Chang/.spyder')

##---(Wed May 31 23:17:26 2017)---
runfile('C:/Users/Sean Chang/.spyder/music.py', wdir='C:/Users/Sean Chang/.spyder')

##---(Fri Jun 02 08:05:08 2017)---
runfile('C:/Users/Sean Chang/.spyder/imagecutting.py', wdir='C:/Users/Sean Chang/.spyder')

##---(Fri Jun 02 14:46:15 2017)---
runfile('C:/Users/Sean Chang/.spyder/youkuvedio.py', wdir='C:/Users/Sean Chang/.spyder')
from selenium import webdriver
runfile('C:/Users/Sean Chang/.spyder/youkuvedio.py', wdir='C:/Users/Sean Chang/.spyder')

##---(Fri Jun 02 21:04:51 2017)---
runfile('C:/Users/Sean Chang/.spyder/stockclassify.py', wdir='C:/Users/Sean Chang/.spyder')

##---(Fri Jun 02 21:52:39 2017)---
runfile('C:/Users/Sean Chang/numrecognize.py', wdir='C:/Users/Sean Chang')

##---(Fri Jun 02 22:11:16 2017)---
runfile('C:/Users/Sean Chang/.spyder/knn.py', wdir='C:/Users/Sean Chang/.spyder')
runfile('C:/Users/Sean Chang/.spyder/numclassify.py', wdir='C:/Users/Sean Chang/.spyder')

##---(Sat Jun 03 10:23:17 2017)---
runfile('C:/Users/Sean Chang/.spyder/youkuvedio.py', wdir='C:/Users/Sean Chang/.spyder')