# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:43:04 2017

@author: Sean Chang
"""

import rankNB
import xml.dom.minidom
import random
import numpy as np
emo = ['like','fear','disgust','anger','surprise','sadness','happiness','none']
'''
emotionlist,Ncount = rankNB.readXML("NLPtest.xml")
for e in emotionlist:
    print(len(e))


testdata = []
tfile = open('test.txt')
for line in tfile:
    testdata.append(line)


pre = [2,2,3,6,0,4,4]
predlabelfile = open("predlabelfile.txt",'w')
for p in pre:
    predlabelfile.write(str(emo[p])+'\n')

testlabel = []
labelfile = open('testlabel.txt')
for l in labelfile:
    line = l.strip('\n')
    if line in emo:
        testlabel.append(emo.index(line))
    else:
        testlabel.append(7)
labelfile.close()

for i,val in enumerate(testlabel):
    print(i,val)

bwfile = open("badwordlist.txt",encoding='utf8')
bwlist = [line.strip('\n') for line in bwfile]
def readTrain(filename):
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    
    weibos = collection.getElementsByTagName("weibo")
    traindata = []
    trainlabel = []
    for weibo in weibos:
        sentence = weibo.getElementsByTagName('sentence')
        for e in sentence:
            sen = e.childNodes[0].data
            if e.getAttribute('opinionated')=='Y':
                emotion1 = e.getAttribute('emotion-1-type')
                emotion2 = e.getAttribute('emotion-2-type')
                traindata.append(sen)
                trainlabel.append(emo.index(emotion1))
                if emotion2 != 'none':
                    traindata.append(sen)
                    trainlabel.append(emo.index(emotion2))
            else:
                traindata.append(sen)
                trainlabel.append(7)
    return traindata,trainlabel
def readBWTrain(filename):
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    
    weibos = collection.getElementsByTagName("weibo")
    traindata = []
    trainlabel = []
    for weibo in weibos:
        sentence = weibo.getElementsByTagName('sentence')
        for e in sentence:
            sen = e.childNodes[0].data
            for bw in bwlist:
                if sen.find(bw)!=-1:
                    if e.getAttribute('opinionated')=='Y':
                        emotion1 = e.getAttribute('emotion-1-type')
                        emotion2 = e.getAttribute('emotion-2-type')
                        traindata.append(sen)
                        trainlabel.append(emo.index(emotion1))
                        if emotion2 != 'none':
                            traindata.append(sen)
                            trainlabel.append(emo.index(emotion2))
                    else:
                        traindata.append(sen)
                        trainlabel.append(7)
    return traindata,trainlabel        
def adjustData(data,label):
    newdata = []
    newlabel = []
    rst = [[],[],[],[],[],[],[],[]]
    for index,d in enumerate(data):
        rst[label[index]].append(d)
    l = []
    for r in rst:
        l.append(len(r))
    maxl = max(l)
    for i in range(len(rst)):
        rst[i] *= maxl//l[i]
    print(rst)
    for i in range(len(rst)):
        for e in rst[i]:
            newdata.append(e)
            newlabel.append(i)
    return newdata,newlabel      

               
traindata,label = readBWTrain("Training data for Emotion Classification.xml")
print(len(traindata))
traindata,label = adjustData(traindata,label)
print(len(traindata))

testdata = []
tfile = open('weibo.txt',encoding='utf8')
for line in tfile:
    testdata.append(line)

def readTrain(filename):
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    
    weibos = collection.getElementsByTagName("weibo")
    traindata = []
    trainlabel = []
    data = [[],[],[],[],[],[],[],[]]
    for weibo in weibos:
        sentence = weibo.getElementsByTagName('sentence')
        for e in sentence:
            sen = e.childNodes[0].data
            if e.getAttribute('opinionated')=='Y':
                emotion1 = e.getAttribute('emotion-1-type')
                emotion2 = e.getAttribute('emotion-2-type')
                data[emo.index(emotion1)].append(sen)
                if emotion2 != 'none':
                    data[emo.index(emotion2)].append(sen)        
            else:
                data[7].append(sen)
    
    data[7] = random.sample(data[7], 3000)
    for i,d in enumerate(data):
        for ele in d:
            traindata.append(ele)
            trainlabel.append(i)
    return traindata,trainlabel

d,l = readTrain("Training data for Emotion Classification.xml")

def readTrain(filename):
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    
    weibos = collection.getElementsByTagName("weibo")
    traindata = []
    trainlabel = []
    data = [[],[],[],[],[],[],[],[]]
    for weibo in weibos:
        sentence = weibo.getElementsByTagName('sentence')
        for e in sentence:
            sen = e.childNodes[0].data
            if e.getAttribute('opinionated')=='Y':
                emotion1 = e.getAttribute('emotion-1-type')
                emotion2 = e.getAttribute('emotion-2-type')
                data[emo.index(emotion1)].append(sen)
                if emotion2 != 'none':
                    data[emo.index(emotion2)].append(sen)        
            else:
                data[7].append(sen)
    smalldata = [[],[],[],[],[]]
    smalldata[0]=random.sample(data[2]+data[3],1000)
    smalldata[1]=random.sample(data[0]+data[6],1000)
    smalldata[2]=random.sample(data[5],1000)
    smalldata[3]=random.sample(data[1]+data[4],1000)
    smalldata[4]=random.sample(data[7],1000)
    for i,d in enumerate(smalldata):
        for ele in d:
            traindata.append(ele)
            trainlabel.append(i)
    return traindata,trainlabel
d,l = readTrain("Training data for Emotion Classification.xml")

stop_words = []
sw = open('stopwords.txt')
for line in sw:
    line = line.strip("\n")
    stop_words.append(line)
'''
def readTrain(filename):
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    
    weibos = collection.getElementsByTagName("weibo")
    traindata = []
    trainlabel = []
    data = [[],[],[],[],[],[],[],[]]
    for weibo in weibos:
        sentence = weibo.getElementsByTagName('sentence')
        for e in sentence:
            sen = e.childNodes[0].data
            if e.getAttribute('opinionated')=='Y':
                emotion1 = e.getAttribute('emotion-1-type')
                emotion2 = e.getAttribute('emotion-2-type')
                data[emo.index(emotion1)].append(sen)
                if emotion2 != 'none':
                    data[emo.index(emotion2)].append(sen)        
            else:
                data[7].append(sen)
    #smalldata = [[],[],[],[],[]]
    smalldata = [[],[],[],[]]
    smalldata[0]=data[2]+data[3]
    smalldata[1]=data[0]+data[6]
    smalldata[2]=data[5]
    smalldata[3]=data[1]+data[4]
    smalldata[3] *= 2
    for sm in smalldata:
        print(len(sm))
    #smalldata[4]=data[7]
    for i,d in enumerate(smalldata):
        for ele in d:
            traindata.append(ele)
            trainlabel.append(i)
    return traindata,trainlabel

traindata,trainlabel = readTrain("Training data for Emotion Classification.xml")
