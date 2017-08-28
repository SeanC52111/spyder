# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:23:17 2017

@author: Sean Chang
"""
import xml.dom.minidom
emo = ['like','fear','disgust','anger','surprise','sadness','happiness','none']
bwfile = open("badwordlist.txt",encoding='utf8')
bwlist = [line.strip('\n') for line in bwfile]
'''
s = "今天真是日了狗了！"
for bw in bwlist:
    if s.find(bw)!=-1:
        print(s) 
'''

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

traindata,trainlabel = readBWTrain('Training data for Emotion Classification.xml')