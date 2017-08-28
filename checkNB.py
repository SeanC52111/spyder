# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:53:14 2017

@author: Sean Chang
"""

import rankNB
import jieba
import random
import xml.dom.minidom
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
emo = ['like','fear','disgust','anger','surprise','sadness','happiness']
def segmentWord(cont):
    c = []
    for i in cont:
        a = list(jieba.cut(i))
        b = " ".join(a)
        c.append(b)
    return c
def readTest(filename):
    testdata = []
    testlabel = []
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    weibos = collection.getElementsByTagName("weibo")
    for weibo in weibos:
        emotion1 = weibo.getAttribute("emotion-type1")
        emotion2 = weibo.getAttribute("emotion-type2")
        sen = ""
        if emotion1 in emo:
            sentence = weibo.getElementsByTagName('sentence')
            for e in sentence:
                sen += e.childNodes[0].data
            testdata.append(sen)
            label1 = emo.index(emotion1)
            testlabel.append(label1)
            if emotion2 in emo:
                label2 = emo.index(emotion2)
                testdata.append(sen)
                testlabel.append(label2)
    return testdata,testlabel

emotionlist,Ncount = rankNB.readXML("Training data for Emotion Classification.xml")
emotionlist,Ncount = rankNB.adjustData(emotionlist)
traindata,label = rankNB.createData(emotionlist,Ncount)
traindata = segmentWord(traindata)
traindata = np.array(traindata)
label = np.array(label)
indices = np.arange(traindata.shape[0])
np.random.shuffle(indices)
traindata = traindata[indices]
labels = label[indices]

num_validation_samples = int(0.2 * traindata.shape[0])
testlist,testNcount = rankNB.readXML("NLPtest.xml")
testdata,testlabel = rankNB.createData(testlist,testNcount)
testdata = segmentWord(testdata)
x_train = traindata[:]
y_train = labels[:]
x_val = testdata
y_val = testlabel

vectorizer = CountVectorizer()
tfidf = vectorizer.fit_transform(x_train)  
clf = svm.SVC()
clf.fit(tfidf, y_train)
new_tfidf = vectorizer.transform(x_val)
predicted = clf.predict(new_tfidf)
print(accuracy_score(y_val,predicted))
