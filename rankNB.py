# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 06:41:17 2017

@author: Sean Chang
"""

import xml.dom.minidom
import jieba
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

stop_words = ["的", "一", "不", "在", "人", "有", "是", "为", "以", "于", "上", "他", "而",
            "后", "之", "来", "及", "了", "因", "下", "可", "到", "由", "这", "与", "也",
            "此", "但", "并", "个", "其", "已", "无", "小", "我", "们", "起", "最", "再",
            "今", "去", "好", "只", "又", "或", "很", "亦", "某", "把", "那", "你", "乃",
            "它","要", "将", "应", "位", "新", "两", "中", "更", "我们", "自己", "没有", "“", "”",
            "，", "（", "）", " ",'[',']',' ','~','。','!']


emo = ['like','fear','disgust','anger','surprise','sadness','happiness']

def readXML(filename):
        # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    
    
    weibos = collection.getElementsByTagName("weibo")
    
    Ncount = 0 #count of all the class
    sen_lst = []
    for weibo in weibos:
       sentence = weibo.getElementsByTagName('sentence')
       for e in sentence:
           lst = []
           if e.getAttribute('opinionated')=='Y':
               lst.append(e.childNodes[0].data)
               emotion1 = e.getAttribute('emotion-1-type')
               Ncount = Ncount + 1
               emotion2 = e.getAttribute('emotion-2-type')
               if emotion2 != 'none':
                   Ncount = Ncount + 1
               lst.append([emotion1,emotion2])
               sen_lst.append(lst)
    
    
    emotionlist = [[],[],[],[],[],[],[]]
    
    for e in sen_lst:
        if 'like' in e[1]:
            emotionlist[0].append(e[0])
        if 'fear' in e[1]:
            emotionlist[1].append(e[0])
        if 'disgust' in e[1]:
            emotionlist[2].append(e[0])
        if 'anger' in e[1]:
            emotionlist[3].append(e[0])
        if 'surprise' in e[1]:
            emotionlist[4].append(e[0])
        if 'sadness' in e[1]:
            emotionlist[5].append(e[0])
        if 'happiness' in e[1]:
            emotionlist[6].append(e[0])
    
    
    return emotionlist,Ncount


def readTestXML(filename):
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
            if emotion2 in emo:
                label2 = emo.index(emotion2)
            else:
                label2 = -1
            testlabel.append([label1,label2])
    return testdata,testlabel

def lcm(x,y): # very fast
    s = x*y
    while y: x, y = y, x%y
    return s/x

#solve the bias distribution problem
def adjustData(emotionlist):
    l = []
    for e in emotionlist:
        l.append(len(e))
    m = max(l)
    maxindex = l.index(m)
    Ncount= 0
    for i in range(len(emotionlist)):
        if maxindex == i:
            #emotionlist[i] *= 2
            Ncount += l[i]
        else:
            
            emotionlist[i] *= (m//l[i])
            Ncount += l[i]*(m//l[i])
    
    return emotionlist,Ncount  
        

def createData(emotionlist,Ncount):
    data = []
    label = []
    for i in range(len(emotionlist)):
        for e in emotionlist[i]:
            data.append(e)
            label.append(i)
    return data,label

def segmentWord(cont):
    c = []
    for i in cont:
        a = list(jieba.cut(i))
        b = " ".join(a)
        c.append(b)
    return c

def showresult(rst):
    c = ['like','fear','disgust','anger','surprise','sadness','happiness']
    rs = sorted(rst)
    max1 = rst.index(rs[-1])
    max2 = rst.index(rs[-2])
    return c[max1],c[max2]


def rank(lst):
    sortl = sorted(lst,reverse=False)
    r = []
    for e in lst:
        r.append(sortl.index(e))
    return r
            
def train(data,label):
    vectorizer = CountVectorizer()
    content = segmentWord(data)
    opinion = label
    tfidf = vectorizer.fit_transform(content)  
    clf = MultinomialNB()
    clf.fit(tfidf, opinion)
    return clf,vectorizer

def test(sentence,clf,vectorizer):
    docs = [" ".join(list(jieba.cut(sentence)))]
    new_tfidf = vectorizer.transform(docs)
    predicted = clf.predict_log_proba(new_tfidf)
    return predicted[0]

if __name__ == "__main__":
    '''
    emotionlist,Ncount = readXML("NLPCC.xml")
    emotionlist,Ncount = adjustData(emotionlist)
    
    data,label = createData(emotionlist,Ncount)
    clf,vectorizer = train(data,label)
    print(rank(test("老鼠怕猫？",clf,vectorizer)))
    '''
    testdata,testlabel = readTestXML("NLPtest.xml")

        
        
            











