# -*- coding: utf-8 -*-
import csv
import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
def readXML(filename):
        # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    
    # 在集合中获取所有电影
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

# 读取训练集
def readtrain():
    with open('Train.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row for row in reader]
    content_train = [i[1] for i in column1[1:]] #第一列为文本内容，并去除列名
    opinion_train = [i[2] for i in column1[1:]] #第二列为类别，并去除列名
    print('训练集有 %s 条句子' % len(content_train))
    train = [content_train, opinion_train]
    return train


# 将utf8的列表转换成unicode
def changeListCode(b):
    a = []
    for i in b:
        a.append(i.decode('utf8'))
    return a


# 对列表进行分词并用空格连接
def segmentWord(cont):
    c = []
    for i in cont:
        a = list(jieba.cut(i))
        b = " ".join(a)
        c.append(b)
    return c

def create7Data(emotionlist,Ncount):
    rst = []
    for i in range(7):
        data = []
        label = []
        data = [e for e in emotionlist[i]]
        for j in range(7):
            if j!=i:
                data += [e for e in emotionlist[j]]
        label = [1]*len(emotionlist[i])
        label += [-1]*(Ncount - len(emotionlist[i]))
        rst.append([data,label])
    
    return rst

emotionlist,Ncount = readXML("NLPCC.xml")
rst = create7Data(emotionlist,Ncount)
# corpus = ["我 来到 北京 清华大学", "他 来到 了 网易 杭研 大厦", "小明 硕士 毕业 与 中国 科学院"]

content = segmentWord(rst[0][0])
opinion = rst[0][1]



# 计算权重
vectorizer = CountVectorizer()
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(content))  # 先转换成词频矩阵，再计算TFIDF值
print(tfidf.shape)


# 单独预测

word = vectorizer.get_feature_names()
weight = tfidf.toarray()
# 分类器
clf = MultinomialNB().fit(tfidf, opinion)
docs = [" ".join(list(jieba.cut("只能说熊猫族的小熊猫太可爱。。。。")))]
new_tfidf = tfidftransformer.transform(vectorizer.transform(docs))
predicted = clf.predict(new_tfidf)
print(predicted)


'''
# 训练和预测一体
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(C=0.99, kernel = 'linear'))])
text_clf = text_clf.fit(train_content, train_opinion)
predicted = text_clf.predict(test_content)
print 'SVC',np.mean(predicted == test_opinion)
print set(predicted)
#print metrics.confusion_matrix(test_opinion,predicted) # 混淆矩阵

'''

# 循环调参
'''
parameters = {'vect__max_df': (0.4, 0.5, 0.6, 0.7),'vect__max_features': (None, 5000, 10000, 15000),
              'tfidf__use_idf': (True, False)}
grid_search = GridSearchCV(text_clf, parameters, n_jobs=1, verbose=1)
grid_search.fit(content, opinion)
best_parameters = dict()
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

'''