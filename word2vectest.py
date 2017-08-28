'''This example demonstrates the use of Convolution1D for text classification.
Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''

import rankNB
import jieba
import xml.dom.minidom
import random
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding,GlobalMaxPooling1D
from keras.models import Sequential
from gensim.models.keyedvectors import KeyedVectors
from keras.models import load_model
np.random.seed(1337)  # For Reproducibility
batch_size = 32
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 50 # 每个文本的最长选取长度，较短的文本可以设短些
MAX_NB_WORDS = 600000 # 整体词库字典中，词的多少，可以略微调大或调小
emo = ['like','fear','disgust','anger','surprise','sadness','happiness','none']
emodict = {'like':['like','happiness'],'fear':['fear','surprise'],'disgust':['disgust','anger'],'anger':['disgust','anger'],\
           'surprise':['fear','surprise'],'sadness':['sadness'],'happiness':['like','happiness']}
stop_words = ["的", "一", "不", "在", "人", "有", "是", "为", "以", "于", "上", "他", "而",
            "后", "之", "来", "及", "了", "因", "下", "可", "到", "由", "这", "与", "也",
            "此", "但", "并", "个", "其", "已", "无", "小", "我", "们", "起", "最", "再",
            "今", "去", "好", "只", "又", "或", "很", "亦", "某", "把", "那", "你", "乃",
            "它","要", "将", "应", "位", "新", "两", "中", "更", "我们", "自己", "没有", "“", "”",
            "，", "（", "）", " ",'[',']',' ','~','。','!','：','、','/','…']

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
    
    data[7] = random.sample(data[7], 10000)
    for i,d in enumerate(data):
        for ele in d:
            traindata.append(ele)
            trainlabel.append(i)
    return traindata,trainlabel
def text_to_index_array(p_new_dic, p_sen):  # 文本转为索引数字模式
    new_sentences = []
    for sen in p_sen:
        new_sen = []
        for word in sen:
            try:
                new_sen.append(p_new_dic[word])  # 单词转索引数字
            except:
                new_sen.append(0)  # 索引字典里没有的词转为数字0
        new_sentences.append(new_sen)

    return new_sentences
'''
emotionlist,Ncount = rankNB.readXML("Training data for Emotion Classification.xml")
emotionlist,Ncount = rankNB.adjustData(emotionlist)
traindata,label = rankNB.createData(emotionlist,Ncount)
traindata = segmentWord(traindata)

#testdata,testlabel = readTest('NLPCC.xml')

testlist,testNcount = rankNB.readXML("NLPtest.xml")
testdata,testlabel = rankNB.createData(testlist,testNcount)
testdata = segmentWord(testdata)
'''

'''
traindata,label = readTest("NLPtest.xml")
traindata = segmentWord(traindata)
testdata,testlabel = readTest('NLPCC.xml')
testdata = segmentWord(testdata)
'''

testdata = []
tfile = open('test.txt')
for line in tfile:
    testdata.append(line)
testdata = segmentWord(testdata)

testlabel = []
labelfile = open('testlabel.txt')
for l in labelfile:
    line = l.strip('\n')
    if line in emo:
        testlabel.append(emo.index(line))
    else:
        testlabel.append(7)
labelfile.close()


traindata,label = readTrain("Training data for Emotion Classification.xml")
#traindata,label = adjustData(traindata,label)
traindata = segmentWord(traindata)
'''
testdata = []
tfile = open('weibo.txt',encoding='utf8')
for line in tfile:
    testdata.append(line)
testdata = segmentWord(testdata)
'''
traintexts = [[word for word in document.split() if word not in stop_words] for document in traindata]
testtexts = [[word for word in document.split() if word not in stop_words] for document in testdata]

word_vectors = KeyedVectors.load_word2vec_format('zhwiki_2017_03.sg_50d.word2vec', binary=False)


gensim_dict = Dictionary()
gensim_dict.doc2bow(word_vectors.vocab.keys(), allow_update=True)
w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
w2vec = {word: word_vectors[word] for word in w2indx.keys()}
trainseq = text_to_index_array(w2indx, traintexts)
testseq = text_to_index_array(w2indx, testtexts)

traindata = pad_sequences(trainseq, maxlen=MAX_SEQUENCE_LENGTH)
testdata = pad_sequences(testseq, maxlen=MAX_SEQUENCE_LENGTH)
word_index = w2indx
print('Found %s unique tokens.' % len(word_index))
labels = to_categorical(np.asarray(label))
testlabels = to_categorical(np.asarray(testlabel))
indices = np.arange(traindata.shape[0])
np.random.shuffle(indices)
traindata = traindata[indices]
labels = labels[indices]

num_validation_samples = int(VALIDATION_SPLIT * traindata.shape[0])
'''
x_train = traindata[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = traindata[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
'''

x_train = traindata[:] # 训练集
y_train = labels[:]# 训练集的标签
x_val =  testdata[:]# 测试集，英文原意是验证集
y_val =  testlabels[:]# 测试集的标签

print('Preparing embedding matrix.')
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = w2vec.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector # word_index to word_embedding_vector ,<20000(nb_words)

# load pre-trained word embeddings into an Embedding layer
# 神经网路的第一层，词向量层，本文使用了预训练glove词向量，可以把trainable那里设为False
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix],
                            trainable=False)
print('Training model.')

# train a 1D convnet with global maxpoolinnb_wordsg

#left model 第一块神经网络，卷积窗口是5*50（50是词向量维度）
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(250,
                 5,
                 padding='valid',
                 activation='relu',
                 strides=1))

model.add(GlobalMaxPooling1D())

model.add(Dense(250))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('sigmoid'))
# 优化器我这里用了adadelta，也可以使用其他方法
model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

# =下面开始训练，nb_epoch是迭代次数，可以高一些，训练效果会更好，但是训练会变慢
model.fit(x_train, y_train,nb_epoch=10,batch_size=64,validation_data=(x_val,y_val))

#model.save('my_model15.h5')

#model = model = load_model('my_model.h5')
'''
score = model.evaluate(x_train, y_train, verbose=0) # 评估模型在训练集中的效果，准确率约99%
print('train score:', score[0])
print('train accuracy:', score[1])
score = model.evaluate(x_val, y_val, verbose=0)  # 评估模型在测试集中的效果，准确率约为97%，迭代次数多了，会进一步提升
print('Test score:', score[0])
print('Test accuracy:', score[1])
'''

pre = model.predict_classes(x_val,verbose=0,batch_size=64)

prelabel = []
predlabelfile = open("predfile.txt",'w+')
for p in pre:
    predlabelfile.write(str(emo[p])+'\n')
predlabelfile.close()
