'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''


import jieba
import numpy as np
import xml.dom.minidom
import random
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten,Activation,Dropout,Merge  
from keras.layers import Conv1D, MaxPooling1D, Embedding,GlobalMaxPooling1D
from keras.models import Model
from keras.models import Sequential
from gensim.models.keyedvectors import KeyedVectors
from keras.callbacks import EarlyStopping,ModelCheckpoint,Callback 
np.random.seed(1337)  

MAX_SEQUENCE_LENGTH = 70
MAX_NB_WORDS = 600000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
bwfile = open("badwordlist.txt",encoding='utf8')
bwlist = [line.strip('\n') for line in bwfile]
bwfile.close()
for bw in bwlist:
    jieba.add_word(bw)  
emo = ['like','fear','disgust','anger','surprise','sadness','happiness','none']
smallemo= ['angry/disgusted','happy/like','sad','afraid/surprised','other']
'''
stop_words = ["的", "一", "不", "在", "人", "有", "是", "为", "以", "于", "上", "他", "而",
            "后", "之", "来", "及", "了", "因", "下", "可", "到", "由", "这", "与", "也",
            "此", "但", "并", "个", "其", "已", "无", "小", "我", "们", "起", "最", "再",
            "今", "去", "好", "只", "又", "或", "很", "亦", "某", "把", "那", "你", "乃",
            "它","要", "将", "应", "位", "新", "两", "中", "更", "我们", "自己", "没有", "“", "”",
            "，", "（", "）", " ",'[',']',' ','~','。','!','：','、','/','…']
'''
stop_words = []
sw = open('stopwords.txt')
for line in sw:
    line = line.strip("\n")
    stop_words.append(line)
sw.close()
def segmentWord(cont):
    c = []
    for i in cont:
        a = list(jieba.cut(i,cut_all=True))
        b = " ".join(a)
        c.append(b)
    return c

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
    #smalldata[4]=data[7]
    for i,d in enumerate(smalldata):
        for ele in d:
            traindata.append(ele)
            trainlabel.append(i)
    return traindata,trainlabel


traindata,label = readTrain("Training data for Emotion Classification.xml")
#traindata,label = adjustData(traindata,label)
traindata = segmentWord(traindata)

testdata = []
tfile = open('withoutother.txt',encoding='utf8')
for line in tfile:
    testdata.append(line)
tfile.close()
testdata = segmentWord(testdata)
testlabel = []
labelfile = open('testlabelnoother.txt')
for l in labelfile:
    line = l.strip('\n')
    testlabel.append(smallemo.index(line))
labelfile.close()
traintexts = [[word for word in document.split() if word not in stop_words] for document in traindata]
testtexts = [[word for word in document.split() if word not in stop_words] for document in testdata]

word_vectors = KeyedVectors.load_word2vec_format('zhwiki_2017_03.sg_50d.word2vec', binary=False)
#word_vectors = Word2Vec(traintexts+testtexts, size=EMBEDDING_DIM, window=5, min_count=1)
#word_vectors.wv.save_word2vec_format('smallwv.txt',binary=False)
#word_vectors = KeyedVectors.load_word2vec_format('smallwv.txt', binary=False)

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


x_train = traindata[:] # 训练集
y_train = labels[:]# 训练集的标签
x_val =  testdata[:]# 测试集，英文原意是验证集
y_val =  testlabels[:]
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
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(256,4,padding='valid',activation='relu',strides=1))
model.add(GlobalMaxPooling1D())
#model.add(Dropout(0.2)) 
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(4))
model.add(Activation('softmax'))
# 优化器我这里用了adadelta，也可以使用其他方法
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# =下面开始训练，nb_epoch是迭代次数，可以高一些，训练效果会更好，但是训练会变慢
#early_stopping =EarlyStopping(monitor='val_loss', patience=2) 
#checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
model.fit(x_train, y_train,batch_size=64,epochs=4,validation_data=(x_val,y_val))
pre = model.predict_classes(x_val,verbose=0,batch_size=64)

prelabel = []
predlabelfile = open("predfile.txt",'w+')
for p in pre:
    predlabelfile.write(str(smallemo[p])+'\n')
predlabelfile.close()
