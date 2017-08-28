# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 06:28:52 2017

@author: Sean Chang
"""

import binaryNB
import rankNB
import jieba
import readExcel
import random 



if __name__ == "__main__":
    emo = ['like','fear','disgust','anger','surprise','sadness','happiness']
    #read the train document
    emotionlist,Ncount = rankNB.readXML("NLPCC.xml")
    #emotionlist,Ncount = rankNB.adjustData(emotionlist)
    #adjust the train data for average distribution
    #emotionlist,Ncount = parseXML.adjustData(emotionlist)
    #load emotion lexicon
    print("load emotion lexicon")
    emolist,emoindex = readExcel.EmoLexicon('emotion ontology.xlsx')
    
    data,label = rankNB.createData(emotionlist,Ncount)
    #train the data for first ranking
    clf,vectorizer = rankNB.train(data,label)
    
        
    #merge the 7 binary classification train data
    rst7data = binaryNB.create7Data(emotionlist,Ncount)
    
    '''
    testdata,testlabel = rankNB.readTestXML("NLPtest.xml")
    
    
    #randomly choose 20 sentence for testing
    randomlen = testdata.__len__()  
    indexList = range(randomlen)  
    randomIndex = random.sample(indexList, 100)  
    
    test_data = []
    test_label = []
    for i in randomIndex:  
        test_data.append(testdata[i])
        test_label.append(testlabel[i])
    '''
    #sen_counter :the ith sentence
    sen_counter = 0
    #correct number
    correct = 0   
    
    test_data=['生气生气生气吃惊吃惊吃惊！']
    test_label = [[3,4]]
    
    for string in test_data:
        print("calculating sentence",sen_counter+1)
        print(string)
        docs = [" ".join(list(jieba.cut(string)))]
        #merge the first rank list by testing
        r = rankNB.rank(rankNB.test(string,clf,vectorizer))
        print(r)
        
        #updating ranking 
        print("updating ranking")
        #7 binary classification for updating
        for i in range(7):
            r[i] += binaryNB.testSentence(rst7data[i],docs)
            print(r)
        
        #cut the test sentence
        word_list = jieba.cut(string,cut_all=True)
        #update the corresponding index in rank []
        for word in word_list:
            if word in emolist:
                i = emolist.index(word)
                r[emoindex[i]] += 1
        print(r)
        
        # find the top 2 emotion:
        copyr = [e for e in r]
        m1 = max(copyr)
        index1 = r.index(m1)
        del copyr[index1]
        m2 = max(copyr)
      
        if m1 != m2:
            index2 = r.index(m2)
        else:
            r[index1] += 1
            index2 = r.index(m2)
            r[index1] -= 1
        l = test_label[sen_counter]
        if -1 not in l:
            if (index1 in l) and (index2 in l):
                correct += 1
        else:
            if (index1 in l) or (index2 in l):
                correct += 1
        sen_counter += 1
        print("now correct:",correct)
    
    print("precision is:",correct/len(test_data))
        
        
    

        
        
            











