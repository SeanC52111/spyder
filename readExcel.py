# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 10:03:17 2017

@author: Sean Chang
"""

import xlrd



emodic = {"PA":6,"PE":6,"PD":0,"PH":0,"PG":0,"PB":0,"PK":0,"NA":3,"NB":5,"NJ":5 \
          ,"NH":5,"PF":5,"NI":1,"NC":1,"NG":1,"NE":2,"ND":2,"NN":2,"NK":2,"NL":2,"PC":4}

def EmoLexicon(filename):
    data = xlrd.open_workbook(filename)
    table = data.sheets()[0]
    nrows = table.nrows
    emolist = []
    emoindex = []
    for i in range(1,nrows):
        emolist.append(table.row_values(i)[0])
        emoindex.append(emodic[table.row_values(i)[4].strip()])
    
    return emolist,emoindex

if __name__ == "__main__":
    emolist,emoindex = EmoLexicon('emotion ontology.xlsx')