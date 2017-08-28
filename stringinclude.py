# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 21:45:05 2017

@author: Sean Chang
"""

def stringinclude(str1,str2):
    dic1 = {}
    for i in str1:
        if (dic1.get(i)==None):
            dic1[i]=1
        else:
            dic1[i]=dic1[i]+1
    for j in str2:
        if dic1.get(j)==None:
            return False
    return True

def main():
    str1='ABCD'
    str2='BCE'
    print(stringinclude(str1,str2))

main()
    
    