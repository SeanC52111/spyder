# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:01:40 2017

@author: Sean Chang
"""

def ispalindrome1(string):
    length = len(string)
    i = 0
    j = length - 1
    while i<j:
        if(string[i] != string[j]):
            return False
        i = i+1
        j = j-1
    return True

def ispalindrome2(string):
    length = len(string)
    middle = (length+1)//2
    if length%2 == 0:
        i = middle - 1
        j = i + 1
    else:
        i = middle - 2
        j = middle
    while i>=0:
        if(string[i] != string[j]):
            return False
        i = i - 1
        j = j + 1
    return True
    
print(ispalindrome2("abcda"))