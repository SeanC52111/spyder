# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:26:07 2017

@author: Sean Chang
"""

#!/usr/bin/env python  
from random import randint, choice  
from bisect import bisect_left  
from collections import deque  
  
#异常类  
class InitError(Exception):  
    pass  
class ParaError(Exception):  
    pass  
      
#生成键值对  
class KeyValue(object):  
    __slots__=('key', 'value')  
    def __init__(self, key, value):  
        self.key=int(key) #一定要保证键值是整型  
        self.value=value  
          
    def __str__(self):  
        return str((self.key, self.value))  
      
    def __cmp__(self, key):  
        if self.key>key:  
            return 1  
        elif self.key < key:  
            return -1  
        else:  
            return 0  
      
    def __lt__(self, other):  
        if (type(self) == type(other)):  
            return self.key < other.key;  
        else:  
            return int(self.key) < int(other);  
              
    def __eq__(self, other):  
        if (type(self) == type(other)):  
            return self.key == other.key;  
        else:  
            return int(self.key) == int(other);  
  
    def __gt__(self, other):  
        return not self < other;     
  
          
              
#预留文件读写操作，暂未实现  
class IndexFile(object):  
    def __init__(fname,cellsize):  
        f=open(fname,'wb')  
        f.close()  
        self.name=fname  
        self.cellsize=cellsize  
    def write_obj(obj,pos):  
        pass  
    def read_obj(obj,pos):  
        pass          
  
#B树实现  
class Btree(object):  
    class __BtreeNode(object):  
        def __init__(self, t, parent=None):  
            if not isinstance(t,int):  
                raise InitError('degree of Btree must be int type')  
            if t<2:  
                raise InitError('degree of Btree must be equal or greater then 2')  
            else:  
                self.vlist=[] #存放数据点  
                self.clist=[] #存放数据区间  
                self.parent=parent  
                self.__degree=t #度数限制了容量，也就是枝条上叶子的数量范围  
        @property  
        def degree(self):  
            return self.__degree  
        def isleaf(self):  
            return len(self.clist)==0  
  
        #遍历  
        def traversal(self):  
            result=[]  
            def get_value(n):  
                if n.clist==[]:  
                    result.extend(n.vlist)  
                else:  
                    for i,v in enumerate(n.vlist):  
                        get_value(n.clist[i])  
                        result.append(v)  
                    get_value(n.clist[-1])  
            get_value(self)  
            return result  
          
        #显示节点信息  
        def show(self):  
            q=deque()  
            h=0  
            q.append([self,h])  
            while True:  
                try:  
                    w,hei=q.popleft()  
                except IndexError:  
                    return  
                else:  
                    print([(v.key,v.value) for v in w.vlist],'the height is',hei)  
                    if w.clist==[]:  
                        continue  
                    else:  
                        if hei==h:  
                            h+=1  
                        q.extend([[v,h] for v in w.clist])  
  
        #最大和最小值               
        def getmax(self):  
            n=self  
            while not n.isleaf():  
                n=n.clist[-1]  
            return (n.vlist[-1],n)  
          
        def getmin(self):  
            n=self  
            while not n.isleaf():  
                n=n.clist[0]  
            return (n.vlist[0],n)  
  
    #初始化          
    def __init__(self,t):  
        self.__degree=t  
        self.__root=Btree.__BtreeNode(t)  
         
    @property  
    def degree(self):  
        return self.__degree  
  
    #深度优先遍历  
    def traversal(self):  
        """ 
        use dfs to search a btree's node 
        """  
        return self.__root.traversal()  
  
    #广度优先遍历  
    def show(self):  
        """ 
        use bfs to show a btree's node and its height 
        """  
        return self.__root.show()  
  
    #搜索  
    def search(self,mi=None,ma=None):  
        """ 
        search key-value pair within range mi<=key<=ma. 
        if mi or ma is not specified,the searching range 
        is key>=mi or key <=ma 
        """  
        result=[]  
        node=self.__root  
        if mi is None or ma is None:  
            raise ParaError('you need setup searching range')  
        elif mi > ma:  
            raise ParaError('upper bound must be greater or equal than lower bound')  
        def search_node(n):  
            if mi is None:  
                if not n.isleaf():  
                    for i,v in enumerate(n.vlist):  
                        if v.key<=ma:  
                            result.extend(n.clist[i].traversal())  
                            result.append(v)  
                        else:  
                            search_node(n.clist[i])  
                            return  
                    search_node(n.clist[-1])  
                else:  
                    for v in n.vlist:  
                        if v.key<=ma:  
                            result.append(v)  
                        else:  
                            break  
            elif ma is None:  
                if not n.isleaf():  
                    for i,v in enumerate(n.vlist):  
                        if v.key<mi:  
                            continue  
                        else:  
                            search_node(n.clist[i])  
                            while i<len(n.vlist):  
                                result.append(n.vlist[i])  
                                result.extend(n.clist[i+1].traversal())  
                                i+=1  
                            break  
                    if v.key<mi:  
                        search_node(n.clist[-1])  
                else:  
                    for v in n.vlist:  
                        if v.key>=mi:  
                            result.append(v)  
            else:  
                if not n.isleaf():  
                    for i,v in enumerate(n.vlist):  
                        if v.key<mi:  
                            continue  
                        elif mi<=v.key <=ma:  
                            search_node(n.clist[i])  
                            result.append(v)  
                        elif v.key>ma:  
                            search_node(n.clist[i])  
                    if v.key<=ma:  
                        search_node(n.clist[-1])  
                else:  
                    for v in n.vlist:  
                        if mi<=v.key<=ma:  
                            result.append(v)  
                        elif v.key>ma:  
                            break  
        search_node(node)  
        return result  
  
    #插入  
    def insert(self,key_value):  
        node=self.__root  
        full=self.degree*2-1  
        mid=full//2+1 #此处需注意，可能会出错  
  
        #分裂，可以保证两部分具有相等长度  
        def split(n):  
            new_node=Btree.__BtreeNode(self.degree,parent=n.parent)  
            new_node.vlist=n.vlist[mid:]  
            new_node.clist=n.clist[mid:]  
            for c in new_node.clist:  
                c.parent=new_node  
            if n.parent is None:  
                newroot=Btree.__BtreeNode(self.degree)  
                newroot.vlist=[n.vlist[mid-1]]  
                newroot.clist=[n,new_node]  
                n.parent=new_node.parent=newroot  
                self.__root=newroot  
            else:  
                i=n.parent.clist.index(n)  
                n.parent.vlist.insert(i,n.vlist[mid-1])  
                n.parent.clist.insert(i+1,new_node)  
            n.vlist=n.vlist[:mid-1]  
            n.clist=n.clist[:mid]  
            return n.parent  
        def insert_node(n):  
            if len(n.vlist)==full:  
                insert_node(split(n))  
            else:  
                if n.vlist==[]:  
                    n.vlist.append(key_value)  
                else:  
                    if n.isleaf():  
                        p=bisect_left(n.vlist,key_value)  #locate insert point in ordered list vlist  
                        n.vlist.insert(p,key_value)  
                    else:  
                        p=bisect_left(n.vlist,key_value)  
                        insert_node(n.clist[p])  
        insert_node(node)  
  
    #删除  
    def delete(self,key_value):  
        node=self.__root  
        mini=self.degree-1  
        #归并  
        def merge(n,i):  
            n.clist[i].vlist=n.clist[i].vlist+[n.vlist[i]]+n.clist[i+1].vlist  
            n.clist[i].clist=n.clist[i].clist+n.clist[i+1].clist  
            n.clist.remove(n.clist[i+1])  
            n.vlist.remove(n.vlist[i])  
            if n.vlist==[]:  
                n.clist[0].parent=None  
                self.__root=n.clist[0]  
                del n  
                return self.__root  
            else:  
                return n  
  
        #左区让最大值到右区  
        def tran_l2r(n,i):  
            left_max,left_node=n.clist[i].getmax()  
            right_min,right_node=n.clist[i+1].getmin()  
            right_node.vlist.insert(0,n.vlist[i])  
            del_node(n.clist[i],left_max)  
            n.vlist[i]=left_max  
  
        #右区让最小值到左区  
        def tran_r2l(n,i):  
            left_max,left_node=n.clist[i].getmax()  
            right_min,right_node=n.clist[i+1].getmin()  
            left_node.vlist.append(n.vlist[i])  
            del_node(n.clist[i+1],right_min)  
            n.vlist[i]=right_min  
        def del_node(n,kv):  
            p=bisect_left(n.vlist,kv)  
            if not n.isleaf():  
                if p==len(n.vlist):  
                    if len(n.clist[-1])>mini:  
                        del_node(n.clise[p],kv)  
                    elif len(n.clist[p-1])>mini:  
                        tran_l2r(n,p-1)  
                        del_node(n.clist[-1],kv)  
                    else:  
                        del_node(merge(n,p-1),kv)  
                else:  
                    if n.vlist[p]==kv:  
                        left_max,left_node=n.clist[p].getmax()  
                        if len(n.clist[p].vlist)>mini:  
                            del_node(n.clist[p],left_max)  
                            n.vlist[p]=left_max  
                        else:  
                            right_min,right_node=n.clist[p+1].getmin()  
                            if len(n.clist[p+1].vlist)>mini:  
                                del_node(n.clist[p+1],right_min)  
                                n.vlist[p]=right_min  
                            else:  
                                del_node(merge(n,p),kv)  
                    else:  
                        if len(n.clist[p].vlist)>mini:  
                            del_node(n.clist[p],kv)  
                        elif len(n.clist[p+1].vlist)>mini:  
                            tran_r2l(n,p)  
                            del_node(n.clist[p],kv)  
                        else:  
                            del_node(merge(n,p),kv)  
            else:  
                try:  
                    pp=n.vlist[p]  
                except IndexError:  
                    return -1  
                else:  
                    if pp!=kv:  
                        return -1  
                    else:  
                        n.vlist.remove(kv)  
                        return 0  
        del_node(node,key_value)  
  
def test():  
    #初始化数据源  
    mini=50  
    maxi=200  
    testlist=[]  
    for i in range(50):  
        key=randint(1,1000)  
        #key=i  
        value=choice(['Do', 'Re', 'Mi', 'Fa', 'So', 'La', 'Si'])  
        testlist.append(KeyValue(key,value))  
  
    #初始化B树  
    mybtree=Btree(4)  
  
    #插入操作  
    for x in testlist:  
        mybtree.insert(x)  
    print('my btree is:\n')  
    mybtree.show()  
  
    #删除操作  
    mybtree.delete(testlist[0])  
  
    print('\n删除 {0}后， the newtree is:\n'.format(testlist[0]));  
    mybtree.show()  
  
      
    #查找操作  
    print('\nnow we are searching item between %d and %d\n==>'%(mini,maxi))  
    print([v.key for v in mybtree.search(mini,maxi)])  
  
    #深度优先遍历  
    print('\nDFS遍历B树：');  
    for x in mybtree.traversal():  
        print(x)  
      
  
if __name__ == '__main__':  
    test();  