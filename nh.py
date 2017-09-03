# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 11:18:01 2017

@author: cezhang
"""

import hashlib

class MerkleNode(object):
    def __init__(self,rawdata=None,hashvalue=None,left=None,right=None,isleaf=None,ranlst=None,layer=0):
        self.rawdata=rawdata
        self.hashvalue=hashvalue
        self.left=left
        self.right=right
        self.isleaf=isleaf
        self.ranlst=ranlst
        self.layer=layer
        
def createLeafNodes(inputlist):
    inputlist = sorted(inputlist)
    rst = []
    for elem in inputlist:
        md5 = hashlib.md5()
        md5.update(str(elem).encode())
        d=md5.hexdigest()
        rst.append(MerkleNode(rawdata=str(elem),hashvalue=d,isleaf=True,ranlst=[elem],layer=0))
    return rst


def createTree(nodes):
    list_len = len(nodes)
    if list_len == 0:
        return 0
    else:
        secondary = []
        if list_len %2 == 0:
            for k in [nodes[x:x+2] for x in range(0,list_len,2)]:
                d1 = k[0].hashvalue.encode()
                d2 = k[1].hashvalue.encode()
                md5 = hashlib.md5()
                md5.update(d1+d2)
                newdata = md5.hexdigest()
                node = MerkleNode(rawdata=str(k[0].rawdata)+str(k[1].rawdata),hashvalue=newdata,\
                                  left=k[0],right=k[1],isleaf=False,\
                                  ranlst=[k[0].ranlst[0],k[1].ranlst[-1]],layer=k[0].layer+1)
                secondary.append(node)
            if len(secondary) == 1:
                return secondary[0]
            else:
                return createTree(secondary)
        else:
            for k in [nodes[x:x+2] for x in range(0,list_len-1,2)]:
                d1 = k[0].hashvalue.encode()
                d2 = k[1].hashvalue.encode()
                md5 = hashlib.md5()
                md5.update(d1+d2)
                newdata = md5.hexdigest()
                node = MerkleNode(rawdata=str(k[0].rawdata)+str(k[1].rawdata),hashvalue=newdata,\
                                  left=k[0],right=k[1],isleaf=False,\
                                  ranlst=[k[0].ranlst[0],k[1].ranlst[-1]],layer=k[0].layer+1)
                secondary.append(node)
            lastnode = MerkleNode(rawdata=str(nodes[-1].rawdata),hashvalue=nodes[-1].hashvalue,\
                                  left=nodes[-1],isleaf=False,ranlst=nodes[-1].ranlst,layer=nodes[-1].layer+1)
            secondary.append(lastnode)
            if len(secondary) == 1:
                return secondary[0]
            else:
                return createTree(secondary)
            
def bfs(root):
    print('start bfs')
    queue = []
    queue.append(root)
    while(len(queue)>0):
        e = queue.pop(0)
        print("rawdata:",e.rawdata,"hashvalue:",e.hashvalue,"isleaf:",e.isleaf,"ranlst:",e.ranlst,e.layer)
        if e.left != None:
            queue.append(e.left)
        if e.right != None:
            queue.append(e.right)

def search(root,s):
    print("start search")
    queue = []
    vo = []
    queue.append(root)
    while(len(queue)>0):
        e = queue.pop(0)
        l = e.ranlst[0]
        u = e.ranlst[-1]
        if s==l and s==u:
            while(len(queue)>0):
                last = queue.pop(0)
                vo.append(last)
            return True,e.rawdata,vo
        else:
            if(s>=l and s<=u):
                if e.left != None:
                    queue.append(e.left)
                if e.right != None:
                    queue.append(e.right)
            else:
                vo.append(e)
    return False,[],[]

def verify(result,vo,roothash):
    md5 = hashlib.md5()
    md5.update(str(result).encode())
    rh = md5.hexdigest()
    rn = MerkleNode(rawdata=result,hashvalue=rh,isleaf=True,ranlst=[result],layer=0)
    if(len(vo)==0):
        return rn.hashvalue == roothash
    def kf(merklenode):
        return float(merklenode.ranlst[0])
    vo.append(rn)
    vo.sort(key=kf)
    l = rn.layer
    while(len(vo)>2):
        curlst = []
        for v in vo:
            if v.layer==l:
                curlst.append(v)
        for c in curlst:
            vo.remove(c)
        for k in [curlst[x:x+2] for x in range(0,len(curlst),2)]:
            d1 = k[0].hashvalue.encode()
            d2 = k[1].hashvalue.encode()
            md5 = hashlib.md5()
            md5.update(d1+d2)
            newdata = md5.hexdigest()
            node = MerkleNode(rawdata=str(k[0].rawdata)+str(k[1].rawdata),hashvalue=newdata,\
                              left=k[0],right=k[1],isleaf=False,\
                              ranlst=[k[0].ranlst[0],k[1].ranlst[-1]],layer=k[0].layer+1)
            vo.append(node)
        vo.sort(key=kf)
        l += 1
    d1 = vo[0].hashvalue.encode()
    d2 = vo[1].hashvalue.encode()
    md5 = hashlib.md5()
    md5.update(d1+d2)
    newdata = md5.hexdigest()
    if newdata == roothash:
        return True
    else:
        return False
    

        
    
          
if __name__ == "__main__":
    rst = createLeafNodes([1,2,3,4,5,6,7,8,9])
    root = createTree(rst)
    r,result,vo = search(root,1)
    if r:
        print("find the result!")
        for v in vo:
            print(v.rawdata,v.ranlst,v.layer,v.hashvalue)
        print(verify(result,vo,root.hashvalue))
    else:
        print("not found")