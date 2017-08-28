# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:51:07 2017

@author: Sean Chang
"""

import hashlib

class MerkleTreeHash(object):
    def __init__self(self):
        pass
    
    def find_merkle_hash(self,file_hashes):
        #here we want to find the merkle tree hash of all
        #the file hashes passed to this function. Note we 
        #are going to be using recurssion to solve the problem.
        
        #this is the simple procedure we will follow for
        #finding the hash given alist of hashes we first 
        #group the all the hashes in twos next we concatinate
        #the hashes in each group and compute the hash of
        #the group, then keep track of the group hashes,
        #we will repeat this steps until we have a single
        #hash then that becomes the hash we are looking for
        
        blocks = []
        
        if not file_hashes:
            raise ValueError("Missing required file hashes for computing merkle tree hash")
            
        #first sort the hashes
        for m in sorted(file_hashes):
            blocks.append(m)
        
        list_len = len(blocks)
        #adjust the block of hashes until we have an even number of items.
        #in the blocks, this entails appending to the end of the block
        #the last entry. To do this we use modulus math to determine when 
        #we have an even number of items.
        while list_len % 2 != 0:
            blocks.extend(blocks[-1:])
            list_len = len(blocks)
            
        #now we have an even number of items in the block we need to group
        #the items in twos
        secondary = []
        for k in [blocks[x:x+2] for x in range(0,len(blocks),2)]:
            #note k is a list with only two items, which is what we want. This
            #is so that we can concatinate them and create a new hash from them
            k[0]=k[0].encode(encoding='utf8')
            k[1]=k[1].encode(encoding='utf8')
            hasher = hashlib.sha256()
            hasher.update(k[0]+k[1])
            secondary.append(hasher.hexdigest())
        
        #now because this is a recursive method, we nee to determin when we only
        #have a single item in the list this marks the end of the iteration and
        #we can return the last hash as the merkle root
        if len(secondary) == 1:
            #note only returning the first 64 characters, however if you want 
            #to return the entire ash just remove the last section [0:64]
            return secondary[0][0:64]
        else:
            #if the number of items in the lists is more than one, we still need
            #to iterate through this so we pass it back to the mothod. We pass
            #the secondary list since it holds the second iteration results.
            return self.find_merkle_hash(secondary)
        
if __name__ == '__main__':
    
    #It's time to test the class. We will test by generatring 13 random hashes and
    #then try to find their merkle tree hash.
    file_hashes = []
    for i in range(0,13):
        file_hashes.append(str(i))
    
    print("finding the merkle tree hash of {0} random hashes".format(len(file_hashes)))
    
    cls = MerkleTreeHash()
    mk = cls.find_merkle_hash(file_hashes)
    print('The merkle tree hash of the hashes below is :{0}'.format(mk))
    print("...")
    print(file_hashes)
    
            
        
