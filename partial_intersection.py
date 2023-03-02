'''
Author: Maricarmen Arenas
Year : 2023

This code assumes that the list of communities (obtained using Louvain or any other community detecetion method of your choice) for the user-connection matrix
 (user-mentions in our case) and user-content (URLs posted by users as per the paper) matrix. 

The community information is required to be in a dictionary format as {"node id": community it belongs to} and saved as pickle files

To run:

python partial_intersection.py /path/to/connection_communities /path/to/content_communities

'''
from sklearn.metrics.cluster import normalized_mutual_info_score
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
import emoji
import pandas as pd
import nltk as nltk
import numpy as np
import string
import sys
from nltk.corpus import stopwords

import datetime

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import pickle


stop = stopwords.words('english')

##loop through ---communities
      ##if value -parition- in dictionary is equal to the parition then append to the list for that particular partition
#get key and value from dictionary 

# For example:

# {"'923786142": 7174,
#  "'2880652700": 2609,
#  "'896915441011929090": 721,
#  "'257063622": 1899,
#  "'1171747324755922944": 1899,
#  "'17554961": 1899,
#  "'1315742243647492097": 2435,
#  "'968543968236658688": 446,
#  "'1325734033385467912": 3383,
#  "'2906930111": 1858,
#  "'1271439033344815107": 1558,
#  "'1214257788295077891": 457,
#  "'609307614": 1558,
#  "'1044433092784652288": 457,
#  "'1211580667290537984": 457,
#  "'1264793507404959749": 457,
#  "'1614115225": 232,
#  "'58027339": 457,
#  "'124106577": 457,
#  "'2616299979": 457,
#  "'1318622535160909826": 457,}



####we feed the dictionnary to this simple algo that creates a group of groups

##loop through ---communities
      ##if a node in a community in dictionary is equal to the community then append to the list for that particular community
      ###this is going to give a list of nodes for every community
      ###hence creating a list of lists


def createGroupsofParitions(NewDict):
    groupofgroups=[]
    for i in set(NewDict.values()): 
        groups=[]
        for k,v in NewDict.items():
            if v==i:
                groups.append(k)
        ###append to list of lists 
        groupofgroups.append(groups)
    
    return groupofgroups

####we want a group of groups from the URL communities and a group of groups from the USER communities

NewCommUser = pickle.load(open(sys.argv[1],'rb'))
NewCommUrl = pickle.load(open(sys.argv[2], 'rb'))

groupofgroups1 = createGroupsofParitions(NewCommUrl)
groupofgroups2 = createGroupsofParitions(NewCommUser)


#############intersect two lists - gives a new list with unique elements (nodes)
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


count=0
intersgroups=[]

for i in groupofgroups1:

    count= count+1
    #inters=[]
    c=0
    for j in groupofgroups2:
        c= c+1

        inters= intersection(i, j)
        if len(inters)>1:
            if len(inters) >= (len(i))/2.5:
                if len(inters) >=(len(j))/2.5:
                #  if len(inters)>= (len(n))/2.5:
                    print('intersecion between community # '  + str(count)+ ' ' +  str(len(i))+ ' ' +  ' and between community # ' + str(c)+ ' ' +  str(len(j)))    
                    print(len(inters))
                    intersgroups.append(inters)

    
print('total number of intersections: '+ str(len(intersgroups)))

pickle.dump(intersgroups, open("partial_intersection_groups.pkl",'wb'))
############the output is a list of lists containing nodes that intersected between URL and USERMENTION communities