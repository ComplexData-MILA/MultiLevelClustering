####################
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
import re
from nltk.corpus import stopwords

import datetime

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import pickle


stop = stopwords.words('english')
###this is an example with year 2021

##loop through ---communities
      ##if value -parition- in dictionary is equal to the parition then append to the list for that particular partition
#get key and value from dictionary 


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
#  "'1318622535160909826": 457,
#  "'273482834": 7634,
#  "'872479764602273792": 1558,
#  "'1280720974170603521": 1558,
#  "'796621679648763904": 457,
#  "'1288751207650074625": 1831,
#  "'403972917": 1558,
#  "'1205240905894170624": 2294,
#  "'1146001293309202434": 1558,
#  "'379647134": 557,
#  "'878389780551172096": 831,
#  "'741131580459077632": 1558,
#  "'1138982536519069696": 1484,
#  "'1309143997424824324": 1558,
#  "'4175523802": 8523,
#  "'763791774586441729": 1484,
#  "'1275739568239599618": 1558,
#  "'716111672": 7520,
#  "'533005696": 1555,
#  "'974758363262267392": 1555,
#  "'901939045743423489": 1555,
#  "'701528478": 3817,
#  "'792189247260356608": 1793,
#  "'1326457022087303174": 1555, 


####we feed the dictionnary to this simple algo that creates a group of groups

##loop through ---communities
      ##if a node in a community in dictionary is equal to the community then append to the list for that particular community
      ###this is going to give a list of nodes for every community
      ###hence creating a list of lists


##### for example nodes 44,34,55 are part of the same community while 55,61,72,71 are nodes part of another community
#[[  44, 34, 55,],[55,61,71,81] ,[42,66,77,88]]


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


############the printing part is not necessary 
############the output is a list of lists containing nodes that intersected between URL and USERNAME communities





#################the below code is to visualize the list --to see the number of Onlyfans, Twitter accounts, etc
########counts the number of elements from a specific dataframe column
def counthowmany(dfo):
    listofsom=[]
    for i in dfo:
        for j in i:
            listofsom.append(j)
    return(len(list(set(listofsom))))





########counts the number of UNIQUE elements from a specific dataframe column
def uniqueList(dfo):
    listofsom=[]
    for i in dfo:
        for j in i:
            j=j.strip()
            listofsom.append(j)
    return(list(set(listofsom)))

########count the number of twitter accounts from a specific dataframe column

def countTwitter(dfo):
    listTwitter=[]
    countT=[]
    
    for i in dfo:
        for j in i:
           
            if j.find('twitter.com') != -1: 
                url_string=j.lower()
                a= url_string.rsplit("/")[3]
                a=a.rsplit("?")[0]
                urln= 'https://twitter.com/' + a
                listTwitter.append(urln)
                #print(len(listTwitter))
                countT= Counter(listTwitter)
               
        ####returns list of Twitter account and Number of Twitter accounts, and the number of each specific twitter account
    return(len(list(set(listTwitter))), listTwitter, countT)



########count the number of Onlyfans accounts accounts from a specific dataframe column

def countOnlyfans(dfo):
    listOnly=[]
    countO=[]
    
    for i in dfo:
        for j in i:
            
            if j.find('onlyfans.com') != -1:
                count=Counter(j)
                if count['/']<3:
                    urln=j
                
                else:
                    
                    url_string=j.lower()
                    a= url_string.rsplit("/")[3]
                    a=a.rsplit("?")[0]
                    urln= 'www.onlyfans.com/' + a
          
               

                listOnly.append(urln)
                countO= Counter(listOnly)
            
                
    ####returns list of Onlyfans account and Number of onlyfans account, and the number of each specific Onlyfans account
    return(len(list(set(listOnly))), listOnly, countO)

        
############################ we need to feed the presentation algorithm with the dataframe we used for that year, it contains the dataframe with 
############


def presentation_algo(df, intersgroups ):
    listurls=[]
    listusers=[]
    List_urls =[]
    List_users =[]
    ListOnly=[]
    ListTwitter=[]
    onlyfans=[]
    twitters=[]
    size=[]
    rts=[]
    countO=[]
    countT=[]
    
    for i in range(len(intersgroups)):
        newdf = df[df['author id'].isin(intersgroups[i])]
         ####here we extract from the original dataframe, the rows that correspond to the nodes we have, we iterate through it so we go through all 
         #### the nodes we have and collect all the information

        siz=len(newdf)

        r1=len(newdf[newdf['Retweet']=="'retweeted"])
        r2=len(newdf[newdf['Retweet']=="'replied_to"])
        r3=len(newdf[newdf['Retweet']=="'quoted"])
        rt= r1+r2+r3
        pr=(rt/siz)*100

        #####the number of retweets versus total tweets
        pr=str(round(pr, 2))
        
        ###number of urls (shortcut url)
        a = counthowmany(newdf['urls_union']) 
         ###number of userames
        b = counthowmany(newdf['username_union'])
         ###number of extended urls
        c = uniqueList(newdf['urls_ex_union']) 

       # d =uniqueList(newdf['username_union'])
        
        ####from the extended links, we extract the number 
        e, p, k=countOnlyfans(newdf['urls_ex_union'])
        f, t, l=countTwitter(newdf['urls_ex_union'])
        
  
        listurls.append(a)
        listusers.append(b)
        List_urls.append(c)
        #List_users.append(d)
        ListOnly.append(e)
        ListTwitter.append(f)
        onlyfans.append(p)
        twitters.append(t)
        countO.append(k)
        countT.append(l)
        size.append(siz)
        rts.append(pr)
        
    return listurls, listusers, List_urls, ListOnly, ListTwitter, onlyfans, twitters,size,rts, countO, countT


listurls, listusers, List_urls, ListOnly, ListTwitter, onlys, twitters,size, rts,cO,cT =presentation_algo(dfm,intersgroups)



dfinz= pd.DataFrame(listurls, columns=['urls numbers'])
dfinz['user numbers']=listusers
dfinz['number of Onlyfans']=ListOnly
dfinz['count Onlyfans']=cO
dfinz['number of Twitter accounts']=ListTwitter
dfinz['count tweets']=cT
dfinz['number of authors']=numofauth
dfinz['number of tweets']=size
dfinz['percentage retweets']=rts


###############get a table with the above titles

####################we only want to see the ones with more than one user and url and onlyfans account
dfinzt[(dfinzt['user numbers']>1) & (dfinzt['urls numbers']>1) & (dfinzt['number of Onlyfans']>1)]
len(a)