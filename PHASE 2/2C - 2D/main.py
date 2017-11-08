#!/usr/bin/env python
import sys
import pandas
import numpy
import math
# -*- coding: utf-8 -*-
from DBConnect import DBConnect
from TFIDF import TFIDF
from MovieTensor import MovieTensor

def compute_actor_weightage(row):
    return (1+ row['max_actor_rank'] - row['actor_movie_rank'])/(1+ row['max_actor_rank'] - row['min_actor_rank'])

def CalculateTimestampWeights(row, min_timestamp, max_timestamp):
    return ((pandas.to_datetime(row['timestamp'])-min_timestamp).days + 1)/((max_timestamp-min_timestamp).days+1)

def aggregate_tf_weightages(row):
    return numpy.round(row['actor_rank_weightage'] + row['timestamp_weightage'], decimals=4)

def ComputeTF(row):
    return row['tag_weightage'] / row ['total_weightage_actor']

def ComputeIDF(row, total_actors):
    return math.log10(total_actors / row ['actor_count'])

def CalculateTFIDF(row):
    return row['tf']*row['idf']

def ProcessTFandIDFtoTFIDF(tfdata, idfdata):
    tfidfdata = tfdata.merge(idfdata, on='tagid')
    tfidfdata['tfidf'] = tfidfdata.apply(CalculateTFIDF, axis=1)
    return tfidfdata[['actorid','tagid','tfidf']]

def main():
    
    command = sys.argv[1]
    no = int(sys.argv[2])
    if command == "CP":
        if no == 1:
            
            tensor = MovieTensor(1)
            tensor.getTensor()
        elif no == 2:
            tensor = MovieTensor(2)
            tensor.getTensor()
    elif command == "SVD":
        allactormoviesdata =pandas.read_csv("movie-actor.csv")

        alltagsdata = pandas.read_csv("mltags.csv")
        
        allactormoviesdata['max_actor_rank'] = allactormoviesdata.groupby(['movieid'])['actor_movie_rank'].transform(max)
        allactormoviesdata['min_actor_rank'] = allactormoviesdata.groupby(['movieid'])['actor_movie_rank'].transform(min)
        
        allactormoviesdata['actor_rank_weightage'] = allactormoviesdata.apply(compute_actor_weightage, axis=1)
        #
        # print(allactormoviesdata)
        
        
        min_timestamp = pandas.to_datetime(min(alltagsdata['timestamp']))
        max_timestamp = pandas.to_datetime(max(alltagsdata['timestamp']))
        
        alltagsdata['timestamp_weightage'] = alltagsdata.apply(CalculateTimestampWeights, axis=1, args=(min_timestamp, max_timestamp))
        
        
        mergeddata = allactormoviesdata[['actorid','movieid','actor_rank_weightage']].merge(alltagsdata[['movieid','tagid','timestamp_weightage']], on='movieid')
        
        #print(mergeddata[mergeddata['actorid'].isin([878356,1860883,316365,128645])])
        
        mergeddata['total_weightage'] = mergeddata.apply(aggregate_tf_weightages, axis=1)
        
        
        mergeddata['tag_weightage'] = mergeddata.groupby(['actorid','tagid'])['total_weightage'].transform('sum')
        tfdata = mergeddata[['actorid', 'tagid', 'tag_weightage']].drop_duplicates(subset=['tagid', 'actorid'])
        
        tfdata['total_weightage_actor'] = tfdata.groupby(['actorid'])['tag_weightage'].transform('sum')
        
        tfdata['tf'] = tfdata.apply(ComputeTF, axis=1)
        
        
        taglist = tfdata['tagid'].tolist()
        alltagsdata = pandas.read_csv("mltags.csv")
        alltagsdata = alltagsdata[alltagsdata['tagid'].isin(taglist)]
        
        #print(alltagsdata)
        
        allactormoviesdata = pandas.read_csv("movie-actor.csv")
        requiredtagsdata = alltagsdata.merge(allactormoviesdata, on='movieid')
        
        requiredtagsdata.drop_duplicates(subset=['tagid', 'actorid'], inplace=True)
        requiredtagsdata['actor_count'] = requiredtagsdata.groupby('tagid')['actorid'].transform('count')
        requiredtagsdata.drop_duplicates(subset=['tagid'], inplace=True)
        
        actordata = pandas.read_csv("imdb-actor-info.csv")
        total_actors = actordata.shape[0]
        
        requiredtagsdata['idf'] = requiredtagsdata.apply(ComputeIDF, axis=1, total_actors=total_actors)
        #
        # print(total_actors)
        # print(requiredtagsdata)
        
        tfidfdata = ProcessTFandIDFtoTFIDF(tfdata, requiredtagsdata[['tagid', 'idf']])
        
        # print(tfdata)
        
        
        #tfidfdata = tfidfdata[tfidfdata['actorid'].isin([878356,1860883,316365,128645])]
        
        #print(tfidfdata)
        
        actor_tag_matrix = tfidfdata.pivot_table(index='actorid', columns='tagid', values='tfidf', fill_value=0)         
        print "Actor Tag Matrix"
        print    actor_tag_matrix     
                
                    
    
        
        
        
        tf = TFIDF("",1,"_actor_")
        tf.calcMoviesVector()
        
    
    
    
#    if command == "print_actor_vector":
#        tf = TFIDF(sys.argv[3],int(sys.argv[2]),"_actor_")
#        tf.getWeightedTagVector()
#    elif command == "print_genre_vector":
#        tf = TFIDF(sys.argv[3],sys.argv[2],"_genre_")
#        tf.getWeightedTagVector()
#    elif command == "print_user_vector":
#        tf = TFIDF(sys.argv[3],int(sys.argv[2]),"_user_")
#        tf.getWeightedTagVector()
#    elif command == "differentiate_genre":
#        tf = TFIDF(command[3],"","")
#        if sys.argv[4] == "TF-IDF-DIFF":
#            tf.tfIdfDiff(sys.argv[2],sys.argv[3])
#        else:
#            tf.pDiff1(sys.argv[4],sys.argv[2],sys.argv[3])
    
   
   
   
    
        
        
    
    
    
    
    
    
        
        
        
    
   
    
    
       
       
   
    
    
    #db = DBConnect()
    #db.executeQuery("select * from mltags limit 5")
# display some lines

if __name__ == "__main__": main()
