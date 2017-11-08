# -*- coding: utf-8 -*-
from __future__ import division
from DBConnect import DBConnect
from datetime import datetime
from TFIDF import TFIDF
import pprint
import operator
import math
import numpy as np
import tensorly as tl
import tensorly.decomposition as dec

class MovieTensor:
    model=None
    db = None
    tfIdf = None
    def __init__(self,model):
        self.model = model
        self.db = DBConnect()
        self.tfIdf = TFIDF("","","_actor_")
        
        
    def getListAsString(self,moviesList):
        moviesListStr = str(moviesList)
        moviesListStr = moviesListStr.replace('[','(')
        moviesListStr = moviesListStr.replace(']',')')
        return moviesListStr
    
    def getTensor(self):
        if self.model == 1:
            
            
            yearsCountQuery = "select count(distinct year) from mlmovies"
            #movieActorsCountQuery = "select count(distinct movieid) from mlmovies where movieid  in (6058,9818,5914,6097,7232,9443,7062,8929,4354,10059)  "
            res = self.db.executeQuery(yearsCountQuery)
            countStr = res[0] 
            countString = str(countStr)
            countString = self.tfIdf.getCount(countString)
            noOfDistinctYear = int(countString)
            
            
            # get the no of actors
            movieActorsCountQuery = "select count(*) from imdb_actor_info  "
            #movieActorsCountQuery = "select count(distinct actorid) from imdb_actor_info  where actorid in (17838,45899,61523,68671,96585,99457,128645,133985) "
            res = self.db.executeQuery(movieActorsCountQuery)
            countStr = res[0] 
            countString = str(countStr)
            countString = self.tfIdf.getCount(countString)
            noOfActors = int(countString)
            
            # get the no of movies
            movieActorsCountQuery = "select count(*) from mlmovies  "
            #movieActorsCountQuery = "select count(distinct movieid) from mlmovies where movieid  in (6058,9818,5914,6097,7232,9443,7062,8929,4354,10059)  "
            res = self.db.executeQuery(movieActorsCountQuery)
            countStr = res[0] 
            countString = str(countStr)
            countString = self.tfIdf.getCount(countString)
            noOfMovies = int(countString)
            #noOfMovies = 2 
            
#            actorMovieYearTensor = np.ndarray(  shape=(noOfActors,noOfMovies,noOfDistinctYear))
#            for i in range(0,noOfActors):
#                for j in range(0,noOfMovies):
#                    for k in range(0,noOfDistinctYear):
#                        actorMovieYearTensor[i,j,k] = 0.0
#                        #print actorMovieYearTensor[i,j,k]
            
            #build movie indices
            movieIdVsIndex = {}
            movieIndexVsName = {}
            query = "select * from mlmovies order by movieid"
            #query = "select *  from mlmovies where movieid  in (6058,9818,5914,6097,7232,9443,7062,8929,4354,10059) order by movieid"
            movieIndex = 0
            res = self.db.executeQuery(query)
            for movie in res:
                movieId = movie[0]
                movieName = movie[1]
                movieIdVsIndex[movieId] = movieIndex
                movieIndexVsName[movieIndex] = movieName
                movieIndex = movieIndex +1
                
                
            #build year indices
            yearVsIndex= {}
            yearIndexVsYear = {}
            q = "select distinct year from mlmovies order by year"
            res = self.db.executeQuery(q)
            yearIndex = 0
            for yearRow in res:
                year = yearRow[0]
                yearVsIndex[str(year)]=yearIndex
                yearIndexVsYear[yearIndex]=year
                yearIndex = yearIndex+1
              
            
            actorMovieYearMatrix = np.zeros((noOfActors,noOfMovies,noOfDistinctYear))
            
                
            
            
            
            query = "select * from imdb_actor_info order by actorid "
            actors = self.db.executeQuery(query)
            actorIndex = 0
            actorIdVsIndex = {}
            actorIndexVsName = {}
            for actor in actors:
                actorid = actor[0]
                actorName = actor[1]
                actorrelatedMoviesQ = "select * from movie_actor where actorid = "+str(actorid)
                actorrelatedMovies = self.db.executeQuery(actorrelatedMoviesQ)
                movieIds = []
                for movie in actorrelatedMovies:
                    movieIds.append(movie[0])
                # we got the movies
                moviesQuery = "select * from mlmovies where movieid in "+self.getListAsString(movieIds)
                res = self.db.executeQuery(moviesQuery)
                for movieYear in res:
                    movieid = movieYear[0]
                    year = movieYear[2]
                    #actorMovieYearTensor[actorIndex,movieIdVsIndex[movieid],yearVsIndex[str(year)]] = 1.0
                    actorMovieYearMatrix[actorIndex][movieIdVsIndex[movieid]][yearVsIndex[str(year)]]=1
                    
                actorIdVsIndex[actorid]=actorIndex
                actorIndexVsName[actorIndex] = actorName
                actorIndex=actorIndex+1            

            actorMovieYearMatrix[0][0][0]=1
            actorMovieYearMatrix[1][1][1]=1
            actorMovieYearTensor = tl.tensor(actorMovieYearMatrix)
            
            decomposed = dec.parafac(actorMovieYearTensor,rank = 5)
            
            semanticsActor = decomposed[0]
            semanticsMovie = decomposed[1]
            semanticsYear = decomposed[2]
            for i in range(0, semanticsActor.shape[1]):
                
                actorsRow = semanticsActor[:,i]                
                mean = np.mean(actorsRow)                
                print ("ACTORS GROUPED UNDER LATENT SEMANTICS {0} ".format( i+1))
                for j in range(0, noOfActors):
                    if(actorsRow[j] >= mean):
                        print(actorIndexVsName[j])
            
            
            for i in range(0, semanticsMovie.shape[1]):
                
                moviesRow = semanticsMovie[:,i]
                mean = np.mean(moviesRow)
                print("MOVIES GROUPED UNDER LATENT SEMANTICS {0}".format(i+1))
                for j in range(0, noOfMovies):
                    if(moviesRow[j] >= mean):
                        print(movieIndexVsName[j])
    
            for i in range(0, semanticsYear.shape[1]):
                yearsRow = semanticsYear[:,i]
                mean = np.mean(yearsRow)
                print("YEARS GROUPED UNDER LATENT SEMANTICS {0}".format(i+1))
                for j in range(0, noOfDistinctYear):
                    if(yearsRow[j] >= mean):
                        print(yearIndexVsYear[j])
            
            
        elif self.model == 2:
            noOfTags = 0
            query = "select count(*) from genome_tags"
            count = self.db.executeQuery(query)
            countStr = self.tfIdf.getCount(str(count[0]))
            noOfTags = int(countStr)
            
            # get the no of movies
            movieActorsCountQuery = "select count(*) from mlmovies  "
            res = self.db.executeQuery(movieActorsCountQuery)
            countStr = res[0] 
            countString = str(countStr)
            countString = self.tfIdf.getCount(countString)
            noOfMovies = int(countString)
            
            q = "select count(distinct rating) from mlratings"
            res = self.db.executeQuery(q)
            countStr = res[0] 
            countString = str(countStr)
            countString = self.tfIdf.getCount(countString)
            noOfRatings = int(countString)
            
            
            tagMovieRatingMatrix = np.zeros((noOfTags,noOfMovies,noOfRatings))
            
            
            
            #print tagMovieRatingTensor
            
            # build tag index
            query = "select * from genome_tags order by tagid"
            tags = self.db.executeQuery(query)
            tagIndex = 0
            tagIdVsIndex = {}
            tagIndexVsName = {}
            for tag in tags:
                tagid = tag[0]
                tagName = tag[1]
                tagIdVsIndex[tagid] = tagIndex
                tagIndexVsName[tagIndex]=tagName
                tagIndex = tagIndex + 1
                
            
            
            
            query = "select * from mlmovies order  by movieid"
            movieIndex = 0
            movieIdVsIndex = {}
            movieIndexVsName = {}
            movies = self.db.executeQuery(query)
            for movie in movies:
                movieid = movie[0]
                movieName = movie[1]
                movieIdVsIndex[movieid] = movieIndex
                movieIndexVsName[movieIndex]=movieName
                
                movieTagsQ = "select * from mltags where movieid = "+str(movieid)
                movieTags = self.db.executeQuery(movieTagsQ)
                movieTagsList = []
                for movieTag in movieTags:
                    movieTagsList.append(movieTag[2])
                totalNoOfRatingsQ = "select count(*) from mlratings where movieid = "+str(movieid)
                res = self.db.executeQuery(totalNoOfRatingsQ)
                totalRatingsStr = self.tfIdf.getCount(str(res[0]))
                totalRatings = int(totalRatingsStr)
                
                sumQ = "select movieid, sum(rating) from mlratings  where movieid = "+str(movieid)+" group by movieid"
                res = self.db.executeQuery(sumQ)
                sumRating = 0
                for r in res:
                    sumRating = sumRating + r[1]
                avgRating = float(sumRating)/totalRatings
                
                for tag in movieTagsList:
                    tagIndex = tagIdVsIndex[tag]
                    
                    for i in range(1,noOfRatings+1):
                        if avgRating <= float(i):
                            tagMovieRatingMatrix[tagIndex][movieIndex][i-1] = 1
                            #print "setting one"
                
                            
                movieIndex = movieIndex + 1
                
            tagMovieRatingMatrix[0][0][0]=1
            tagMovieRatingMatrix[1][1][1]=1
            tagMovieRatingTensor = tl.tensor(tagMovieRatingMatrix)
            
            decomposed = dec.parafac(tagMovieRatingTensor,rank = 5)
            
            semanticsTag = decomposed[0]
            semanticsMovie = decomposed[1]
            semanticsRating = decomposed[2]
            
            for i in range(0, semanticsTag.shape[1]):
                
                tagRows = semanticsTag[:,i]
                mean = np.mean(tagRows)                
                print (" TAGS GROUPED UNDER LATENT SEMANTICS {0} ".format( i+1))
                for j in range(0, noOfTags):
                    if(tagRows[j] >= mean):
                        print(tagIndexVsName[j])
            
            for i in range(0, semanticsMovie.shape[1]):
                
                movieRows = semanticsMovie[:,i]
                mean = np.mean(movieRows)
                print("MOVIES GROUPED UNDER LATENT SEMANTICS {0}".format(i+1))
                for j in range(0, noOfMovies):
                    if(movieRows[j] >= mean):
                        print(movieIndexVsName[j])
            
    
            for i in range(0, semanticsRating.shape[1]):
                ratingRows = semanticsRating[:,i]
                mean = np.mean(ratingRows)
                print("RATINGS GROUPED UNDER LATENT SEMANTICS {0}".format(i+1))
                for j in range(0, noOfRatings):
                    if(ratingRows[j] >= mean):
                        print(j+1)    
                
                
                    
                    
                
            
            
            
        
        