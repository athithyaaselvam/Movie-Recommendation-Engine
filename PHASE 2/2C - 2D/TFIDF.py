# -*- coding: utf-8 -*-
from __future__ import division
from DBConnect import DBConnect
from datetime import datetime
import pprint
import operator
import math
import numpy as np
class TFIDF:
    model=None
    entityid=None
    commandStr=None
    relation = None
    db = None
    tableName = None
    epoch = datetime.utcfromtimestamp(0)
    def __init__(self,modelP,entityidP,commandstrP):
        self.model = modelP
        self.entityid=entityidP
        self.commandStr = commandstrP
        self.db = DBConnect()
        if "_actor_" in self.commandStr:
            self.relation = "actor"
        elif "_genre_" in self.commandStr:
            self.relation = "genre"
        elif "_user_" in self.commandStr:
            self.relation = "user"
        
        
    def getEntityMovieTableName(self):
        if self.relation == "actor":
            return "movie_actor"
        elif self.relation == "genre":
            return "mlmovies"
        elif self.relation == "user":
            return "mlratings"
        
    def getWeight(self,rank,totalCount):
        per = rank/totalCount
        inv = 1 - per
        return inv
    
    def getMillis(self,timestampDiff):
        if timestampDiff == None:
            return None
        #print "getMillis : "+str(timestampDiff)
        #timestampDiffStr = str(timestampDiff) 
        return (timestampDiff - self.epoch).total_seconds() * 1000.0
#        d = datetime.strptime(timestampDiffStr, "%Y-%m-%d %H:%M:%S,%f").strftime('%s')
#        d_in_ms = int(d)*1000
        #return d_in_ms
    
    def getCount(self, countString):
        countString = countString.replace('(','')
        countString = countString.replace('[','')
        countString = countString.replace(']','')
        countString = countString.replace(')','')
        countString = countString.replace('L','')
        countString = countString.replace(',','')
        countString = countString.replace('\'','')
        countString = countString.replace('\'','')
        return countString
    
    
    def calcTFIDFApproach2(self,movies):
        isActor = None
        #print "self relation = "+self.relation
        if self.relation == "actor":
            isActor = True
        else:
            isActor = False
            
        
        
        
        #records will be list of tuples
        #print "type of records object = "+str(type(records))
        movieVsWeight = {}
        moviesList = []
        moviesListStr = None
        moviesCount = 0
        globalTagIdVsTF = {}
        globalTagIdVsIDF = {}
        globalTagNameVsTFIDF = {}
        tagIdVsMovieList = {}
        
        
        
        for tup in movies:
            movieid = tup[0]
            rank = 0
            if isActor:
                rank = tup[2]
            
                
            
            moviesList.append(movieid)
            #print "movie id = "+str(movieid)
            #print "rank = "+str(rank)
            
            

            weight = 0
            if isActor:
                #Weight for actor rank for this movie
                movieActorsCountQuery = "select count(*) from "+self.tableName+" where movieid = "+str(movieid)
                res = self.db.executeQuery(movieActorsCountQuery)
                countStr = res[0] 
                countString = str(countStr)
                countString = self.getCount(countString)
                weight=self.getWeight(rank,int(countString)+1)#movie weight
            #print "weight = "+str(weight)
            movieVsWeight[movieid]=weight
            moviesCount = moviesCount+1            
            
            # Weight for tags for this movie
            tagsRelatedToThisMovie = "select * from mltags where movieid ="+str(movieid)+" order by mtimestamp desc"
            tagIdVsTimeStamp = {}
            tags = self.db.executeQuery(tagsRelatedToThisMovie)
            if tags == None or len(tags) == 0:
                continue
            i=0
            latest = None
            oldest = None
            for tag in tags:
                movieid = tag[1]
                tagid = tag[2]
                timestamp = tag[3]
                if i==0:
                    latest = timestamp
                oldest = timestamp
                tagIdVsTimeStamp[tagid]=timestamp
                i=i+1
                
                if tagid in tagIdVsMovieList:
                    tagMoviesList = tagIdVsMovieList[tagid]
                    tagMoviesList.append(movieid)
                    tagIdVsMovieList[tagid] = tagMoviesList
                else:
                    tagIdVsMovieList[tagid] = [movieid]
                    
            
            latestMillis = self.getMillis(latest)
            oldestMillis = self.getMillis(oldest)
            timeStampDiff = latestMillis - oldestMillis
            tagIdVsWeight  = {}
            totalTagWeights = 0.00
            for tagId,timeStamp in tagIdVsTimeStamp.items():
                tagWeight = 0.00
                if len(tags) == 1:
                    tagWeight = 0.9
                else:    
                    tagWeight = (self.getMillis(timeStamp) - oldestMillis) / timeStampDiff
                #print "tag= "+str(tagId)+"  tagWeight = "+str(tagWeight)
                combinedWeight = weight + tagWeight # actor weight + tag weight
                tagIdVsWeight[tagId]= combinedWeight
                #print "tag= "+str(tagId)+"  combinedWeight = "+str(combinedWeight)
                totalTagWeights = totalTagWeights + combinedWeight
                
            # TF calculation
            for tagId,tagWeight in tagIdVsWeight.items():
                tf = 0
                if totalTagWeights != 0.0:                
                    tf = tagIdVsWeight[tagId]/totalTagWeights # weight / totalWeight for this movie tags
                    
                    
                #print "tagId = "+str(tagId)+" tf = "+str(tf)
                if tagId in globalTagIdVsTF:
                    currentWeight = globalTagIdVsTF[tagId]
                    currentWeight = currentWeight + tf
                    globalTagIdVsTF[tagId] = currentWeight
                else:
                    globalTagIdVsTF[tagId] = tf
                    
        # end for
        #print "total Movies = "+str(moviesCount)
        # IDF calculation
        tagsList = []
        for tagId,movieSet in tagIdVsMovieList.items():
            noOfMoviesAssociated = len(movieSet)
            #print "tagid = "+str(tagId)
            #print "no of movies associated = "+str(noOfMoviesAssociated)
            idf = moviesCount  / noOfMoviesAssociated
            globalTagIdVsIDF[tagId] = idf
            tagsList.append(tagId)
        
        
        tagsListStr = str(tagsList)
        tagsListStr = tagsListStr.replace('[','(')
        tagsListStr = tagsListStr.replace(']',')')
        tagsQuery = "select * from genome_tags where tagid in "+tagsListStr
        tags = self.db.executeQuery(tagsQuery)
        tagIdVsName = {}
        for tag in tags:
            tagIdVsName[tag[0]]=tag[1]
    
    
        
        
        
        
        for tagId,Name in tagIdVsName.items():
            globalTagNameVsTFIDF[Name] = globalTagIdVsTF[tagId] * globalTagIdVsIDF[tagId]
            
        
        print "TF IDF SORTED"
        sortedTagVsIDF = sorted(globalTagNameVsTFIDF.items(),key=operator.itemgetter(1),reverse=True)
        print ""+str(sortedTagVsIDF)
            
        
#        for tagId,v in globalTagIdVsTF.items():
#            print " "+tagIdVsName[tagId]+" = "+str(v)
            
            
            
    def calcTFIDFApproach1(self,movies ):
        isActor = False
        if self.relation == "actor":
            isActor = True
            
        movieVsWeight = {}
        
        moviesList = []
        for movie in movies:
            movieid = movie[0]
            rank = movie[2]
            weight = 0
            if isActor:
                #Weight for actor rank for this movie
                movieActorsCountQuery = "select count(*) from "+self.tableName+" where movieid = "+str(movieid)
                res = self.db.executeQuery(movieActorsCountQuery)
                countStr = res[0] 
                countString = str(countStr)
                countString = self.getCount(countString)
                weight=self.getWeight(rank,int(countString)+1)#movie weight
            #print "weight = "+str(weight)
            movieVsWeight[movieid]=weight
            moviesList.append(movieid)
            
        
        moviesListStr = str(moviesList)
        moviesListStr = moviesListStr.replace('[','(')
        moviesListStr = moviesListStr.replace(']',')')
        #print "movieslist = "+moviesListStr
        
        #Get the tags related to the actor/genre/user
        oldestTagQuery = "select * from mltags where movieid in "+moviesListStr+" order by mtimestamp limit 1"
        #print "oldestTagQuery = "+oldestTagQuery
        oldestTagQueryRes = self.db.executeQuery(oldestTagQuery)
        oldestTimeStamp= None
        newestTimeStamp = None
        timeRange = None
        
        for oldTag in oldestTagQueryRes:
            oldestTimeStamp = oldTag[3]
        oldesMillis = self.getMillis(oldestTimeStamp)
            
            
        
        tagsQuery = "select * from mltags where movieid in "+moviesListStr+" order by mtimestamp desc"
        tags = self.db.executeQuery(tagsQuery)
        actorTagsCount = len(tags)
        tagVsTotalWeight = {}
        tagVsTF = {}
        tagIdVsTF={}
        taglist=[]
        movieVsTags = {}
        n = 1
        for tag in tags:
            movieid = tag[1]
            tagid = tag[2]
            timestamp = tag[3]
            if n == 1:
                newestTimeStamp = timestamp
                timeRange = self.getMillis(newestTimeStamp) - oldesMillis
                
#            if movieid in movieVsTags:
#                l = movieVsTags[movieid]
#                if tagid in l and l.count(tagid) > 3: # same movieid and tag id might be irrelevant after a certain count
#                    print" same movieid and tag id skipping..."
#                    continue                
#                else:
#                    l.append(tagid)
#                    movieVsTags[movieid]=l
#            else:
#                l=[tagid]
#                movieVsTags[movieid]=l
            taglist.append(tagid)
#            print "tagid = "+str(tagid)
#            print "movie id = "+str(movieid)
#            print "timestamp = "+str(tag[3])
            tagWeight = self.getWeight(n,actorTagsCount+1)
            tagWeight = (self.getMillis(timestamp) - oldesMillis) / timeRange
            rankWeight = movieVsWeight[movieid]
            #print "rankweight = "+str(rankWeight)
            n= n+1
            #print "tagWeight = "+str(tagWeight)
            #combinedWeight = (tagWeight + (3 *rankWeight))/4
            combinedWeight = (tagWeight + rankWeight)
            #print "combinedWeight = "+str(combinedWeight)
            if tagid in tagVsTotalWeight:
                tempWeight = tagVsTotalWeight[tagid]
                tempWeight +=combinedWeight
                tagVsTotalWeight[tagid] = tempWeight
            else:
                tagVsTotalWeight[tagid]=combinedWeight
            # dict ("tagid" ,"weight1,weight 2")
            
        #print "tagVsTotalWeight = "+str(tagVsTotalWeight)
        
        tagsListStr = str(taglist)
        tagsListStr = tagsListStr.replace('[','(')
        tagsListStr = tagsListStr.replace(']',')')
        
        
         #Get the tags related to the actor for Tag Id Vs Name dictionary
        tagsQuery = "select * from genome_tags where tagid in "+tagsListStr
        tags = self.db.executeQuery(tagsQuery)
        tagIdVsName = {}
        for tag in tags:
            tagIdVsName[tag[0]]=tag[1]
        
            
        
        
        
        totalWeight = 0
        for key,val in tagVsTotalWeight.items():
#            print "key - "+str(key)
#            print "weight = "+str(val)
            totalWeight +=val
        
        
        # Calcualting TF for each tag
        for key,val in tagVsTotalWeight.items():
            
            tf = tagVsTotalWeight[key]/totalWeight
            tagVsTF[tagIdVsName[key]]=tf
            tagIdVsTF[key]=tf
            
        sortedTagVsTF = sorted(tagVsTF.items(),key=operator.itemgetter(1),reverse=True)  
        if self.model == "TF":
            print "TAG Vs TF "+str(sortedTagVsTF)
        
        if self.model == "TF":
            print "model = TF"
            return
        
        # Calculating IDF
        totalDocsCount = 27279
        if isActor:
            totalDocsCount = 27279
        elif self.relation == "genre":
            totalDocsCount = 19
        elif self.relation  == "user":
            totalDocsCount = 71567
            
            
        #print "Total documenst = "+str(totalDocsCount)
        tagIdVsIDF = {}
        tagIdVsTFIDF={}
        
        #Total no of documents
        for key,val in tagIdVsTF.items():
            tagid = key
            #print "tagid = "+str(tagid)
            moviesRelatedToThisTag = "select movieid from mltags where tagid = "+str(tagid)
            movies = self.db.executeQuery(moviesRelatedToThisTag)
            moviesList=[]
            for mov in movies:
                moviesList.append(mov[0])
            
            movListStr = str(moviesList)
            movListStr = movListStr.replace('[','(')
            movListStr = movListStr.replace(']',')')
            genreSet = set()
            totalRelatedWithThisTag= 0
            
    
            if isActor:                
                actorIds = "select count(distinct actorid ) from movie_actor where movieid in "+movListStr
                res = self.db.executeQuery(actorIds)
                #print "actorids query = "+str(res)
                actorSet = self.getCount(str(res))
                #print "actorSet = "+str(actorSet)
                totalRelatedWithThisTag = int(actorSet)
            elif self.relation == "genre":
                genres = "select * from mlmovies where movieid in "+movListStr
                res = self.db.executeQuery(genres)
                for genre in res:
                    genreStr = genre[2]
                    genreList = genreStr.split('|')
                    for g in genreList:
                        genreSet.add(g)    
                        
                    
                totalRelatedWithThisTag = len(genreSet)
                #print "genres = "+str(genreSet)
            elif self.relation == "user":
                users = "select count(distinct userid) from mltags where tagid ="+str(tagid)
                countUsers = self.db.executeQuery(users)
                countofUsers = self.getCount(str(countUsers))
                totalRelatedWithThisTag = int(countofUsers)
                
                
                
            #print "totalGenresWithThisTag = "+str(totalRelatedWithThisTag)
            idf = totalDocsCount / totalRelatedWithThisTag
            idf = math.log(idf)
            tagIdVsIDF[tagid]= idf
            tagIdVsTFIDF[tagIdVsName[key]]=tagIdVsTF[tagid] * idf
            #print "tagId = "+str(tagid)
            #print "IDF = "+str(idf)
            
        print "Tag vs TF-IDF "
        sortedTagVsIDF = sorted(tagIdVsTFIDF.items(),key=operator.itemgetter(1),reverse=True)
        print ""+str(sortedTagVsIDF)
        
    def calcSVD(self,movies ):
        isActor = False
        
            
        movieVsWeight = {}
        
        
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
        
        moviesList = []
        noOfMovies = len(movies)
        q = "select count(*) from genome_tags"
        res = self.db.executeQuery(q)
        countStr = self.getCount(str(res[0]))
        noOfTags = int(countStr)
        movieTFIDF = np.zeros((noOfMovies, noOfTags))
        
        movieIndex = 0
        movieIdVsIndex = {}
        for movie in movies:
            movieid = movie[0]
            
            weight = 0
            #Get the tags related to the actor/genre/user
            oldestTagQuery = "select * from mltags where movieid = "+str(movieid)+" order by time_stamp limit 1"
            #print "oldestTagQuery = "+oldestTagQuery
            oldestTagQueryRes = self.db.executeQuery(oldestTagQuery)
            oldestTimeStamp= None
            newestTimeStamp = None
            timeRange = None
            
            for oldTag in oldestTagQueryRes:
                oldestTimeStamp = oldTag[3]
            oldesMillis = self.getMillis(oldestTimeStamp)
                
                
            
            tagsQuery = "select * from mltags where movieid = "+str(movieid)+" order by time_stamp desc"
            tags = self.db.executeQuery(tagsQuery)
            actorTagsCount = len(tags)
            tagVsTotalWeight = {}
            tagVsTF = {}
            tagIdVsTF={}
            taglist=[]
            movieVsTags = {}
            n = 1
            for tag in tags:
                movieid = tag[1]
                tagid = tag[2]
                timestamp = tag[3]
                if n == 1:
                    newestTimeStamp = timestamp
                    timeRange = self.getMillis(newestTimeStamp) - oldesMillis
                    
                taglist.append(tagid)
                tagWeight = self.getWeight(n,actorTagsCount+1)
                if timeRange != 0 :
                    tagWeight = (self.getMillis(timestamp) - oldesMillis) / timeRange
                
                #print "rankweight = "+str(rankWeight)
                n= n+1
                #print "tagWeight = "+str(tagWeight)
                #combinedWeight = (tagWeight + (3 *rankWeight))/4
                combinedWeight = (tagWeight )
                #print "combinedWeight = "+str(combinedWeight)
                if tagid in tagVsTotalWeight:
                    tempWeight = tagVsTotalWeight[tagid]
                    tempWeight +=combinedWeight
                    tagVsTotalWeight[tagid] = tempWeight
                else:
                    tagVsTotalWeight[tagid]=combinedWeight
                # dict ("tagid" ,"weight1,weight 2")
                
            #print "tagVsTotalWeight = "+str(tagVsTotalWeight)
            
            tagsListStr = str(taglist)
            tagsListStr = tagsListStr.replace('[','(')
            tagsListStr = tagsListStr.replace(']',')')
            
            
             
            
                
            
            
            
            totalWeight = 0
            for key,val in tagVsTotalWeight.items():
    #            print "key - "+str(key)
    #            print "weight = "+str(val)
                totalWeight +=val
            
            
            # Calcualting TF for each tag
            for key,val in tagVsTotalWeight.items():
                
                tf = tagVsTotalWeight[key]/totalWeight
                
                tagIdVsTF[key]=tf
                
            
            
            # Calculating IDF
            totalDocsCount = noOfMovies
            
                
                
            #print "Total documenst = "+str(totalDocsCount)
            tagIdVsIDF = {}
            tagIdVsTFIDF={}
            
        
            totalRelatedWithThisTag= 0
            
            for tagid in taglist:
                
                users = "select count(distinct movieid) from mltags where tagid ="+str(tagid)
                countUsers = self.db.executeQuery(users)
                countofUsers = self.getCount(str(countUsers))
                totalRelatedWithThisTag = int(countofUsers)                                        
                #print "totalGenresWithThisTag = "+str(totalRelatedWithThisTag)
                idf = totalDocsCount / totalRelatedWithThisTag
                idf = math.log(idf)
                tagIdVsIDF[tagid]= idf
                tagIdVsTFIDF[tagid]=tagIdVsTF[tagid] * idf
                
            for key,val in tagIdVsTFIDF.items():
                tagid = key
                tfIdf = val
                movieTFIDF[movieIndex][tagIdVsIndex[tagid]]= tfIdf
            movieIdVsIndex[movieid]=movieIndex
            movieIndex = movieIndex+1
                
            
            
        print "Movie tag Matrix"
        print movieTFIDF
        print movieTFIDF[movieIdVsIndex[7247]][tagIdVsIndex[1128]]
        u, s, v = np.linalg.svd(movieTFIDF,full_matrices = False)
        print "S"
        
        print s
        print "V"
        print v
    
    
    
    
    
    
        
    def calcUserVector(self):
        query = "select * from mlratings where userid = "+str(self.entityid)
        movies = self.db.executeQuery(query)
        self.calcTFIDFApproach1(movies)
        
        
    def calcGenreVector(self):
        self.tableName = self.getEntityMovieTableName()
        query = "select * from mlmovies where genres like '%"+str(self.entityid)+"%'"
        #print "query = "+query
        movies = self.db.executeQuery(query)
        self.calcTFIDFApproach1(movies)
        return None
    
    def calcMoviesVector(self):
        self.tableName = self.getEntityMovieTableName()
        query = "select * from mlmovies "
        movies = self.db.executeQuery(query)
        self.calcSVD(movies)
        return None
        
        
    def calcActorVector(self):
        
        tableName = self.getEntityMovieTableName()
        query = "select * from "+tableName+" where actorid = "+str(self.entityid)
        movies = self.db.executeQuery(query)
        #self.calcTFIDFApproach2(movies)
        self.calcTFIDFApproach1(movies)
        
        
            
            
                
                
                
            
                
                
                
                
                    
                
                
                
                
                
            
                
                
                
                
            
            
            
            
            
        
        
                
            
                
            
            
        
        
        
            
            
           
           
           
       
            
             
            
            
            
            
        
    
        
            
            
            
        
        
        
        
        
        
        
    def getWeightedTagVector(self):
        #print "Inside getWeightedTagVector"
        self.tableName = self.getEntityMovieTableName()
        if self.relation == "actor":
            self.calcActorVector()
        elif self.relation == "genre":
            self.calcGenreVector()
        else:
            
            self.calcUserVector()
            
            
    def pDiff1(self,model, genre1,genre2):
        
        #print "Model = "+model
        genre1MovieList = []
        genre1TagsList = []
        genre2MovieList = []
        genre2TagsList = []
        genre1TagVsWeight = {}
        genre2TagVsWeight = {}
        totalMoviesSet = set()
        # Movies associated with Genre 1
        query = "select distinct movieid from mlmovies where genres like '%"+str(genre1)+"%'"
        movies = self.db.executeQuery(query)
        for movie in movies:
            genre1MovieList.append(movie[0])
            totalMoviesSet.add(movie[0])
        noOfMoviesGenre1 = len(genre1MovieList)
        #print "Genre1 movies = "+str(genre1MovieList)
        #print "Count1 = "+str(noOfMoviesGenre1)
        
        # Movies associated with Genre 2
        query = "select distinct movieid from mlmovies where genres like '%"+str(genre2)+"%'"
        movies = self.db.executeQuery(query)
        for movie in movies:
            genre2MovieList.append(movie[0])
            totalMoviesSet.add(movie[0])
        noOfMoviesGenre2 = len(genre2MovieList)
        #print "Count2 = "+str(noOfMoviesGenre2)
        totalMovies = len(totalMoviesSet)
        
        #print "moviesList "+str(genre1MovieList)
        movListStr1 = str(genre1MovieList)
        movListStr1 = movListStr1.replace('[','(')
        movListStr1 = movListStr1.replace(']',')')
            
        movListStr2 = str(genre2MovieList)
        movListStr2 = movListStr2.replace('[','(')
        movListStr2 = movListStr2.replace(']',')')
            
        # tags associated to Genre 1
        query = "select distinct tagid from mltags where movieid in "+movListStr1
        tags = self.db.executeQuery(query)
        tagsList = []
        for tag in tags:
            tagsList.append(tag[0])
            genre1Genre2MoviesForTag = set()
            genre2MoviesForTag = []
            noOfMoviesAssociatedWithThisTagGenre1 = 0
            r = 0
            m= 0 
            R = 0
            M = 0
            if model == "P-DIFF1":
                query = "select distinct movieid from mltags where movieid in "+movListStr1+" and tagid = "+str(tag[0])
                res = self.db.executeQuery(query)
                for movie in res:
                    genre1Genre2MoviesForTag.add(movie[0])
                noOfMoviesAssociatedWithThisTagGenre1 = len(genre1Genre2MoviesForTag)
                
                query = "select distinct movieid from mltags where movieid in "+movListStr2+" and tagid = "+str(tag[0])
                res = self.db.executeQuery(query)
                for movie in res:
                    genre1Genre2MoviesForTag.add(movie[0])
                    genre2MoviesForTag.append(movie[0])                        
                r = noOfMoviesAssociatedWithThisTagGenre1 
                m = len(genre1Genre2MoviesForTag)
                R = noOfMoviesGenre1
                M = totalMovies         
            elif model == "P-DIFF2":
                query = "select count(distinct movieid) from mltags where movieid in "+movListStr2+" and tagid != "+str(tag[0])
                res = self.db.executeQuery(query)
                r = int(self.getCount(str(res)))
                
                query = "select count(distinct movieid) from mltags where movieid in "+movListStr1+" or movieid in "+movListStr2+"and tagid != "+str(tag[0])
                res = self.db.executeQuery(query)
                m = int(self.getCount(str(res)))
                R = noOfMoviesGenre2
                M = totalMovies
            
#            print "tagid = "+str(tag[0])
#            print "r = "+str(r)
#            print "m = "+str(m)
#            print "R = "+str(R)
#            print "M = "+str(M)
        
#            smallmMinusr = smallmMinusr + 0.5
#            R =R + 1
#            r = r+0.5
#            CapMminusR= CapMminusR+1
                
            
#            if tag[0] == 1013:
                
#                print "m = "+str(m)
#                print "r = "+str(r)
#                print "M = "+str(M)
#                print "R = "+str(R)
#                print "m-r = "+str(smallmMinusr)    
            x = float(r + float(m)/float(M))/ float(R+1)    
            y =  float( m-r + float(m)/M) / float(M-R + 1)
            w =  float(((x*(1-y)) / (y * (1-x))) * math.fabs((x -y)))
          
            w = math.log(float(w))
#            if tag[0] == 1013:
#                print "w = "+str(w)
#            num = r / (R - r)
#            denom1 = m-r
#            denom2 = M - m;
#            denom2 = denom2 - R + r
#            denom = denom1 * denom2
#            leftExpression = num / denom
#            right1 = r / R
#            right2  = (m - r) / (M - R)
#            right = math.fabs(right1 - right2)
#            w = leftExpression * right
            genre1TagVsWeight[tag[0]] = w 
        tagsListStr = str(tagsList)
        tagsListStr = tagsListStr.replace('[','(')
        tagsListStr = tagsListStr.replace(']',')')    
        tagsQuery = "select * from genome_tags where tagid in "+tagsListStr
        tags = self.db.executeQuery(tagsQuery)
        tagIdVsName = {}
        for tag in tags:
            tagIdVsName[tag[0]]=tag[1]       
        genre1TagNamVsWeight = {}
        for tagid,weight in genre1TagVsWeight.items():
            genre1TagNamVsWeight[tagIdVsName[tagid]] = weight
        sortedTagVsWeight = sorted(genre1TagNamVsWeight.items(),key=operator.itemgetter(1),reverse=True)
            
        print "Tag vs Weight = "+str(sortedTagVsWeight)
        #print "Genre2 = "+str(genre2TagVsWeight)
        
        # tags associated to Genre 2            
#        query = "select distinct tagid from mltags where movieid in "+movListStr2
#        tags = self.db.executeQuery(query)
#        for tag in tags:
#            genre2genre1MoviesForTag = set()
#            genre1MoviesListForThisTag = []
#            query = "select distinct movieid from mltags where movieid in "+movListStr2+" and tagid = "+str(tag[0])
#            res = self.db.executeQuery(query)
#            for movie in res:
#                genre2genre1MoviesForTag.add(movie[0])
#            noOfMoviesAssociatedWithThisTagGenre2 = len(genre2genre1MoviesForTag)
#            
#            query = "select distinct movieid from mltags where movieid in "+movListStr1+" and tagid = "+str(tag[0])
#            res = self.db.executeQuery(query)
#            for movie in res:
#                genre2genre1MoviesForTag.add(movie[0])
#                genre1MoviesListForThisTag.append(movie[0])
#            noOfMoviesAssociatedWithThisTagGenre2 = len(genre1MoviesListForThisTag)
#            
#            
#            r = noOfMoviesAssociatedWithThisTagGenre2
#            m = len(genre2genre1MoviesForTag)
#            R = noOfMoviesGenre2
#            M = totalMovies       
#     
#            x = (r + float(m)/float(M))/ float(R)    
#            y =  ( m-r + float(m)/M) / float(M-R + 1)
#            w =  ((x*(1-y)) / (y * (1-x))) * (x -y)
#            #w = math.log(w)
#            genre2TagVsWeight[tag[0]] = w
#            print "Genre 2 Tag weight = "+str(genre2TagVsWeight)
#            num = r / (R - r)
#            denom1 = m-r
#            denom2 = M - m;
#            denom2 = denom2 - R + r
#            denom = denom1 * denom2
#            leftExpression = num / denom
#            right1 = r / R
#            right2  = (m - r) / (M - R)
#            right = math.fabs(right1 - right2)
#            w = leftExpression * right
#            genre1TagVsWeight[tag[0]] = w
            
#            print "tagid  = "+str(tag[0])
#            print "r=  = "+str(r)
#            print "m  = "+str(m)
#            print "R  = "+str(R)
#            print "M  = "+str(M)
            #print "weight  = "+str(w)
            
            
            
    
    def tfIdfDiff(self,genre1,genre2):
        query = "select distinct movieid from mlmovies where genres like '%"+str(genre1)+"%'"
        movies = self.db.executeQuery(query)
        moviesList = []
        for movie in movies:
            moviesList.append(movie[0])
        moviesListStr = str(moviesList)
        moviesListStr = moviesListStr.replace('[','(')
        moviesListStr = moviesListStr.replace(']',')')    
        query = "select tagid from mltags where movieid in "+moviesListStr
        totaltags = 0
        tagIdVsFreq = {}
        tagIdVsTF = {}
        tagIdVsIDF = {}
        tagIdVsName = {}
        tagVsTFIDFDIFF = {}
        genre1MovieList = []
        totalMoviesSet = set()
        genre2MovieList= []
        tagsList = []
        
        tags = self.db.executeQuery(query)
        for tag in tags :
            tagsList.append(tag[0])
            totaltags = totaltags + 1
            if tag[0] in tagIdVsFreq.items():
                freq = tagIdVsFreq[tag[0]]
                freq = freq + 1;
                tagIdVsFreq = freq
            else:
                tagIdVsFreq[tag[0]]=1
        for tagid,freq in tagIdVsFreq.items():
            tagIdVsTF[tagid] = tagIdVsFreq[tagid] / totaltags
        
        query = "select distinct movieid from mlmovies where genres like '%"+str(genre1)+"%'"
        movies = self.db.executeQuery(query)
        for movie in movies:
            totalMoviesSet.add(movie[0])
        
        #print "Genre1 movies = "+str(genre1MovieList)
        #print "Count1 = "+str(noOfMoviesGenre1)
        
        # Movies associated with Genre 2
        query = "select distinct movieid from mlmovies where genres like '%"+str(genre2)+"%'"
        movies = self.db.executeQuery(query)
        for movie in movies:
            totalMoviesSet.add(movie[0])
        totalMovies = len(totalMoviesSet)
        
        # IDF Calcualtion
        for tag in tags:
            query = "select count(distinct movieid) from mltags where tagid = "+str(tag[0])
            count = self.db.executeQuery(query)
            moviesWithThisTag = self.getCount(str(count))
            tagIdVsIDF[tag[0]] = math.log(totalMovies / int(moviesWithThisTag))
            
        tagsListStr = str(tagsList)
        tagsListStr = tagsListStr.replace('[','(')
        tagsListStr = tagsListStr.replace(']',')')    
        tagsQuery = "select * from genome_tags where tagid in "+tagsListStr
        tags = self.db.executeQuery(tagsQuery)
        tagIdVsName = {}
        for tag in tags:
            tagIdVsName[tag[0]]=tag[1]           
        
        for tagid,val in tagIdVsTF.items():
            tagVsTFIDFDIFF[tagIdVsName[tagid]] = tagIdVsTF[tagid] * tagIdVsIDF[tagid]
             
        sortedTagVsWeight = sorted(tagVsTFIDFDIFF.items(),key=operator.itemgetter(1),reverse=True)
        print "Tag vs TF-IDF-DIFF = "+str(sortedTagVsWeight)
             
             
            
       


    


