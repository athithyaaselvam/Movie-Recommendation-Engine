import sys
import pandas
import numpy
import math
import datetime

def ComputeRankWeightage(row):
    return (1+ row['max_actor_rank'] - row['actor_movie_rank'])/(1+ row['max_actor_rank'] - row['min_actor_rank'])

def ComputeTimestampWeights(row, min_timestamp, max_timestamp):
    return ((pandas.to_datetime(row['timestamp'])-min_timestamp).days + 1)/((max_timestamp-min_timestamp).days+1)

def AddWeightages(row):
    return numpy.round(row['actor_rank_weightage'] + row['timestamp_weightage'], decimals=4)

def ComputeActorTF(row):
    return row['tag_weightage'] / row['total_actor_weightage']

def ComputeMovieTF(row):
    return row['tag_weightage'] / row['total_movie_weightage']

def ProcessWeightsToTF(combineddata):

    combineddata['all_weightages'] = combineddata.apply(AddWeightages, axis=1)
    combineddata['tag_weightage'] = combineddata.groupby(['actorid','tagid'])['all_weightages'].transform('sum')
    combineddata = combineddata[['actorid', 'tagid', 'tag_weightage']].drop_duplicates(subset=['actorid', 'tagid'])
    combineddata['total_actor_weightage'] = combineddata.groupby(['actorid'])['tag_weightage'].transform('sum')
    combineddata['tf'] = combineddata.apply(ComputeActorTF, axis=1)
    return combineddata

def ComputeActorIDF(row, total_actors):
    return math.log10(total_actors / row['count_of_actors'])

def ComputeMoviesIDF(row, total_movies):
    return math.log10(total_movies / row['count_of_movies'])

def ComputeTFIDF(row):
    return row['tf']*row['idf']

def ProcessActorTFandIDFtoTFIDF(tfdata, idfdata):
    tfidfdata = tfdata.merge(idfdata, on='tagid')
    tfidfdata['tfidf'] = tfidfdata.apply(ComputeTFIDF, axis=1)
    return tfidfdata[['actorid','tagid','tfidf']]

def ProcessMovieTFandIDFtoTFIDF(tfdata, idfdata):
    tfidfdata = tfdata.merge(idfdata, on='tagid')
    tfidfdata['tfidf'] = tfidfdata.apply(ComputeTFIDF, axis=1)
    return tfidfdata[['movieid','tagid','tfidf']]

def GenerateAllActorsTFIDF():

    allactormoviesdata =pandas.read_csv("movie-actor.csv")

    allmoviestagsdata = pandas.read_csv("mltags.csv")

    allactormoviesdata['max_actor_rank'] = allactormoviesdata.groupby(['movieid'])['actor_movie_rank'].transform(max)
    allactormoviesdata['min_actor_rank'] = allactormoviesdata.groupby(['movieid'])['actor_movie_rank'].transform(min)

    allactormoviesdata['actor_rank_weightage'] = allactormoviesdata.apply(ComputeRankWeightage, axis=1)

    min_timestamp = pandas.to_datetime(min(allmoviestagsdata['timestamp']))
    max_timestamp = pandas.to_datetime(max(allmoviestagsdata['timestamp']))

    allmoviestagsdata['timestamp_weightage'] = allmoviestagsdata.apply(ComputeTimestampWeights, axis=1, args=(min_timestamp, max_timestamp))

    combineddata = allactormoviesdata[['actorid','movieid','actor_rank_weightage']].merge(allmoviestagsdata[['movieid','tagid','timestamp_weightage']], on='movieid')

    tfdata = ProcessWeightsToTF(combineddata)

    taglist = tfdata['tagid'].tolist()
    alltagsdata = pandas.read_csv("mltags.csv")
    specifictagsdata = alltagsdata[alltagsdata['tagid'].isin(taglist)]

    allactormoviesdata = pandas.read_csv("movie-actor.csv")
    specificactortagsdata = specifictagsdata.merge(allactormoviesdata, on='movieid')

    specificactortagsdata.drop_duplicates(subset=['tagid', 'actorid'], inplace=True)
    specificactortagsdata['count_of_actors'] = specificactortagsdata.groupby('tagid')['actorid'].transform('count')
    specificactortagsdata.drop_duplicates(subset=['tagid'], inplace=True)

    actordata = pandas.read_csv("imdb-actor-info.csv")
    total_actors = actordata.shape[0]

    specificactortagsdata['idf'] = specificactortagsdata.apply(ComputeActorIDF, axis=1, total_actors=total_actors)

    # print(total_actors)
    # print(specificactortagsdata)

    tfidfdata = ProcessActorTFandIDFtoTFIDF(tfdata, specificactortagsdata[['tagid', 'idf']])

    # print(tfdata)
    # print(tfidfdata)

    # tfidfdata = tfidfdata[tfidfdata['actorid'].isin([61523,1014988,1606367,277151,542238])]

    return tfidfdata

def GetActorDetails(tfidf):

    actordetails = pandas.read_csv("imdb-actor-info.csv")
    actordetails = actordetails[actordetails['id'].isin(tfidf['actorid'].tolist())]
    actornamelist = actordetails.values.tolist()
    actornamelist = sorted(actornamelist, key=lambda x: x[0])
    return actornamelist

def GenerateGivenMovieVector(movieid):

    allmovietagsdata =pandas.read_csv("mltags.csv")

    actorsingivenmovie = allmovietagsdata[allmovietagsdata['movieid'].isin([movieid])]

    min_timestamp = pandas.to_datetime(min(actorsingivenmovie['timestamp']))
    max_timestamp = pandas.to_datetime(max(actorsingivenmovie['timestamp']))

    actorsingivenmovie['timestamp_weightage'] = actorsingivenmovie.apply(ComputeTimestampWeights, axis=1, args=(min_timestamp, max_timestamp))

    actorsingivenmovie['tag_weightage'] = actorsingivenmovie.groupby(['tagid'])['timestamp_weightage'].transform('sum')

    actorsingivenmovie = actorsingivenmovie[['movieid','tagid','tag_weightage']].drop_duplicates(subset=['tagid'])

    actorsingivenmovie['total_movie_weightage'] = actorsingivenmovie.groupby(['movieid'])['tag_weightage'].transform('sum')

    print(actorsingivenmovie)

    actorsingivenmovie['tf'] = actorsingivenmovie.apply(ComputeMovieTF, axis=1)

    print(actorsingivenmovie)

    taglist = actorsingivenmovie['tagid'].tolist()
    alltagsdata = pandas.read_csv("mltags.csv")
    specifictagsdata = alltagsdata[alltagsdata['tagid'].isin(taglist)]

    specifictagsdata.drop_duplicates(subset=['tagid', 'movieid'], inplace=True)
    specifictagsdata['count_of_movies'] = specifictagsdata.groupby('tagid')['movieid'].transform('count')
    specifictagsdata.drop_duplicates(subset=['tagid'], inplace=True)

    moviesdata = pandas.read_csv("mlmovies.csv")
    total_movies = moviesdata.shape[0]

    specifictagsdata['idf'] = specifictagsdata.apply(ComputeMoviesIDF, axis=1, total_movies=total_movies)

    # print(total_actors)
    # print(specificactortagsdata)

    tfidfdata = ProcessMovieTFandIDFtoTFIDF(actorsingivenmovie, specifictagsdata[['tagid', 'idf']])

    # filteredtfidfdata = tfidfdata[tfidfdata['actorid'].isin(actorsingivenmovie)]

    return tfidfdata


def PrintRelatedActors(movieid, movieactorsimilarity, actornamelist):

    relatedactor = []

    allmovieactorsdata =pandas.read_csv("movie-actor.csv")

    actorsingivenmovie = allmovieactorsdata[allmovieactorsdata['movieid'].isin([movieid])]['actorid'].tolist()

    print(actorsingivenmovie)
    for i in range (0, len(actornamelist)):
        if(actornamelist[i][0] not in actorsingivenmovie  and movieactorsimilarity[0][i] > 0):
        # if(movieactorsimilarity[0][i] > 0):
            relatedactor.append((actornamelist[i][0], actornamelist[i][1], movieactorsimilarity[0][i]))
    print(relatedactor)

    relatedactor = sorted(relatedactor, reverse=1, key = lambda x:x[2])

    print("Top {0} related actors to actor {1} \n".format(min(10,len(relatedactor)), movieid))

    for i in range(0, min(10,len(relatedactor))):
        print("ID: {0}, Name: {1}".format(relatedactor[i][0],relatedactor[i][1]))

def PrintRelatedActorsWithConcepts(actorid, latentsematics, vt, tfidf):

    actornamelist = GetActorDetails(tfidf)

    relatedactor = []

    for i in range(0, min(10,latentsematics)):
        latentsemanticrow = vt[i]
        actorsrelatedtoconcepts = []
        mean = numpy.mean(latentsemanticrow)
        for j in range(0, len(actornamelist)):
            if(latentsemanticrow[j] >= mean):
                actorsrelatedtoconcepts.append(actornamelist[j][0])
        # print(actorsrelatedtoconcepts)
        if(actorid in actorsrelatedtoconcepts):
            relatedactor.extend(actorsrelatedtoconcepts)

    print(numpy.unique(relatedactor))

def GetRelatedActorsWithTFIDFVectors(movieid):

    tfidfdata = GenerateAllActorsTFIDF()

    # print(tfidfdata)

    movieactortagsvector = GenerateGivenMovieVector(movieid)

    print(movieactortagsvector)

    given_movie_tag_matrix = movieactortagsvector.pivot_table(index='movieid', columns='tagid', values='tfidf', fill_value=0)

    print(given_movie_tag_matrix)

    requiredtfidfdata = tfidfdata[tfidfdata['tagid'].isin(movieactortagsvector['tagid'].tolist())]

    print(requiredtfidfdata)

    all_actors_tag_matrix = requiredtfidfdata.pivot_table(index='actorid', columns='tagid', values='tfidf', fill_value=0)

    print(all_actors_tag_matrix)

    all_tag_actors_matrix = all_actors_tag_matrix.transpose()

    movieactorsimilarity = numpy.matmul(given_movie_tag_matrix, all_tag_actors_matrix);

    print(movieactorsimilarity)

    actornamelist = GetActorDetails(requiredtfidfdata)

    print(actornamelist)

    PrintRelatedActors(movieid, movieactorsimilarity, actornamelist)

def GetRelatedActorsWithSVD(actorid):

    tfidfdata = GenerateAllActorsTFIDF()

    actor_tag_matrix = tfidfdata.pivot_table(index='actorid', columns='tagid', values='tfidf', fill_value=0)

    u, s, vt = numpy.linalg.svd(actor_tag_matrix.transpose(), full_matrices=False)

    print(u)
    print(s)
    print(vt)

    PrintRelatedActorsWithConcepts(actorid, s.size, vt, tfidfdata)

GetRelatedActorsWithTFIDFVectors(4252)
# GetRelatedActorsWithSVD(4252)
