import sys
import pandas
import numpy
import math
import datetime
import operator

def ComputeRankWeightage(row):
    return (1+ row['max_actor_rank'] - row['actor_movie_rank'])/(1+ row['max_actor_rank'] - row['min_actor_rank'])

def ComputeTimestampWeights(row, min_timestamp, max_timestamp):
    return ((pandas.to_datetime(row['timestamp'])-min_timestamp).days + 1)/((max_timestamp-min_timestamp).days+1)

def AddWeightages(row):
    return numpy.round(row['actor_rank_weightage'] + row['timestamp_weightage'], decimals=4)

def ComputeTF(row):
    return row['tag_weightage'] / row['total_actor_weightage']

def ProcessWeightsToTF(combineddata):

    combineddata['all_weightages'] = combineddata.apply(AddWeightages, axis=1)
    combineddata['tag_weightage'] = combineddata.groupby(['actorid','tagid'])['all_weightages'].transform('sum')
    combineddata = combineddata[['actorid', 'tagid', 'tag_weightage']].drop_duplicates(subset=['actorid', 'tagid'])
    combineddata['total_actor_weightage'] = combineddata.groupby(['actorid'])['tag_weightage'].transform('sum')
    combineddata.drop_duplicates(subset=['actorid'], inplace=True)
    combineddata['tf'] = combineddata.apply(ComputeTF, axis=1)
    return combineddata

def ComputeIDF(row, total_actors):
    return math.log10(total_actors / row['count_of_actors'])

def ComputeTFIDF(row):
    return row['tf']*row['idf']

def ProcessTFandIDFtoTFIDF(tfdata, idfdata):
    tfidfdata = tfdata.merge(idfdata, on='tagid')
    tfidfdata['tfidf'] = tfidfdata.apply(ComputeTFIDF, axis=1)
    return tfidfdata[['actorid','tagid','tfidf']]

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

    specificactortagsdata['idf'] = specificactortagsdata.apply(ComputeIDF, axis=1, total_actors=total_actors)

    # print(total_actors)
    # print(specificactortagsdata)

    tfidfdata = ProcessTFandIDFtoTFIDF(tfdata, specificactortagsdata[['tagid', 'idf']])

    # print(tfdata)
    # print(tfidfdata)

    return tfidfdata

def GenerateGivenActorVector(tfidfdata, actorid):

    duplicatetags = pandas.DataFrame({'actorid': actorid, 'tagid': numpy.unique(tfidfdata['tagid'].tolist()), 'tfidf': 0})

    actortagsvector = pandas.concat([tfidfdata[tfidfdata['actorid'].isin([actorid])], duplicatetags])

    # print(actortagsvector)

    actortagsvector = actortagsvector[['actorid', 'tagid', 'tfidf']].drop_duplicates(subset=['tagid', 'actorid'])

    return actortagsvector

def GetActorDetails(tfidf):

    actordetails = pandas.read_csv("imdb-actor-info.csv")
    actordetails = actordetails[actordetails['id'].isin(tfidf['actorid'].tolist())]
    actornamelist = actordetails.values.tolist()
    actornamelist = sorted(actornamelist, key=lambda x: x[0])
    return actornamelist

def PrintRelatedActors(actorid, actoractorsimilarity,actornamelist):

    relatedactor = []

    for i in range (0, len(actornamelist)):
        if(actorid != actornamelist[i][0] and actoractorsimilarity[0][i] > 0):
            relatedactor.append((actornamelist[i][0], actornamelist[i][1], actoractorsimilarity[0][i]))
    print(relatedactor)

    relatedactor = sorted(relatedactor, reverse=1, key = lambda x:x[2])

    print("Top {0} related actors to actor {1} \n".format(min(10,len(relatedactor)), actorid))

    for i in range(0, min(10,len(relatedactor))):
        print("ID: {0}, Name: {1}".format(relatedactor[i][0],relatedactor[i][1]))


def GetRelatedActorsWithTFIDFVectors(actorid):

    tfidfdata = GenerateAllActorsTFIDF()

    actortagsvector = GenerateGivenActorVector(tfidfdata, actorid)

    given_actor_tag_matrix = actortagsvector.pivot_table(index='actorid', columns='tagid', values='tfidf', fill_value=0)

    # print(given_actor_tag_matrix)

    actor_tag_matrix = tfidfdata.pivot_table(index='actorid', columns='tagid', values='tfidf', fill_value=0)

    # print(actor_tag_matrix)

    transpose_actor_tag_matrix = actor_tag_matrix.transpose()

    actoractorsimilarity = numpy.matmul(given_actor_tag_matrix, transpose_actor_tag_matrix);

    # print(actoractorsimilarity)

    actornamelist = GetActorDetails(tfidfdata)

    PrintRelatedActors(actorid, actoractorsimilarity, actornamelist)

def PrintRelatedActorsWithConcepts(actorid, latentsematics, u, tfidf):

    actornamelist = GetActorDetails(tfidf)

    actorindex = 0

    for i in range (0, len(actornamelist)):
        if(actornamelist[i][0] == actorid):
            actorindex = i

    given_actor_tags = u[actorindex]

    actoractorlatentsimilarity = numpy.matmul(given_actor_tags, u.transpose());

    relatedactor = []

    for i in range (0, len(actornamelist)):
        if(actornamelist[i][0] != actorid  and actoractorlatentsimilarity[i] > 0):
            relatedactor.append((actornamelist[i][0], actornamelist[i][1], actoractorlatentsimilarity[i]))
    # print(relatedactor)

    relatedactor = sorted(relatedactor, reverse=1, key = lambda x:x[2])

    print("Top {0} related actors to actor {1} \n".format(min(10,len(relatedactor)), actorid))

    for i in range(0, min(10,len(relatedactor))):
        print("ID: {0}, Name: {1}".format(relatedactor[i][0],relatedactor[i][1]))


def GetRelatedActorsWithSVD(actorid):

    tfidfdata = GenerateAllActorsTFIDF()

    actor_tag_matrix = tfidfdata.pivot_table(index='actorid', columns='tagid', values='tfidf', fill_value=0)

    # print(actor_tag_matrix)

    u, s, vt = numpy.linalg.svd(actor_tag_matrix, full_matrices=False)

    # print(u)
    # print(s)
    # print(vt)

    PrintRelatedActorsWithConcepts(actorid, s.size, u, tfidfdata)

# GetRelatedActorsWithSVD(1860883)
# GetRelatedActorsWithTFIDFVectors(128645)