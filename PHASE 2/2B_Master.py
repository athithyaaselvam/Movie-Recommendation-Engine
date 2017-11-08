import sys
import pandas
import numpy
import math
import datetime

def GenerateCoActorCoActorMatrix(allactormoviesdata):

    allactormoviesdata['value'] = 1

    actor_movie_matrix = allactormoviesdata.pivot_table(index='actorid', columns='movieid', values='value', fill_value=0)

    # print(actor_movie_matrix)

    transpose_actor_movie_matrix = actor_movie_matrix.transpose()

    coactorcoactorimilarity = numpy.matmul(actor_movie_matrix, transpose_actor_movie_matrix);

    numpy.fill_diagonal(coactorcoactorimilarity, 0)

    # print(coactorcoactorimilarity)

    return coactorcoactorimilarity

def GetActorDetails(actordata):

    actordetails = pandas.read_csv("imdb-actor-info.csv")
    actordetails = actordetails[actordetails['id'].isin(actordata['actorid'].tolist())]
    actornamelist = actordetails.values.tolist()
    actornamelist = sorted(actornamelist, key=lambda x: x[0])
    return actornamelist

def PrintNonOverlappingActors(latentsematics,vt,actornamelist):

    v = vt.transpose();

    nvlist = {0 : [], 1 : [], 2 : [] }

    modified_v = numpy.delete(v, numpy.s_[3:latentsematics], axis=1)
    for i in range(0, len(actornamelist)):
        latentsemanticrow = modified_v[i]
        max = numpy.nanargmax(latentsemanticrow);
        nvlist[max].append(i)

    for i in nvlist.keys():
        print("Actors in group {0}:\n".format(i+1))
        eachgrouplist = nvlist[i]
        for j in eachgrouplist:
            print ("{0} ({1}) \n". format(actornamelist[j][1],actornamelist[j][0]))

def PrintLatentSematics(latentsematics,vt,actornamelist):

    for i in range(0, min(latentsematics, 3)):
        latentsemanticrow = vt[i]
        mean = numpy.mean(latentsemanticrow)
        print("\nLatent semantic {0}".format(i+1))
        for j in range(0, len(actornamelist)):
            if(latentsemanticrow[j] >= mean):
                print("{0} ({1})".format(actornamelist[j][0], actornamelist[j][1]))

def FindSemanticsWithSVD():

    allactormoviesdata =pandas.read_csv("movie-actor.csv")

    coactorcoactorimilarity = GenerateCoActorCoActorMatrix(allactormoviesdata)

    u, s, vt = numpy.linalg.svd(coactorcoactorimilarity, full_matrices=False)

    # print(u)
    # print(s)
    # print(vt)

    actornamelist = GetActorDetails(allactormoviesdata)

    PrintLatentSematics(s.size, vt, actornamelist)

    PrintNonOverlappingActors(s.size, vt, actornamelist)


FindSemanticsWithSVD()
