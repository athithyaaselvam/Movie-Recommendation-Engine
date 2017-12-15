import sys
import pandas
import numpy
import math
import datetime
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac


movieActor = pandas.read_csv("data/movie-actor.csv")
mlmovies = pandas.read_csv("data/mlmovies.csv")
mlratings = pandas.read_csv("data/mlratings.csv")
movieActorData = movieActor[['movieid','actorid']]
movieYearData = mlmovies[['movieid','year']]
requiredFinalData = pandas.read_csv('data/mltags.csv' , nrows = 100)
usersUniqueList = sorted(requiredFinalData['userid'].unique())
moviesUniqueList = sorted(requiredFinalData['movieid'].unique())
tagUniqueList = sorted(requiredFinalData['tagid'].unique())
p = len(usersUniqueList)
q = len(moviesUniqueList)
r = len(tagUniqueList)
matrix = numpy.zeros((p,q,r))
for i in range(0, len(requiredFinalData)):
    data = requiredFinalData.iloc[i]
    userIndex = usersUniqueList.index(data['userid'])
    movieIndex = moviesUniqueList.index(data['movieid'])
    tagIndex = tagUniqueList.index(data['tagid'])
    matrix[userIndex][movieIndex][tagIndex] = 1
X = tl.tensor(matrix)
tensorfactors = parafac(X, rank = 5)
userSem = tensorfactors[0]
movieSem = tensorfactors[1]
tagSem = tensorfactors[2]

def movieRecom(userId, userSem, movieSem):
    inputUserIndex = usersUniqueList.index(userId)
    userVec = userSem[inputUserIndex,:]
    print(userVec.shape,movieSem.shape)
    moviesemT = movieSem.T
    magnitude = np.matmul(userVec.asnumpy,moviesemT.asnumpy)
    sortIndex = np.argsort(-magnitude)
    sorted_values = magnitude[np.argsort(-magnitude)]
    similarMovie = pandas.DataFrame(sortIndex,columns={'index'})
    similarMovie['weight'] = sorted_values
    print('movie watched')
    taggedMovie = requiredFinalData[requiredFinalData['userid']==userId]['movieid']
    ratedMovie = mlratings[mlratings['userid']==userId]['movieid']
    watchedMovie = taggedMovie.append(ratedMovie).unique()
    watchedMovieDetails = mlmovies[mlmovies['movieid'].isin(watchedMovie)]
    print(watchedMovieDetails)
    recommendedMovies = []
    for i in range(0,10):
        recommendedMovies.append(moviesUniqueList[sortIndex[i]])
    recommendedMoviesDetails = mlmovies[mlmovies['movieid'].isin(recommendedMovies)]

    print(watchedMovie, recommendedMoviesDetails)

#
# movieRecom(146, userSem, movieSem)