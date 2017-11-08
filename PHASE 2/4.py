import sys
import pandas
import numpy
import math
def SuggestMovies(userpreferences):

    allusermoviesdata = pandas.read_csv("mlratings.csv")

    allusermoviesdata = allusermoviesdata[['userid', 'movieid', 'rating']]

    allusermoviesdata = allusermoviesdata[allusermoviesdata['rating'] > 3]

    # allusermoviesdata = allusermoviesdata[allusermoviesdata['userid'].isin([6, 18, 29])]

    # print(allusermoviesdata)

    actor_rating_matrix = allusermoviesdata.pivot_table(index='userid', columns='movieid', values='rating', fill_value=0)

    givenuserdata = pandas.DataFrame({'userid': 0, 'movieid': userpreferences, 'rating': 5})

    # print(givenuserdata)

    additionalgivenuserdata = pandas.DataFrame({'userid': 0, 'movieid': allusermoviesdata['movieid'].tolist(), 'rating': 0})

    givenuserdata = pandas.concat([givenuserdata, additionalgivenuserdata])

    # print(mergeddata1)

    givenuserdata = givenuserdata[['userid', 'movieid', 'rating']].drop_duplicates(subset=['userid', 'movieid'])

    given_user_rating_matrix = givenuserdata.pivot_table(index='userid', columns='movieid', values='rating', fill_value=0)

    # print(given_user_rating_matrix)
    #
    # print(actor_rating_matrix.transpose())

    givenactoractorsimilarity = numpy.matmul(given_user_rating_matrix, actor_rating_matrix.transpose());

    # print(givenactoractorsimilarity)

    userdetails = sorted(numpy.unique(allusermoviesdata['userid'].tolist()))

    # print(userdetails)

    similardetails = []
    for i in range(0, len(givenactoractorsimilarity[0])):
        similardetails.append([userdetails[i],givenactoractorsimilarity[0][i]])

    similardetails = sorted(similardetails, key=lambda x: x[1], reverse=True)

    # print(similardetails)

    masterlist = []
    for i in range(0, len(similardetails)):
        # print(similardetails[i][0])

        if(similardetails[i][1] > 0):
            movieslist = allusermoviesdata[allusermoviesdata['userid'] == similardetails[i][0]]['movieid'].tolist()
            # print(movieslist)
            masterlist.extend(movieslist)

            suggested = set(masterlist).difference(userpreferences)
            if(len(suggested) > 10):
                break;

    moviesdata = pandas.read_csv("mlmovies.csv")

    print("\nUser has watched {0}".format(moviesdata[moviesdata['movieid'].isin(userpreferences)]['moviename'].tolist()))

    print("\nUser could watch {0}".format(moviesdata[moviesdata['movieid'].isin(suggested)]['moviename'].tolist()))


SuggestMovies([3216, 3366, 5076])
