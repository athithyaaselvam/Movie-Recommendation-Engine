import sys
import pandas
import numpy
import math
import warnings
from sklearn import tree
from sklearn import decomposition
from collections import Counter

warnings.filterwarnings("ignore")

def ComputeTimestampWeights(row, min_timestamp, max_timestamp):
    return ((pandas.to_datetime(row['timestamp'])-min_timestamp).days + 1)/((max_timestamp-min_timestamp).days+1)

def ComputeTFIDF(row):
    return row['tf']*row['idf']

def ComputeMovieTF(row):
    return row['tag_weightage'] / row['total_movie_weightage']

def ComputeMoviesIDF(row, total_movies):
    return math.log10(total_movies / row['count_of_movies'])

def ProcessMovieTFandIDFtoTFIDF(tfdata, idfdata):
    tfidfdata = tfdata.merge(idfdata, on='tagid')
    tfidfdata['tfidf'] = tfidfdata.apply(ComputeTFIDF, axis=1)
    return tfidfdata[['movieid','tagid','tfidf']]

def GetMoviesTagsData():

    allmovietagsdata =pandas.read_csv("data/mltags.csv")

    min_timestamp = pandas.to_datetime(min(allmovietagsdata['timestamp']))
    max_timestamp = pandas.to_datetime(max(allmovietagsdata['timestamp']))

    allmovietagsdata['timestamp_weightage'] = allmovietagsdata.apply(ComputeTimestampWeights, axis=1, args=(min_timestamp, max_timestamp))

    allmovietagsdata['tag_weightage'] = allmovietagsdata.groupby(['movieid','tagid'])['timestamp_weightage'].transform('sum')

    allmovietagsdata = allmovietagsdata[['movieid','tagid','tag_weightage']].drop_duplicates(subset=['movieid','tagid'])

    allmovietagsdata['total_movie_weightage'] = allmovietagsdata.groupby(['movieid'])['tag_weightage'].transform('sum')

    allmovietagsdata['tf'] = allmovietagsdata.apply(ComputeMovieTF, axis=1)

    taglist = allmovietagsdata['tagid'].tolist()
    alltagsdata = pandas.read_csv("data/mltags.csv")
    specifictagsdata = alltagsdata[alltagsdata['tagid'].isin(taglist)]

    specifictagsdata.drop_duplicates(subset=['tagid', 'movieid'], inplace=True)
    specifictagsdata['count_of_movies'] = specifictagsdata.groupby('tagid')['movieid'].transform('count')
    specifictagsdata.drop_duplicates(subset=['tagid'], inplace=True)

    moviesdata = pandas.read_csv("data/mlmovies.csv")
    total_movies = moviesdata.shape[0]

    specifictagsdata['idf'] = specifictagsdata.apply(ComputeMoviesIDF, axis=1, total_movies=total_movies)

    tfidfdata = ProcessMovieTFandIDFtoTFIDF(allmovietagsdata, specifictagsdata[['tagid', 'idf']])

    return tfidfdata

def GetMoviesDetails(movielist):

    moviedetails = pandas.read_csv("data/mlmovies.csv")
    moviedetails = moviedetails[moviedetails['movieid'].isin(movielist)]
    movienamelist = moviedetails.values.tolist()
    movienamelist = sorted(movienamelist, key=lambda x: x[0])
    return movienamelist

def GetLabelDetails():
    labeldetails = pandas.read_csv("data/mllabels.csv")
    labellist = labeldetails.values.tolist()
    labellist = sorted(labellist, key=lambda x: x[0])
    return labellist

def rNNClassifier(k):

    print("Finding label using rNNClassifier\n")

    labelsdetaillist = GetLabelDetails()

    labelledmovieslist = sorted(numpy.unique([movie[0] for movie in labelsdetaillist]))
    # print(labelledmovieslist)

    tfidf = GetMoviesTagsData()

    movieslist = sorted(numpy.unique(tfidf['movieid'].tolist()))

    moviedetaillist = GetMoviesDetails(movieslist)

    movie_tag_matrix = tfidf.pivot_table(index='movieid', columns='tagid', values='tfidf', fill_value=0)

    newlabellist =[]

    inputmovielist = list(set(movieslist) - set(labelledmovieslist))

    for movieid in inputmovielist:

        givenmovieindex = movieslist.index(movieid)

        # print(givenmovieindex)

        given_movie_tags = movie_tag_matrix.values[givenmovieindex]
        # print("{0}-{1} \n{2}\n".format(givenmovieindex,movieslist[givenmovieindex],given_movie_tags))

        relatedmovies = []
        topklabels = []

        for i in range (0, len(labelsdetaillist)):

            labeledmovieindex = movieslist.index(labelsdetaillist[i][0])
            labeled_movie_tags = movie_tag_matrix.values[labeledmovieindex]
            # labeled_movie_tags = u[labeledmovieindex]
            # print("{0}-{1}\n {2}\n".format(labeledmovieindex,movieslist[labeledmovieindex],labeled_movie_tags))
            moviemovielatentsimilarity = numpy.matmul(given_movie_tags, labeled_movie_tags.transpose());
            relatedmovies.append((moviedetaillist[labeledmovieindex][0], moviedetaillist[labeledmovieindex][1], moviemovielatentsimilarity, labelsdetaillist[i][1]))

        relatedmovies = sorted(relatedmovies, reverse=1, key = lambda x:x[2])

        # print(relatedmovies)
        # print("Top {0} neighbours to movie[{1}]\n".format(k, movieid))

        for i in range(0, min(len(relatedmovies),int(k))):
            # print("ID: {0}, Name: {1}, Similarity: {2} Label: {3}".format(relatedmovies[i][0],relatedmovies[i][1],relatedmovies[i][2],relatedmovies[i][3]))
            topklabels.append(relatedmovies[i][3])

        toplabel,times= Counter(topklabels).most_common(1)[0]
        # print("The movie {0} is classified to label [{1}]\n".format(movieid, toplabel))
        newlabellist.append([movieid,toplabel])

    newlabels = pandas.DataFrame(newlabellist,columns=['movieid','label'])

    print(newlabels)

    newlabels.to_csv('labelsRNN.csv', index=False)


