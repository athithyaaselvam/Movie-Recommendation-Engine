import sys
import pandas
import numpy
import math
import warnings
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

warnings.filterwarnings("ignore")


def ComputeTimestampWeights(row, min_timestamp, max_timestamp):
    return ((pandas.to_datetime(row['timestamp'])-min_timestamp).days + 1)/((max_timestamp-min_timestamp).days+1)

def ComputeTF(row):
    return row['tag_weightage'] / row ['total_weightage']

def ComputeIDF(row, total_movies):
    return math.log10(total_movies / row ['count_of_movies'])

def ComputeTFIDF(row):
    return row['tf']*row['idf']

def ProcessTFandIDFtoTFIDF(tfdata, idfdata):
    tfidfdata = tfdata.merge(idfdata, on='tagid')
    tfidfdata['tfidf'] = tfidfdata.apply(ComputeTFIDF, axis=1)
    return tfidfdata[['movieid','tagid','tfidf']]

def GetMoviesTagsData(genre):

    allgenremoviesdata = pandas.read_csv("mlmovies.csv")

    movieslist = allgenremoviesdata[allgenremoviesdata['genres'].str.contains(genre)]['movieid'].tolist()

    allmoviestagsdata = pandas.read_csv("mltags.csv")

    combineddata = allmoviestagsdata[allmoviestagsdata['movieid'].isin(movieslist)]

    min_timestamp = pandas.to_datetime(min(combineddata['timestamp']))
    max_timestamp = pandas.to_datetime(max(combineddata['timestamp']))

    combineddata['weight_of_timestamp'] = combineddata.apply(ComputeTimestampWeights, axis=1, args=(min_timestamp, max_timestamp))

    combineddata['tag_weightage'] = combineddata.groupby(['movieid','tagid'])['weight_of_timestamp'].transform('sum')

    combineddata = combineddata[['movieid','tagid','tag_weightage']].drop_duplicates(subset=['movieid','tagid'])

    combineddata['total_weightage'] = combineddata.groupby('movieid')['tag_weightage'].transform('sum')
    combineddata['tf'] = combineddata.apply(ComputeTF, axis=1)

    # print(combineddata.sort_values(['movieid','tagid'], ascending=False))
    tfdata = combineddata[['movieid', 'tagid', 'tf']].sort_values(['movieid','tagid'], ascending=False)
    #
    # print(tfdata)

#IDF starts
    total_movies = allgenremoviesdata.shape[0]

    taglist = tfdata['tagid'].tolist()
    specifictagsdata = allmoviestagsdata[allmoviestagsdata['tagid'].isin(taglist)]
    specifictagsdata.drop_duplicates(subset=['tagid', 'movieid'], inplace=True)
    specifictagsdata['count_of_movies'] = specifictagsdata.groupby('tagid')['movieid'].transform('count')
    specifictagsdata.drop_duplicates(subset=['tagid'], inplace=True)
    specifictagsdata['idf'] = specifictagsdata.apply(ComputeIDF, axis=1, total_movies=total_movies)
    #
    # print(specifictagsdata)

    tfidfdata = ProcessTFandIDFtoTFIDF(tfdata[['movieid', 'tagid', 'tf']], specifictagsdata[['tagid', 'idf']])

    # print(tfidfdata)

    # idfdata = idfdata[idfdata['tagid'].isin([19,305,849,1127,792,807, 489,824,903])]

    return tfidfdata

def GetTagDetails(tfidf):

    tagdetails = pandas.read_csv("genome-tags.csv")
    tagdetails = tagdetails[tagdetails['tagId'].isin(tfidf['tagid'].tolist())]
    tagnamelist = tagdetails.values.tolist()
    tagnamelist = sorted(tagnamelist, key=lambda x: x[0])
    return tagnamelist

def PrintLatentSematics(latentsematics,vt,tfidf):

    tagnamelist = GetTagDetails(tfidf)

    for i in range(0, min(latentsematics, 4)):
        latentsemanticrow = vt[i]
        mean = numpy.mean(latentsemanticrow)
        print("\nLatent semantic {0}".format(i+1))
        for j in range(0, len(tagnamelist)):
            if(latentsemanticrow[j] >= mean):
                print("{0} ({1})".format(tagnamelist[j][0], tagnamelist[j][1]))

def FindSemanticsWithSVD(genre):

    print("Finding Latent Semantics using SVD\n")

    tfidf= GetMoviesTagsData(genre)

    # print(tfidf)

    matrix = tfidf.pivot_table(index='movieid', columns='tagid', values='tfidf', fill_value=0)

    u, s, vt = numpy.linalg.svd(matrix, full_matrices=False)

    # print(u)
    # print(s)
    # print(vt)

    PrintLatentSematics(s.size, vt, tfidf)

def FindSemanticsWithPCA(genre):

    print("Finding Latent Semantics using PCA\n")

    tfidf= GetMoviesTagsData(genre)

    # print(tfidf)

    matrix = tfidf.pivot_table(index='movieid', columns='tagid', values='tfidf', fill_value=0)

    pca_mat = decomposition.PCA(n_components=3)
    pca_mat.fit(matrix)

    a = pca_mat.transform(matrix)
    vt = pca_mat.components_
    PrintLatentSematics(vt.shape[0], vt, tfidf)

def GenerateLDAWithTags(genre):

    print("Finding topics using LDA\n")

    allgenremoviesdata = pandas.read_csv("mlmovies.csv")

    movieslist = allgenremoviesdata[allgenremoviesdata['genres'].str.contains(genre)]['movieid'].tolist()

    allmoviestagsdata = pandas.read_csv("mltags.csv")

    combineddata = allmoviestagsdata[allmoviestagsdata['movieid'].isin(movieslist)]

    combineddata = combineddata[['movieid','tagid']]

    movietagdetails = []
    movieslist = numpy.unique(combineddata[['movieid','tagid']]['movieid'].tolist())
    movie_count = len(movieslist)

    for i in (movieslist):
        taglist = combineddata[combineddata['movieid'] == i]['tagid'].tolist()
        new_taglist = []
        for tag in taglist:
            new_taglist.append(str(tag))
        movietagdetails.append(' '.join(new_taglist))

    tf_vectorizer = CountVectorizer(max_features=movie_count)

    tf = tf_vectorizer.fit_transform(movietagdetails)

    tf_feature_names = tf_vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_components=4, max_iter=5, random_state=0)
    lda.fit(tf)

    for topic_idx, topic in enumerate(lda.components_):
        print("Tags belonging to the topic %d:" % (topic_idx))
        print(" ".join([tf_feature_names[i]
            for i in topic.argsort()[:-3 - 1:-1]]))

# FindSemanticsWithSVD("Thriller")
# FindSemanticsWithPCA("Thriller")
# GenerateLDAWithTags("Thriller")
