import sys
import pandas
import numpy
import math
import warnings
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


warnings.filterwarnings("ignore")

def ComputeRankWeightage(row):
    return (1+ row['max_actor_rank'] - row['actor_movie_rank'])/(1+ row['max_actor_rank'] - row['min_actor_rank'])

def ComputeTF(row):
    return row['actor_weightage']/row['total_weightage']

def ProcessWeightsToTF(combineddata):
    combineddata['actor_weightage'] = combineddata.groupby(['movieid','actorid'])['actor_rank_weightage'].transform('sum')
    tfdata = combineddata[['movieid','actorid','actor_weightage']].drop_duplicates(subset=['movieid','actorid'])
    tfdata['total_weightage'] = tfdata.groupby('movieid')['actor_weightage'].transform('sum')
    tfdata['tf'] = tfdata.apply(ComputeTF, axis=1)
    return tfdata[['movieid','actorid', 'tf']].sort_values('tf', ascending=False)

def ComputeIDF(row, total_movies):
    return math.log10(total_movies/row['count_of_movies'])

def ComputeTFIDF(row):
    return row['tf']*row['idf']

def ProcessTFandIDFtoTFIDF(tfdata, idfdata):
    tfidfdata = tfdata.merge(idfdata, on='actorid')
    tfidfdata['tfidf'] = tfidfdata.apply(ComputeTFIDF, axis=1)
    return tfidfdata

def GetActorsTagsData(genre):

    allgenremoviesdata = pandas.read_csv("mlmovies.csv");
    allmoviesactorsdata = pandas.read_csv("movie-actor.csv");

    movieslist = allgenremoviesdata[allgenremoviesdata['genres'].str.contains(genre)]['movieid'].tolist()

    combineddata = allmoviesactorsdata[allmoviesactorsdata['movieid'].isin(movieslist)]


    combineddata['max_actor_rank'] = combineddata.groupby(['movieid'])['actor_movie_rank'].transform(max)
    combineddata['min_actor_rank'] = combineddata.groupby(['movieid'])['actor_movie_rank'].transform(min)

    combineddata['actor_rank_weightage'] = combineddata.apply(ComputeRankWeightage, axis=1)

    tfdata = ProcessWeightsToTF(combineddata)

    total_movies = allgenremoviesdata.shape[0]

    actorlist = tfdata['actorid'].tolist()
    specificactorsdata = allmoviesactorsdata[allmoviesactorsdata['actorid'].isin(actorlist)]

    specificactorsdata['count_of_movies'] = specificactorsdata.groupby(['actorid'])['movieid'].transform('count')
    specificactorsdata.drop_duplicates(subset=['actorid'], inplace=True)

    specificactorsdata['idf'] = specificactorsdata.apply(ComputeIDF,axis=1,total_movies=total_movies)

    tfidfdata = ProcessTFandIDFtoTFIDF(tfdata[['movieid', 'actorid', 'tf']], specificactorsdata[['actorid', 'idf']])
    return tfidfdata

def GetActorDetails(tfidf):

    actordetails = pandas.read_csv("imdb-actor-info.csv")
    actordetails = actordetails[actordetails['id'].isin(tfidf['actorid'].tolist())]
    actornamelist = actordetails.values.tolist()
    actornamelist = sorted(actornamelist, key=lambda x: x[0])
    return actornamelist

def PrintLatentSematics(latentsematics,vt,tfidf):

    actornamelist = GetActorDetails(tfidf)

    for i in range(0, min(latentsematics, 4)):
        latentsemanticrow = vt[i]
        mean = numpy.mean(latentsemanticrow)
        print("\nLatent semantic {0}".format(i+1))
        for j in range(0, len(actornamelist)):
            if(latentsemanticrow[j] >= mean):
                print("{0} ({1})".format(actornamelist[j][0], actornamelist[j][1]))

def FindSemanticsWithSVD(genre):

    print("Finding Latent Semantics using SVD\n")

    tfidf= GetActorsTagsData(genre)

    # print(tfidf)

    matrix = tfidf.pivot_table(index='movieid', columns='actorid', values='tfidf', fill_value=0)

    # print(matrix)

    u, s, vt = numpy.linalg.svd(matrix, full_matrices=False)

    # print(u)
    # print(s)
    # print(vt)

    PrintLatentSematics(s.size, vt, tfidf)

def FindSemanticsWithPCA(genre):

    print("Finding topics using PCA\n")

    tfidf= GetActorsTagsData(genre)

    # print(tfidf)

    matrix = tfidf.pivot_table(index='movieid', columns='actorid', values='tfidf', fill_value=0)

    # print(matrix)
    pca_mat = decomposition.PCA(n_components=3)
    pca_mat.fit(matrix)

    a = pca_mat.transform(matrix)
    vt = pca_mat.components_

    PrintLatentSematics(vt.shape[0], vt, tfidf)

def GenerateLDAWithActors(genre):

    print("Finding topics using LDA\n")

    allgenremoviesdata = pandas.read_csv("mlmovies.csv")

    allmoviesactorsdata = pandas.read_csv("movie-actor.csv")

    moviesingivengenre = allgenremoviesdata[allgenremoviesdata['genres'].str.contains(genre)]['movieid'].tolist()

    movieactorsdata = allmoviesactorsdata[allmoviesactorsdata['movieid'].isin(moviesingivengenre)]

    movieslist = numpy.unique(movieactorsdata[['movieid','actorid']]['movieid'].tolist())

    movie_count = len(movieslist)

    movieactorsdetails = []

    for i in (movieslist):
        actorlist = movieactorsdata[movieactorsdata['movieid'] == i]['actorid'].tolist()
        concat_actor_list = []
        for actor in actorlist:
            concat_actor_list.append(str(actor))
        movieactorsdetails.append(' '.join(concat_actor_list))

    tf_vectorizer = CountVectorizer(max_features=movie_count)

    tf = tf_vectorizer.fit_transform(movieactorsdetails)

    tf_feature_names = tf_vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_components=5, max_iter=5, random_state=0)
    lda.fit(tf)

    for topic_idx, topic in enumerate(lda.components_):
        print("Actors belonging to the topic %d:" % (topic_idx+1))
        print(" ".join([tf_feature_names[i]
                        for i in topic.argsort()[:-3 - 1:-1]]))

# FindSemanticsWithSVD("Thriller")
# FindSemanticsWithPCA("Thriller")
# GenerateLDAWithActors("Thriller")