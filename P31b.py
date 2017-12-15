import sys
import math
import datetime
import operator
import pandas as pd
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

from collections import Counter

mltags =  pd.io.parsers.read_csv('data/mltags.csv',
                                 names=['user_id', 'movie_id', 'tagid', 'timestamp'],
                                 engine='python',skiprows=[0])

mlrating = pd.io.parsers.read_csv('data/mlratings.csv',
                                  names=['movie_id', 'user_id','imdbid', 'rating', 'time',],
                                  engine='python',skiprows=[0])

mlmovies =  pd.io.parsers.read_csv('data/mlmovies.csv',
                                   names=['movie_id', 'title','year', 'genre'],
                                   engine='python',skiprows=[0])

def LDA_Movie_Recommendation(userId):
    movie_rated = mlrating[mlrating['user_id']==userId]['movie_id'].unique().tolist()
    movie_tagged = mltags[mltags['user_id']==userId]['movie_id'].unique().tolist()
    movie_watched = list(set(movie_rated+movie_tagged))
    user_tags = mltags[mltags['movie_id'].isin(movie_watched)]['tagid'].unique().tolist()
    movie_corpus = []
    for tag in user_tags:
        movie_list = mltags[mltags['tagid']==tag]['movie_id'].unique().tolist()
        tmp_movir_list = []
        for movie in movie_list:
            tmp_movir_list.append(str(movie))
        movie_corpus.append(' '.join(tmp_movir_list))
    vectorizer = CountVectorizer(max_df=0.99, min_df=0.00) #, max_features=len(user_tags))
    TFValue = vectorizer.fit_transform(movie_corpus)
    featureNames = vectorizer.get_feature_names()
    LDAValue = LatentDirichletAllocation(n_components=4, max_iter=5, learning_method='online', learning_offset=50., random_state=0)
    LDAValue.fit(TFValue)
    movie_semantics = numpy.array(LDAValue.components_)
    featureNames_int = list(map(int, featureNames))
    featureNames_array = numpy.array(featureNames_int)
    table = pd.DataFrame(columns={'movie_id','weight'})

    tmp = pd.DataFrame(columns={'movie_id','weight'})
    for movie in movie_watched:
        movie_weight = Counter({})
        if movie in featureNames_int:
            movie_row_index=featureNames_int.index(movie)
            movie_row = movie_semantics[:,movie_row_index]
            semantic_product = numpy.matmul(movie_row,movie_semantics)
            movie_weight_value = semantic_product[numpy.argsort(-semantic_product)]
            movie_id = featureNames_array[numpy.argsort(-semantic_product)]
            tmp['movie_id'] = movie_id
            tmp['weight'] = movie_weight_value
            table = table.append(tmp)

    table['weight_sum'] = table.groupby('movie_id')['weight'].transform('sum')
    table = table.drop_duplicates(subset=['movie_id'])
    rec_movie = table[~table['movie_id'].isin(movie_watched)]
    rec_movie = rec_movie[~rec_movie['movie_id'].isin(movie_watched)][:5]
    rec_movielist = rec_movie['movie_id'].tolist()

    print("Movies Watched by user {0} \n {1} ".format(userId,movie_watched))

    print("Movies Recommended using LDA\n {0} ".format(mlmovies[mlmovies['movie_id'].isin(rec_movielist)][['movie_id','title']]))

    return(movie_watched, rec_movie)
#
#
# LDA_Movie_Recommendation(9254)