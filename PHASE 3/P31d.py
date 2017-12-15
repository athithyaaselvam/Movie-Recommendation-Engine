import sys
import pandas
import numpy
import math
import datetime



movie_actor_input_csv =pandas.read_csv("data/mlmovies.csv")

ml_tags = pandas.read_csv("data/mltags.csv")

ml_ratings = pandas.read_csv("data/mlratings.csv")


def CalculateIDF(row_value, total_actors):
    return math.log10(total_actors / row_value ['actor_count'])

def Actor_Rank_Weights(row_value):
    return (1+ row_value['max_actor_rank'] - row_value['actor_movie_rank'])/(1+ row_value['max_actor_rank'] - row_value['min_actor_rank'])

def TimeStamp_Weights(row_value, minimum_timestamp, maximum_timestamp):
    return ((pandas.to_datetime(row_value['timestamp'])-minimum_timestamp).days + 1)/((maximum_timestamp-minimum_timestamp).days+1)

def TF_Weights_Aggregate(row_value):
    return numpy.round(row_value['timestamp_weightage'], decimals=4)

def Calculate_TF(row_value):
    return row_value['tag_weightage'] / row_value ['total_weightage_actor']


def Calculate_TF_IDF(row_value):
    return row_value['tf']*row_value['idf']

def Process_TF_IDF_to_TFIDF(tf_data, idf_data):
    tfidf_data = tf_data.merge(idf_data, on='tagid')
    tfidf_data['tfidf'] = tfidf_data.apply(Calculate_TF_IDF, axis=1)
    return tfidf_data[['movieid','tagid','tfidf']]

minimum_timestamp = pandas.to_datetime(min(ml_tags['timestamp']))
maximum_timestamp = pandas.to_datetime(max(ml_tags['timestamp']))

ml_tags['timestamp_weightage'] = ml_tags.apply(TimeStamp_Weights, axis=1, args=(minimum_timestamp, maximum_timestamp))


merged_final_data = movie_actor_input_csv[['movieid']].merge(ml_tags[['movieid','tagid','timestamp_weightage']], on='movieid')

merged_final_data['total_weightage'] = merged_final_data.apply(TF_Weights_Aggregate, axis=1)


merged_final_data['tag_weightage'] = merged_final_data.groupby(['movieid','tagid'])['total_weightage'].transform('sum')
tf_data = merged_final_data[['movieid', 'tagid', 'tag_weightage']].drop_duplicates(subset=['tagid', 'movieid'])

tf_data['total_weightage_actor'] = tf_data.groupby(['movieid'])['tag_weightage'].transform('sum')

tf_data['tf'] = tf_data.apply(Calculate_TF, axis=1)


tags_list = tf_data['tagid'].tolist()
ml_tags = pandas.read_csv("data/mltags.csv")
ml_tags = ml_tags[ml_tags['tagid'].isin(tags_list)]

movie_actor_input_csv = pandas.read_csv("data/mlmovies.csv")
mandatory_tags_data = ml_tags.merge(movie_actor_input_csv, on='movieid')

mandatory_tags_data.drop_duplicates(subset=['tagid', 'movieid'], inplace=True)
mandatory_tags_data['actor_count'] = mandatory_tags_data.groupby('tagid')['movieid'].transform('count')
mandatory_tags_data.drop_duplicates(subset=['tagid'], inplace=True)

actordata = pandas.read_csv("data/mlmovies.csv")
total_actors = actordata.shape[0]

mandatory_tags_data['idf'] = mandatory_tags_data.apply(CalculateIDF, axis=1, total_actors=total_actors)

tfidf_data = Process_TF_IDF_to_TFIDF(tf_data, mandatory_tags_data[['tagid', 'idf']])

movie_tag_matrix = tfidf_data.pivot_table(index='movieid', columns='tagid', values='tfidf', fill_value=0)

covariance_matrix = numpy.cov(movie_tag_matrix)

vq = numpy.zeros(len(movie_actor_input_csv)) # np.max(mlrating.movie_id.values))

distinct_movie = movie_actor_input_csv['movieid'].unique().tolist()
c=0.85

def PAGERANK_Movie_Recommendation(userId):

    tagged_movie = ml_tags[ml_tags['userid']==userId]['movieid']
    rated_movie = ml_ratings[ml_ratings['userid']==userId]['movieid']
    watched_movies = tagged_movie.append(rated_movie).unique()
    watched_movie_details = movie_actor_input_csv[movie_actor_input_csv['movieid'].isin(watched_movies)]
    #     print(watched_movies)
    for movie in watched_movies:
        movie_index = distinct_movie.index(movie)
        #         print(movie_index)
        vq[movie_index]=1/len(watched_movies)

    uq = vq
    #     table = pandas.DataFrame
    for i in range(0,20):
        uq = (1-c)*(numpy.matmul(covariance_matrix,uq)) + c * vq
    #         print(uq)

    final_sorted_values = uq[numpy.argsort(-uq)]
    sort_all_indexes = numpy.argsort(-uq) + 1
    #     print(final_sorted_values)
    #     print(sort_all_indexes)
    recommended_movie = []
    for index in sort_all_indexes[:10]:
        #         print(index)
        recommended_movie.append(index)
    recommended_movie_detail = movie_actor_input_csv[movie_actor_input_csv['movieid'].isin(recommended_movie)]

    recommended_movie_detail = recommended_movie_detail[~recommended_movie_detail['movieid'].isin(watched_movies)]

    print("Movies Watched by user {0} \n {1} ".format(userId,watched_movies))

    print("Movies Recommended using PAGERANK \n {0} ".format(recommended_movie_detail[['movieid','moviename']]))

    return(watched_movies, recommended_movie_detail)
#
#
# PAGERANK_Movie_Recommendation(9254)