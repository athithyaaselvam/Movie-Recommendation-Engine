import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


ml_movies = pd.io.parsers.read_csv('data/mlmovies.csv',
                                   names=['movie_id', 'title','year', 'genre'],
                                   engine='python',skiprows=[0])


ml_ratings = pd.io.parsers.read_csv('data/mlratings.csv',
                                    names=['movie_id', 'user_id','imdbid', 'rating', 'time',],
                                    engine='python',skiprows=[0])

ml_tags=pd.io.parsers.read_csv('data/mltags.csv',
                               names=['user_id', 'movie_id','tagid', 'time',],
                               engine='python',skiprows=[0])

marked_users = ml_tags['user_id'].unique().tolist()
# print(len(marked_users))
rated_user_list = ml_ratings['user_id'].unique().tolist()
ml_users = list(set(marked_users+rated_user_list))
# print(len(rated_user_list))
# print(len(ml_users))
ml_users = ml_users[:10000]

userid_previous = sorted(ml_users)
mapped_user_id = pd.DataFrame(userid_previous, columns={'user_id'})
userid_latest = list(range(1,len(userid_previous)+1))
mapped_user_id['userid_latest'] = userid_latest
# mapped_user_id

#execute only once
movieid_previous = ml_movies.movie_id.values
mapped_movie_id = pd.DataFrame(movieid_previous, columns={'movie_id'})
movieid_latest = list(range(1,len(movieid_previous)+1))
mapped_movie_id['movieid_latest'] = movieid_latest
# mapped_movie_id

ml_ratings=ml_ratings[ml_ratings['user_id'].isin(mapped_user_id['user_id'].tolist())]

ml_tags=ml_tags[ml_tags['user_id'].isin(mapped_user_id['user_id'].tolist())]


ml_ratings=ml_ratings.merge(mapped_user_id,on='user_id',how='left')
ml_ratings=ml_ratings.merge(mapped_movie_id,on='movie_id',how='left')


ml_movies=ml_movies.merge(mapped_movie_id,on='movie_id',how='left')
# ml_movies['movie_id'] = ml_movies['movieid_latest']
# ml_movies

ml_movies['movie_id'] = ml_movies['movieid_latest']
ml_movies.drop(ml_movies.columns[[4]],axis=1, inplace=True)

ml_tags = ml_tags.merge(mapped_user_id,on='user_id',how='left')
ml_tags = ml_tags.merge(mapped_movie_id,on='movie_id',how='left')

# print(ml_tags)
ml_tags['user_id'] = ml_tags['userid_latest']
ml_tags['movie_id'] = ml_tags['movieid_latest']
ml_tags.drop(ml_tags.columns[[4,5]],axis=1, inplace=True)


ml_ratings['user_id'] = ml_ratings['userid_latest']
ml_ratings['movie_id'] = ml_ratings['movieid_latest']
ml_ratings.drop(ml_ratings.columns[[5,6]],axis=1, inplace=True)


ratings_matrix = np.ndarray(
    shape=(np.max(ml_ratings.movie_id.values), np.max(ml_ratings.user_id.values)),
    dtype=np.uint8)

ratings_matrix[ml_ratings.movie_id.values-1, ml_ratings.user_id.values-1] = ml_ratings.rating.values

normalised_value_matrix = ratings_matrix - np.asarray([(np.mean(ratings_matrix, 1))]).T

movies_total_count = len(ml_movies)
A = normalised_value_matrix.T / np.sqrt(movies_total_count - 1)
U, S, V = np.linalg.svd(A)

def Movies_Details_Print_Func(header, movielist):

    detail_of_movies = pd.read_csv("data/mlmovies.csv")
    detail_of_movies = detail_of_movies[detail_of_movies['movieid'].isin(movielist)]
    list_of_movies = detail_of_movies.values.tolist()
    list_of_movies = sorted(list_of_movies, key=lambda x: x[0])
    print("{0} for the user are\n".format(header))
    for i in range(0, len(list_of_movies)):
        print("ID: {0}, Name: {1}".format(list_of_movies[i][0], list_of_movies[i][1]))

def cosine_similarity_topmost(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1
    movie_rows = data[index, :]
    mag_value = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity_value = np.dot(movie_rows, data.T) / (mag_value[index] * mag_value)
    index_values_sorted = np.argsort(-similarity_value)

    initial_sorted_values = similarity_value[np.argsort(-similarity_value)]
    index_values_sorted = np.argsort(-similarity_value)
    #     print(intial_sorted_values)
    #     print(index_values_sorted)
    similar_movie = pd.DataFrame(index_values_sorted,columns={'index'})

    #     print(type(initial_sorted_values))
    #     distinct_values = list(initial_sorted_values).unique()
    distinct_values = list(set(initial_sorted_values))

    #     print(type(distinct_values))
    second_largest = sorted(distinct_values)[-2]
    second_lowest = sorted(distinct_values)[1]
    #     print(distinct_values)
    #     print(second_largest)
    array_of_sorted_values = np.array(initial_sorted_values)
    array_of_sorted_values[array_of_sorted_values==np.inf]=second_largest
    array_of_sorted_values[array_of_sorted_values==-np.inf]=second_lowest
    final_sorted_values = list(array_of_sorted_values)
    #     print(final_sorted_values)
    similar_movie['weight'] = final_sorted_values
    return similar_movie

def SVD_Movie_Recommendation(user_Id):
    originaluserid = user_Id
    user_Id = mapped_user_id[mapped_user_id['user_id']==user_Id]['userid_latest'].tolist()
    movie_rated = ml_ratings[ml_ratings['user_id']==user_Id]['movie_id'].tolist()
    marked_movies = ml_tags[ml_tags['user_id']==user_Id]['movie_id'].tolist()
    watched_movie_list = list(set(movie_rated+marked_movies))
    watched_movie_detail = ml_movies[ml_movies['movie_id'].isin(watched_movie_list).tolist()]

    k = 50
    top_n = 10

    movie_list_final = pd.DataFrame(columns={'index','weight'})
    sliced_data = V.T[:, :k] # representative data
    for movie_id in watched_movie_list:
        similar_movies = cosine_similarity_topmost(sliced_data, movie_id, top_n)
        #         print(similar_movies)
        #         similar_movie_print_func(ml_movies, movie_id, indexes)
        movie_list_final = movie_list_final.append(similar_movies)

    movie_list_final['total_weight'] = movie_list_final.groupby('index')['weight'].transform('sum')
    movie_list_final = movie_list_final.drop_duplicates('index')[:10]
    movie_list_final = (movie_list_final.sort_values('total_weight',ascending=False)['index']+1).tolist()

    final_movie_detail = ml_movies[ml_movies['movie_id'].isin(movie_list_final)]
    final_movie_detail = final_movie_detail[~final_movie_detail['movie_id'].isin(watched_movie_list)]
    final_movie_detail = final_movie_detail.merge(mapped_movie_id,how='left',left_on='movie_id',right_on='movieid_latest')
    final_movie_detail['movie_id_x'] = final_movie_detail['movie_id_y']
    final_movie_detail = final_movie_detail.rename(columns={"movie_id_x": "movie_id"})
    final_movie_detail = final_movie_detail.drop(['movie_id_y', 'movieid_latest'], axis=1)[:5]

    print("Movies Watched by user {0} \n {1} ".format(originaluserid,watched_movie_detail[['movie_id', 'title']]))
    print("Movies Recommended using SVD\n {0} ".format(final_movie_detail[['movie_id', 'title']]))

    return (watched_movie_list, final_movie_detail)


normalised_value_matrix = ratings_matrix - np.matrix(np.mean(ratings_matrix, 1)).T
covariance_matrix = np.cov(normalised_value_matrix)
evals, evecs = np.linalg.eig(covariance_matrix)


def PCA_Movie_Recommendation(user_Id):
    originaluserid = user_Id
    user_Id = mapped_user_id[mapped_user_id['user_id']==user_Id]['userid_latest'].tolist()
    movie_rated = ml_ratings[ml_ratings['user_id']==user_Id]['movie_id'].tolist()
    marked_movies = ml_tags[ml_tags['user_id']==user_Id]['movie_id'].tolist()

    watched_movie_list = list(set(movie_rated+marked_movies))
    watched_movie_detail = ml_movies[ml_movies['movie_id'].isin(watched_movie_list).tolist()]

    k = 50
    top_n = 10

    movie_list_final = pd.DataFrame(columns={'index','weight'})
    sliced_data = evecs[:, :k] # representative data
    for movie_id in watched_movie_list:
        similar_movies = cosine_similarity_topmost(sliced_data, movie_id, top_n)
        movie_list_final = movie_list_final.append(similar_movies)

    movie_list_final['total_weight'] = movie_list_final.groupby('index')['weight'].transform('sum')
    movie_list_final = movie_list_final.drop_duplicates('index')[:10]
    movie_list_final = (movie_list_final.sort_values('total_weight',ascending=False)['index']+1).tolist()

    final_movie_detail = ml_movies[ml_movies['movie_id'].isin(movie_list_final)]
    final_movie_detail = final_movie_detail[~final_movie_detail['movie_id'].isin(watched_movie_list)]
    final_movie_detail = final_movie_detail.merge(mapped_movie_id,how='left',left_on='movie_id',right_on='movieid_latest')
    final_movie_detail['movie_id_x'] = final_movie_detail['movie_id_y']
    final_movie_detail = final_movie_detail.rename(columns={"movie_id_x": "movie_id"})
    final_movie_detail = final_movie_detail.drop(['movie_id_y', 'movieid_latest'], axis=1)[:5]

    print("Movies Watched by user {0} \n {1} ".format(originaluserid,watched_movie_detail[['movie_id', 'title']]))

    print("Movies Recommended using PCA\n {0} ".format(final_movie_detail[['movie_id', 'title']]))

    return (watched_movie_list, final_movie_detail)
#
# PCA_Movie_Recommendation(9254)
# SVD_Movie_Recommendation(9254)
