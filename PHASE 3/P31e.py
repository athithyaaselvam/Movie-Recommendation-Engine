import P31a

import P31b

import P31c

import P31d


import pandas as pd
import numpy as np
import math
epsilon = 0.000001

movie_actor_pd = pd.read_csv('data/movie-actor.csv')
movies_pd = pd.read_csv('data/mlmovies.csv')
tags_pd = pd.read_csv('data/mltags.csv')

def to_year(date_tag):
    year = date_tag.split("-")
    return int(year[0])

def GetMoviesDetails(movielist):

    moviedetails = pd.read_csv("data/mlmovies.csv")
    moviedetails = moviedetails[moviedetails['movieid'].isin(movielist)]
    movienamelist = moviedetails.values.tolist()
    return movienamelist

def combined_recom(final_recommendation):

    movie_list = final_recommendation["movieid"].unique().tolist()
    movie_details = movie_actor_pd[movie_actor_pd['movieid'].isin(movie_list)]
    actor_list =  movie_details["actorid"].unique().tolist()
    count = 0
    actor_count = {}
    semantic_actor_list = []

    for actor_id in actor_list:
        count = 0
        for movie in movie_list:
            movie_list_1 = movie_details[movie_details["movieid"] == movie]
            actor_id_list = movie_list_1["actorid"].tolist()
            if actor_id in actor_id_list:
                count = count + 1
        actor_count[actor_id] = count
    actor_count = sorted(actor_count.items(), key=lambda x: x[1] , reverse = True)

    for item in actor_count:
        semantic_actor_list.append(item[0])

    actor_movie = movie_details[movie_details['actorid'].isin(semantic_actor_list)]
    actor_movie_tag_sd = actor_movie.merge(tags_pd[['movieid','tagid','timestamp']])
    actor_movie_tag_sd['year'] = actor_movie_tag_sd['timestamp'].apply(to_year)
    actor_movie_tag_sd['SD'] = actor_movie_tag_sd.groupby('movieid')['year'].transform(np.std).fillna(0)
    actor_movie_tag_sd = actor_movie_tag_sd.drop_duplicates('movieid') #.sort_values('SD',ascending=False)

    mean_sd = actor_movie_tag_sd['SD'].mean()
    final_movie_list = []

    mean_sd_movies = actor_movie_tag_sd #[actor_movie_tag_sd["SD"]]
    #     print(actor_movie_tag_sd)
    for actor in semantic_actor_list:
        tmp_movie_list = movie_details[movie_details["actorid"] == actor]
        tmp_movie_sd = mean_sd_movies[mean_sd_movies['movieid'].isin(tmp_movie_list['movieid'])].sort_values('SD', ascending=False)
        for movie in tmp_movie_sd['movieid']:
            if movie not in final_movie_list:
                final_movie_list.append(movie)

    final_movie_details = (movies_pd[movies_pd['movieid'].isin(final_movie_list)])
    print("Combined recommendation")
    return(final_movie_details[:5])

unique_movies =   pd.DataFrame(movies_pd[['movieid','genres']])
unique_movies['user_feedback'] = [0]*len(unique_movies)
N = len(movies_pd)

initial_weight = [0]*len(tags_pd)
tags_details = tags_pd
tags_details['weight'] = initial_weight

def feedbackrel_1(initial_recom,tags_data,watched_movies):
    #
    # user_feedback = (list(input("enter user feedback: ")))
    user_feedback = []
    for movieid in (initial_recom['movie_id'].tolist()):
        yn = input("Y/N to mark a movie {0} relevant or not".format(movieid))
        user_feedback.append(yn)
    initial_recom['user_feedback'] = user_feedback
    pf_movie = initial_recom[initial_recom['user_feedback']=='Y']['movie_id'].tolist()

    pf_movie_tag = list(set(tags_pd[tags_data['movieid'].isin(pf_movie)]))
    R=len(pf_movie)

    unique_tags = tags_data['tagid'].unique()
    tag_weight = pd.DataFrame(sorted(unique_tags),columns={'tagid'})
    weight = []
    for tag in unique_tags:
        ri = tags_data[tags_data['movieid'].isin(pf_movie)]
        ri = len(ri[ri['tagid']==tag]['movieid'].unique())
        ni = len(tags_data[tags_data['tagid']==tag])
        if (ri == 0):
            weight.append(0)
        else:
            tmp = math.log10((ri+epsilon/(R-ri+epsilon)) / ((ni-ri+epsilon)/(N-R-ni+ri+epsilon)))
            weight.append(tmp)
    # print(type(weight))
    tag_weight['new_weight'] = weight

    tags_data = tags_data.merge(tag_weight,on='tagid',how='left')
    tags_data['weight'] = tags_data['weight'] + tags_data['new_weight']

    tags_data['weight'] = (tags_data.groupby(['movieid'])['weight'].transform('sum'))

    tags_data = tags_data.sort_values('weight',ascending=False)
    tags_data.drop_duplicates(subset=['movieid'],inplace=True)
    movie_recomendatons =tags_data[~tags_data['movieid'].isin(watched_movies)][:5]
    # print(movies_pd[movies_pd['movieid'].isin(movie_recomendatons['movieid'].tolist())])
    return(movie_recomendatons)

def Movie_Recommendation(useridstr):

    userId = int(useridstr)

    method = input("Relevance feedback recommendation using SVA/PCA/LDA/PAGERANK/ALL")

    if (method == 'SVD'):
        movie_watched,movie_recomendatons = P31a.SVD_Movie_Recommendation(userId)
    elif (method == 'PCA'):
        movie_watched,movie_recomendatons = P31a.PCA_Movie_Recommendation(userId)
    elif (method == 'LDA'):
        movie_watched,movie_recomendatons = P31b.LDA_Movie_Recommendation(userId)
    elif  (method == 'TENSORD'):
        movie_watched, movie_recomendatons = P31d.PAGERANK_Movie_Recommendation(userId)
    elif  (method == 'PAGERANK'):
        movie_watched, movie_recomendatons = P31d.PAGERANK_Movie_Recommendation(userId)
    elif  (method == 'ALL'):
        movie_watched,svd_recom = P31a.SVD_Movie_Recommendation(userId)
        movie_watched,pca_recom = P31a.PCA_Movie_Recommendation(userId)
        movie_watched,lda_recom = P31b.LDA_Movie_Recommendation(userId)
        movie_watched,pagerank_recom = P31b.PAGERANK_Movie_Recommendation(userId)

        print(type(svd_recom),type(pca_recom))
        final_recommendation = svd_recom.append(pca_recom)
        final_recommendation = final_recommendation.append(lda_recom)
        final_recommendation = final_recommendation.append(pagerank_recom)
        movie_recomendatons = combined_recom(final_recommendation)

    movie_recomendatons = feedbackrel_1(movie_recomendatons,tags_details,movie_watched)

    print("Recommended movie with relevance are {0}".format(movie_recomendatons['movieid'].tolist()))

Movie_Recommendation('9254')
