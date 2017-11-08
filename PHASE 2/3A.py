import sys
import pandas
import numpy
import math
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize

def ComputeRankWeightage(row):
    return (1+ row['max_actor_rank'] - row['actor_movie_rank'])/(1+ row['max_actor_rank'] - row['min_actor_rank'])

def ComputeTimestampWeights(row, min_timestamp, max_timestamp):
    return ((pandas.to_datetime(row['timestamp'])-min_timestamp).days + 1)/((max_timestamp-min_timestamp).days+1)

def AddWeightages(row):
    return numpy.round(row['actor_rank_weightage'] + row['timestamp_weightage'], decimals=4)

def ComputeTF(row):
    return row['tag_weightage'] / row['total_actor_weightage']

def ProcessWeightsToTF(combineddata):

    combineddata['all_weightages'] = combineddata.apply(AddWeightages, axis=1)
    combineddata['tag_weightage'] = combineddata.groupby(['actorid','tagid'])['all_weightages'].transform('sum')
    combineddata = combineddata[['actorid', 'tagid', 'tag_weightage']].drop_duplicates(subset=['actorid', 'tagid'])
    combineddata['total_actor_weightage'] = combineddata.groupby(['actorid'])['tag_weightage'].transform('sum')
    combineddata['tf'] = combineddata.apply(ComputeTF, axis=1)
    return combineddata

def ComputeIDF(row, total_actors):
    return math.log10(total_actors / row['count_of_actors'])

def ComputeTFIDF(row):
    return row['tf']*row['idf']

def ProcessTFandIDFtoTFIDF(tfdata, idfdata):
    tfidfdata = tfdata.merge(idfdata, on='tagid')
    tfidfdata['tfidf'] = tfidfdata.apply(ComputeTFIDF, axis=1)
    return tfidfdata[['actorid','tagid','tfidf']]

def GenerateAllActorsTFIDF():

    allactormoviesdata =pandas.read_csv("movie-actor.csv")

    allmoviestagsdata = pandas.read_csv("mltags.csv")

    allactormoviesdata['max_actor_rank'] = allactormoviesdata.groupby(['movieid'])['actor_movie_rank'].transform(max)
    allactormoviesdata['min_actor_rank'] = allactormoviesdata.groupby(['movieid'])['actor_movie_rank'].transform(min)

    allactormoviesdata['actor_rank_weightage'] = allactormoviesdata.apply(ComputeRankWeightage, axis=1)

    min_timestamp = pandas.to_datetime(min(allmoviestagsdata['timestamp']))
    max_timestamp = pandas.to_datetime(max(allmoviestagsdata['timestamp']))

    allmoviestagsdata['timestamp_weightage'] = allmoviestagsdata.apply(ComputeTimestampWeights, axis=1, args=(min_timestamp, max_timestamp))

    combineddata = allactormoviesdata[['actorid','movieid','actor_rank_weightage']].merge(allmoviestagsdata[['movieid','tagid','timestamp_weightage']], on='movieid')

    tfdata = ProcessWeightsToTF(combineddata)

    taglist = tfdata['tagid'].tolist()
    alltagsdata = pandas.read_csv("mltags.csv")
    specifictagsdata = alltagsdata[alltagsdata['tagid'].isin(taglist)]

    allactormoviesdata = pandas.read_csv("movie-actor.csv")
    specificactortagsdata = specifictagsdata.merge(allactormoviesdata, on='movieid')

    specificactortagsdata.drop_duplicates(subset=['tagid', 'actorid'], inplace=True)
    specificactortagsdata['count_of_actors'] = specificactortagsdata.groupby('tagid')['actorid'].transform('count')
    specificactortagsdata.drop_duplicates(subset=['tagid'], inplace=True)

    actordata = pandas.read_csv("imdb-actor-info.csv")
    total_actors = actordata.shape[0]

    specificactortagsdata['idf'] = specificactortagsdata.apply(ComputeIDF, axis=1, total_actors=total_actors)

    tfidfdata = ProcessTFandIDFtoTFIDF(tfdata, specificactortagsdata[['tagid', 'idf']])

    return tfidfdata

def GenerateActorsActorsSimilarity(actor_tag_matrix):

    transpose_actor_tag_matrix = actor_tag_matrix.transpose()
    actorsactorsimilarity = numpy.matmul(actor_tag_matrix, transpose_actor_tag_matrix);
    return actorsactorsimilarity

def GetActorList(actordata):

    actorlist = sorted(numpy.unique(actordata['actorid'].tolist()))
    return actorlist

def PrintActorNames(actorlist):

    actordetails = pandas.read_csv("imdb-actor-info.csv")
    actordetails = actordetails[actordetails['id'].isin(actorlist)]
    print(actordetails['name'].tolist())

def map_actor(index,actornamelist):
    return actornamelist[index]
# get initial porbability vector

def get_initial_probability_vector(seeds,num_nodes,actornamelist):
    p_init = [0] * num_nodes
    num_seeds = len(seeds)
    for seed in seeds:
        p_init[actornamelist.index(seed)] = 1 / float(num_seeds)
    return np.array(p_init)

# Calculate next probability
def get_next_probability( p_next, p_init, adj_matrix, SEED_PROB):
    change = np.squeeze(np.asarray(np.dot(adj_matrix, p_next)))
    no_restart = change * (1 - SEED_PROB)
    restart = p_init * SEED_PROB
    return np.add(no_restart, restart)

def get_seeds(args):
    seeds = []
    for i in range(1, len(args)):
        seeds.append(args[i])
    return seeds

def PageRank(seeds):
    NODE_PROB = 0.15
    SEED_PROB = 0.85
    CONV_THRESHOLD = 0.00001
    diff = 1
    # Get seed values in a list
    # seeds = get_seeds()
    # Get similzrity matrix

    tfidfdata = GenerateAllActorsTFIDF()

    actor_tag_matrix = tfidfdata.pivot_table(index='actorid', columns='tagid', values='tfidf', fill_value=0)

    similarity_matrix = GenerateActorsActorsSimilarity(actor_tag_matrix)

    actornamelist = GetActorList(tfidfdata)

    num_nodes = len(similarity_matrix)
    num_seeds = len(seeds)

    # Construct Graph
    graph = nx.Graph()
    for i in range(num_nodes):
        for j in range(num_nodes):
            graph.add_edge(map_actor(i,actornamelist), map_actor(j,actornamelist), weight=similarity_matrix[i][j])

    adj_matrix = nx.to_numpy_matrix(graph)
    normalized_adj_matrix = normalize(adj_matrix, norm='l1', axis=0)
    p_init = get_initial_probability_vector(seeds,num_nodes,actornamelist)
    p_next = np.copy(p_init)
    # Start iterations
    while(diff > CONV_THRESHOLD):
        p = get_next_probability(p_next, p_init, adj_matrix, SEED_PROB)
        diff = np.linalg.norm(np.subtract(p, p_next), 1)
        p_next = p

    probability_list = zip(graph.nodes(), p_next.tolist())
    topactorlist = []

    probability_list = sorted(probability_list, key=lambda x: x[1], reverse=True)

    for i in range(0,10):
        if(probability_list[i][1] > 0):
            topactorlist.append(probability_list[i][0])

    print("Related Actors to the actor based on actor-actor-similarity {0}\n".format(seeds))

    PrintActorNames(topactorlist)

seeds = get_seeds(sys.argv)
PageRank(seeds)