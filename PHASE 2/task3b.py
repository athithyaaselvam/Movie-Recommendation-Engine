import sys
import pandas
import numpy as np
import numpy
import math
import datetime
import networkx as nx
from sklearn.preprocessing import normalize

def GenerateCoActorCoActorMatrix(allactormoviesdata):
   allactormoviesdata['value'] = 1
   actor_movie_matrix = allactormoviesdata.pivot_table(index='actorid', columns='movieid', values='value', fill_value=0)
   # print(actor_movie_matrix)
   transpose_actor_movie_matrix = actor_movie_matrix.transpose()
   coactorcoactorimilarity = numpy.matmul(actor_movie_matrix, transpose_actor_movie_matrix);
   # print(coactorcoactorimilarity)
   return coactorcoactorimilarity

def GetActorList(actordata):
    actordetails = pandas.read_csv("imdb-actor-info.csv")
    # actordetails = actordetails[actordetails['id'].isin(actordata['actorid'].tolist())]
    # actornamelist = actordetails.values.tolist()
    # actornamelist = sorted(actornamelist, key=lambda x: x[0])
    actorlist = sorted(numpy.unique(actordata['actorid'].tolist()))
    # print(actorlist)
    return actorlist

# Get seed nodes from file
def get_seeds(args):
    seeds = []
    for i in range(1, len(args)):
        seeds.append(int(args[i]))
    return seeds

# Map actor id with index
def map_actor(index): 
    return actorlist[index]

# get initial porbability vector
def get_initial_probability_vector(seeds):
    p_init = [0] * num_nodes
    num_seeds = len(seeds)
    for seed in seeds:
        p_init[actorlist.index(seed)] = 1 / float(num_seeds)
    return np.array(p_init)

# Calculate next probability
def get_next_probability( p_next, p_init):
    change = np.squeeze(np.asarray(np.dot(adj_matrix, p_next)))
    no_restart_prob = change * (1 - SEED_PROB)
    restart_prob = p_init * SEED_PROB
    return np.add(no_restart_prob, restart_prob)

def PrintActorNames(actorlist):
    actordetails = pandas.read_csv("imdb-actor-info.csv")
    actordetails = actordetails[actordetails['id'].isin(actorlist)]
    print(actordetails['name'].tolist())

NODE_PROB = 0.15
SEED_PROB = 0.85
CONV_THRESHOLD = 0.001
diff = 1

# Get seed values in a list
seeds = get_seeds(sys.argv)
print "Seeds: "
# print len(seeds), seeds

# Get similzrity matrix
allactormoviesdata =pandas.read_csv("movie-actor.csv")
# allactormoviesdata = allactormoviesdata[allactormoviesdata['actorid'].isin([1856129, 1335137, 1808442, 1170039, 3813562, 973296, 1072584, 1584763, 3368894, 2757658, 3649447, 3134858, 3425734, 2706646, 462304, 99457])]
coactorcoactorimilarity = GenerateCoActorCoActorMatrix(allactormoviesdata)
actorlist = GetActorList(allactormoviesdata)
# print type(coactorcoactorimilarity), len(coactorcoactorimilarity), len(coactorcoactorimilarity[0])
CCSimilarity = coactorcoactorimilarity.tolist()
# print CCSimilarity, type(CCSimilarity)
# print CCSimilarity[0], type(CCSimilarity[0])

num_nodes = len(coactorcoactorimilarity)
# print "Nodes: ", num_nodes
num_seeds = len(seeds)
# print "Seeds: ", num_seeds

# Construct Graph
graph = nx.Graph()
for i in range(num_nodes):
    for j in range(num_nodes):
        # if coactorcoactorimilarity[i][j] != 0:
        graph.add_edge(map_actor(i), map_actor(j), weight=coactorcoactorimilarity[i][j])

# normalize adjacenct matrix
adj_matrix = nx.to_numpy_matrix(graph)
# print type(adj_matrix)
# print adj_matrix
normalized_adj_matrix = normalize(adj_matrix, norm='l1', axis=0)

# get initial probability vector
p_init = get_initial_probability_vector(seeds)
p_next = np.copy(p_init)

# Start iterations
while(diff > CONV_THRESHOLD):
    p = get_next_probability(p_next, p_init)
    diff = np.linalg.norm(np.subtract(p, p_next), 1)
    p_next = p

probability_list = zip(graph.nodes(), p_next.tolist())
# sort by probability (from largest to smallest), and generate a
# sorted list of Entrez IDs
# print "Probability Vector: "
actor_prob = sorted(probability_list, key=lambda x: x[1], reverse=True)
topactorlist = []
probability_list = sorted(probability_list, key=lambda x: x[1], reverse=True)

for i in range(0,10):
    topactorlist.append(probability_list[i][0])

# print topactorlist
print "Related Actors to the actor based on actor-actor-similarity {0}\n".format(seeds)
PrintActorNames(topactorlist)