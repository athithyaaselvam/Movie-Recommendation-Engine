import sys
import pandas
import numpy
import math
import warnings
from sklearn import tree
from sklearn import decomposition
from collections import Counter

warnings.filterwarnings("ignore")

# CART on the Bank Note dataset
from random import seed
from random import randrange
from csv import reader

# Load a CSV file
def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 13
    return correct / float(len(actual)) * 100.0

def test_algo(dataset, test_data, algorithm,*args):
    predicted = algorithm(dataset, test_data, *args)
    # print (predicted)
    return predicted

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        # print (predicted)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini



# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)


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
    #
    # allmovietagsdata = allmovietagsdata[allmovietagsdata['movieid'].isin(movieslist)]

    allmovietagsdata['timestamp_weightage'] = allmovietagsdata.apply(ComputeTimestampWeights, axis=1, args=(min_timestamp, max_timestamp))

    allmovietagsdata['tag_weightage'] = allmovietagsdata.groupby(['movieid','tagid'])['timestamp_weightage'].transform('sum')

    allmovietagsdata = allmovietagsdata[['movieid','tagid','tag_weightage']].drop_duplicates(subset=['movieid','tagid'])

    allmovietagsdata['total_movie_weightage'] = allmovietagsdata.groupby(['movieid'])['tag_weightage'].transform('sum')

    # print(allmovietagsdata)

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

    # print(total_actors)
    # print(specificactortagsdata)

    tfidfdata = ProcessMovieTFandIDFtoTFIDF(allmovietagsdata, specifictagsdata[['tagid', 'idf']])

    # filteredtfidfdata = tfidfdata[tfidfdata['actorid'].isin(actorsingivenmovie)]

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
    return labellist

def DecisionTreeClassifier():

    print("Finding label using DecisionTreeClassifier\n")

    labeldetailslist = GetLabelDetails()

    modifiedlist =[]

    labelnames = sorted(numpy.unique([item[1] for item in labeldetailslist]))

    for i in range (0, len(labeldetailslist)):
        modifiedlist.append((labeldetailslist[i][0],labeldetailslist[i][1],labelnames.index(labeldetailslist[i][1])))

    modifiedlist = sorted(modifiedlist, key=lambda x: x[0])

    labelledmovieslist = sorted(numpy.unique([item[0] for item in modifiedlist]))

    movietagsdata = GetMoviesTagsData()

    templabellist = []
    maxtagid = max(movietagsdata['tagid'])

    for movieid in labelledmovieslist:
        j =labelledmovieslist.index(movieid)
        templabellist.append([movieid,maxtagid+1,modifiedlist[j][2]])

    labelsdf = pandas.DataFrame(templabellist,columns=['movieid','tagid','tfidf'])

    concatmovietagsdata = pandas.concat([movietagsdata,labelsdf])

    movieslist = sorted(numpy.unique(movietagsdata['movieid'].tolist()))

    movie_tag_matrix = concatmovietagsdata.pivot_table(index='movieid', columns='tagid',values='tfidf', fill_value=0)
    data = []

    for movieid in labelledmovieslist:
        i =movieslist.index(movieid)
        data.append(movie_tag_matrix.values[i])

    my_df = pandas.DataFrame(data)
    my_df.to_csv('labledmoviedataforDC.csv', index=False, header=False)


    my_df = pandas.DataFrame(movie_tag_matrix.values)
    my_df.to_csv('tobelabledmoviedataforDC.csv', index=False, header=False)

    seed(1)

    filename = 'labledmoviedataforDC.csv'
    dataset = load_csv(filename)

    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

    #evaluate algorithm
    n_folds = 5
    max_depth = 5
    min_size = 10
    evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)

    test_file = 'tobelabledmoviedataforDC.csv'
    test_data = load_csv(test_file)

    for i in range(len(test_data[0])):
        str_column_to_float(test_data, i)

    labelresults = test_algo(dataset, test_data, decision_tree, max_depth, min_size)

    labelled =[]

    for i in range(0,len(movieslist)):
        index = int(labelresults[i])
        labelled.append([movieslist[i],labelnames[index]])

    newlabelsdf = pandas.DataFrame(labelled,columns=['movieid','label'])

    print(newlabelsdf)

    newlabelsdf.to_csv('labelsDC.csv', index=False)
