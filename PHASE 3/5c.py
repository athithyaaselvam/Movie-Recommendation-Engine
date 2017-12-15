import sys
import pandas
import numpy
import math
import warnings
from sklearn import tree
from sklearn import decomposition
from collections import Counter


import csv
from csv import reader
import numpy as np
import sys

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def projection_simplex(v, z=1):

    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


class MulticlassSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1, max_iter=50, tol=0.05,
                 random_state=None, verbose=0):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol,
        self.random_state = random_state
        self.verbose = verbose

    def _partial_gradient(self, X, y, i):
        # Partial gradient for the ith sample.
        g = np.dot(X[i], self.coef_.T) + 1
        g[y[i]] -= 1
        return g

    def _violation(self, g, y, i):
        # Optimality violation for the ith sample.
        smallest = np.inf
        for k in range(g.shape[0]):
            if k == y[i] and self.dual_coef_[k, i] >= self.C:
                continue
            elif k != y[i] and self.dual_coef_[k, i] >= 0:
                continue

            smallest = min(smallest, g[k])

        return g.max() - smallest

    def _solve_subproblem(self, g, y, norms, i):
        # Prepare inputs to the projection.
        Ci = np.zeros(g.shape[0])
        Ci[y[i]] = self.C
        beta_hat = norms[i] * (Ci - self.dual_coef_[:, i]) + g / norms[i]
        z = self.C * norms[i]

        # Compute projection onto the simplex.
        beta = projection_simplex(beta_hat, z)

        return Ci - self.dual_coef_[:, i] - beta / norms[i]

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Normalize labels.
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)

        # Initialize primal and dual coefficients.
        n_classes = len(self._label_encoder.classes_)
        self.dual_coef_ = np.zeros((n_classes, n_samples), dtype=np.float64)
        self.coef_ = np.zeros((n_classes, n_features))

        # Pre-compute norms.
        norms = np.sqrt(np.sum(X ** 2, axis=1))

        # Shuffle sample indices.
        rs = check_random_state(self.random_state)
        ind = np.arange(n_samples)
        rs.shuffle(ind)

        violation_init = None
        for it in range(self.max_iter):
            violation_sum = 0

            for ii in range(n_samples):
                i = ind[ii]

                # All-zero samples can be safely ignored.
                if norms[i] == 0:
                    continue

                g = self._partial_gradient(X, y, i)
                v = self._violation(g, y, i)
                violation_sum += v

                if v < 1e-12:
                    continue

                # Solve subproblem for the ith sample.
                delta = self._solve_subproblem(g, y, norms, i)

                # Update primal and dual coefficients.
                self.coef_ += (delta * X[i][:, np.newaxis]).T
                self.dual_coef_[:, i] += delta

            if it == 0:
                violation_init = violation_sum

            vratio = violation_sum / violation_init

            if vratio < self.tol:
                break

        return self

    def predict(self, X):
        decision = np.dot(X, self.coef_.T)
        pred = decision.argmax(axis=1)
        # print(pred)
        return pred

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
    return labellist

def SVMClassifier():

    print("Finding label using SVMClassifier\n")

    labeldetailslist = GetLabelDetails()

    modifiedlist =[]

    labelnames = sorted(numpy.unique([item[1] for item in labeldetailslist]))

    for i in range (0, len(labeldetailslist)):
        modifiedlist.append((labeldetailslist[i][0],labeldetailslist[i][1],labelnames.index(labeldetailslist[i][1])))

    modifiedlist = sorted(modifiedlist, key=lambda x: x[0])

    labelledmovieslist = sorted(numpy.unique([item[0] for item in modifiedlist]))

    labelidslist =[]

    for ele in modifiedlist:
        labelidslist.append(ele[2])

    my_df = pandas.DataFrame(labelidslist)

    my_df.to_csv('labelslistforSVM.csv', index=False, header=False)

    movietagsdata = GetMoviesTagsData()

    movieslist = sorted(numpy.unique(movietagsdata['movieid'].tolist()))

    movie_tag_matrix = movietagsdata.pivot_table(index='movieid', columns='tagid',values='tfidf', fill_value=0)
    data = []

    for movieid in labelledmovieslist:
        i =movieslist.index(movieid)
        data.append(movie_tag_matrix.values[i])

    my_df = pandas.DataFrame(data)
    my_df.to_csv('labledmoviedataforSVM.csv', index=False, header=False)

    my_df = pandas.DataFrame(movie_tag_matrix.values)
    my_df.to_csv('tobelabledmoviedataforSVM.csv', index=False, header=False)

    # data1 =[]
    # movieslist = [8171, 9096]
    # for movieid in [8171, 9096]:
    #     i =movieslist.index(movieid)
    #     data1.append(movie_tag_matrix.values[i])
    #
    # my_df = pandas.DataFrame(data1)
    # my_df.to_csv('test1.csv', index=False, header=False)

    # load and prepare data
    filename = 'labledmoviedataforSVM.csv'
    data = load_csv(filename)

    for i in range(len(data[0])):
        str_column_to_float(data, i)

    filename = 'labelslistforSVM.csv'
    labels = load_csv(filename)

    filename = 'tobelabledmoviedataforSVM.csv'

    testdata = load_csv(filename)

    for i in range(len(testdata[0])):
        str_column_to_float(testdata, i)

    label = []
    for i in labels:
        label.append(i[0])
    clf = MulticlassSVM(C=0.1, tol=0.01, max_iter=100, random_state=0, verbose=1)
    clf.fit(np.array(data), label)
    labelresults = clf.predict(np.array(testdata))

    labelled =[]

    for i in range(0,len(movieslist)):
        index = int(labelresults[i])
        labelled.append([movieslist[i],labelnames[index]])

    newlabelsdf = pandas.DataFrame(labelled,columns=['movieid','label'])

    print(newlabelsdf)

    newlabelsdf.to_csv('labelsSVM.csv', index=False)
