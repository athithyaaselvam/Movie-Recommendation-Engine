import sys
import pandas
import numpy
import math
import datetime

class VectorSpace:
	latentSemantics  = None
	movieIndexVsName = {}
	movieIdVsIndex = {}
	movieactorcsv = None
	actor_tag_matrix = None

	

	def getLSHInput(self):

		self.movieactorcsv =pandas.read_csv("mlmovies.csv")
		movietags = pandas.read_csv("mltags.csv")

		
		min_timestamp = pandas.to_datetime(min(movietags['timestamp']))
		max_timestamp = pandas.to_datetime(max(movietags['timestamp']))
		movietags['timestamp_weightage'] = movietags.apply(self.TimestampWeights, axis=1, args=(min_timestamp, max_timestamp))
		

		mergeddata = self.movieactorcsv[['movieid']].merge(movietags[['movieid','tagid','timestamp_weightage']], on='movieid')		

		mergeddata['total_weightage'] = mergeddata.apply(self.aggregate_tf_weightages, axis=1)

		mergeddata['tag_weightage'] = mergeddata.groupby(['movieid','tagid'])['total_weightage'].transform('sum')
		tfdata = mergeddata[['movieid', 'tagid', 'tag_weightage']].drop_duplicates(subset=['tagid', 'movieid'])

		tfdata['total_weightage_actor'] = tfdata.groupby(['movieid'])['tag_weightage'].transform('sum')

		tfdata['tf'] = tfdata.apply(self.CalculateTF, axis=1)

		taglist = tfdata['tagid'].tolist()
		movietags = pandas.read_csv("mltags.csv")
		movietags = movietags[movietags['tagid'].isin(taglist)]

		#print(alltagsdata)
		self.movieactorcsv = pandas.read_csv("mlmovies.csv")
		tagsRequired = movietags.merge(self.movieactorcsv, on='movieid')

		tagsRequired.drop_duplicates(subset=['tagid', 'movieid'], inplace=True)
		tagsRequired['actor_count'] = tagsRequired.groupby('tagid')['movieid'].transform('count')
		tagsRequired.drop_duplicates(subset=['tagid'], inplace=True)

		actordata = pandas.read_csv("mlmovies.csv")
		total_actors = actordata.shape[0]

		tagsRequired['idf'] = tagsRequired.apply(self.CalculateIDF, axis=1, total_actors=total_actors)

		tfidfdata = self.ConvertTFIDF(tfdata, tagsRequired[['tagid', 'idf']])

		self.actor_tag_matrix = tfidfdata.pivot_table(index='movieid', columns='tagid', values='tfidf', fill_value=0)

		u, s, v = numpy.linalg.svd(self.actor_tag_matrix, full_matrices=False)
		s[499:] = 0
		l = u.shape[1]
		resizeu = numpy.delete(u, numpy.s_[499:l-1], 1)
		rs= numpy.diag(s)
		m = rs.shape[0]
		resizes = numpy.delete(rs, numpy.s_[499:m-1], 0)
		n = resizes.shape[1]
		resizes1 = numpy.delete(resizes, numpy.s_[499:n-1], 1)		

		self.latentSemantics = numpy.dot(resizeu, resizes1)				
		return self.latentSemantics

	def getMovieID(self,movie):
		return self.movieactorcsv.loc[self.movieactorcsv['moviename']==movie]['movieid'].values[0]

	def getQueryPoint(self, movieId):
		
		for index, row in self.movieactorcsv.iterrows():
			self.movieIdVsIndex[row['movieid']] = index
			self.movieIndexVsName[index] = row['moviename']
		return self.latentSemantics[self.movieIdVsIndex[movieId]]

	def CalculateIDF(self,row, total_actors):
	    return math.log10(total_actors / row ['actor_count'])

	def actorrankweight(self,row):
	    return (1+ row['max_actor_rank'] - row['actor_movie_rank'])/(1+ row['max_actor_rank'] - row['min_actor_rank'])

	def TimestampWeights(self,row, min_timestamp, max_timestamp):
	    # return ((pandas.to_datetime(row['timestamp']) - min_timestamp).microseconds + 1) / ((max_timestamp - min_timestamp).microseconds + 1)
	    num_diff = (pandas.to_datetime(row['timestamp']) - min_timestamp)
	    den_diff = (max_timestamp - min_timestamp)
	    # print float((num_diff.days * 24 * 60 * 60) + (num_diff.seconds) + 1) / (den_diff.days * 24 * 60 * 60 + den_diff.seconds + 1)
	    return float((num_diff.days * 24 * 60 * 60) + (num_diff.seconds) + 1) / (den_diff.days * 24 * 60 * 60 + den_diff.seconds + 1)

	def aggregate_tf_weightages(self,row):
	    return numpy.round(row['timestamp_weightage'], decimals=4)

	def CalculateTF(self,row):
	    return row['tag_weightage'] / row ['total_weightage_actor']

	def CalculateTFIDF(self,row):
	    return row['tf']*row['idf']

	def ConvertTFIDF(self,tfdata, idfdata):
	    tfidfdata = tfdata.merge(idfdata, on='tagid')
	    tfidfdata['tfidf'] = tfidfdata.apply(self.CalculateTFIDF, axis=1)
	    return tfidfdata[['movieid','tagid','tfidf']]

