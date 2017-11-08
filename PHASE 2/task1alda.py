import sys
import mysql.connector
import math
import datetime
import operator
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

connection = mysql.connector.connect(user='root', password='root', host='127.0.0.1', database='mwdb')
cursor = connection.cursor(buffered=True)
cursor1 = connection.cursor(buffered=True)

genre = sys.argv[1]
genre_str = "'%" + genre + "%'"
fetch_movies = ("SELECT DISTINCT(movieid) FROM mlmovies WHERE genres LIKE %s" % genre_str)
cursor.execute(fetch_movies)
movies = {}
for item in cursor:
	movies[item[0]] = []

tag_count = 0
tags = []

for movie in movies:
	fetch_tags = ("SELECT tagid FROM mltags WHERE movieid = %s" % movie)
	cursor.execute(fetch_tags)
	for item in cursor:
		tagid = item[0]
		fetch_tag_name = ("SELECT tag FROM gnome_tags WHERE tagid = %s" % tagid)
		if tagid not in tags:
			tags.append(tagid)
			tag_count = tag_count + 1
		movies[movie].append(str(tagid))

# for movie in movies: 
	# print movie, ": ", movies[movie]

documents = []
for movie in movies:
	if len(movies[movie]) > 0:
		documents.append(' '.join(movies[movie]))

feature_count_model = CountVectorizer(max_df=0.95, min_df=1, max_features=tag_count, stop_words='english')
feature_count = feature_count_model.fit_transform(documents)
# print type(feature_count)

tag_names = feature_count_model.get_feature_names()
# print "Feature names: "
# print type(tag_names)
# print type(tag_names[0])
# print tag_names

for feature in tag_names:
	fetch_feature_name = ("SELECT tagid FROM mltags WHERE tagid = %s" % feature)
	cursor.execute(fetch_feature_name)
	for item in cursor:
		feature = str(item[0])

lda = LatentDirichletAllocation(n_components=4, learning_offset=50., random_state=0)
lda.fit(feature_count)

num_features = len(lda.components_[0])
for index, component in enumerate(lda.components_):
	print  "Topic %d:" % (index), " ".join([tag_names[i] for i in component.argsort()[:-num_features - 1:-1]])

cursor1.close()
cursor.close()
connection.close()

