import sys
import mysql.connector
import math
import datetime
import operator
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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

actor_count = 0
actors = []

for movie in movies:
	fetch_actors = ("SELECT actorid FROM movie_actor WHERE movieid = %s" % movie)
	cursor.execute(fetch_actors)
	for item in cursor:
		actorid = item[0]
		if actorid not in actors:
			actors.append(actorid)
			actor_count = actor_count + 1
		movies[movie].append(str(item[0]))

for movie in movies: 
	print movie, ": ", movies[movie]

documents = []
for movie in movies:
	if len(movies[movie]) > 0:
		documents.append(' '.join(movies[movie]))

print len(documents), type(documents)
print documents

feature_count_model = CountVectorizer(max_df=0.95, min_df=1, max_features=actor_count)
feature_counts = feature_count_model.fit_transform(documents)
print type(feature_counts)

actor_names = feature_count_model.get_feature_names()
print "Actor IDs"
print type(actor_names)
print actor_names

lda = LatentDirichletAllocation(n_components=4, max_iter=5, learning_offset=50., random_state=0)
lda.fit(feature_counts)

# num_features = len(lda.components_[0])
num_features = 6
for index, component in enumerate(lda.components_):
	print  "Topic %d:" % (index), " ".join([actor_names[i] for i in component.argsort()[:-num_features - 1:-1]])

cursor1.close()
cursor.close()
connection.close()
