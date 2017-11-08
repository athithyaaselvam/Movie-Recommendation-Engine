import sys
import pandas
import numpy
import math
import warnings
from tabulate import tabulate


class Utils:

#Reads the data from CSV to dataframe
    def ReadWholeCSV (self, filename):
        data = pandas.read_csv(filename)
        return data

#Prints the data in dataframe in table format.

    def PrintTable (self, dataframeobj, maxRows = None):
        with pandas.option_context('display.max_rows', maxRows, 'display.max_columns', None):
            print(tabulate(dataframeobj.reset_index(drop=True), headers='keys', tablefmt='fancy_grid'))

#Prints the all output related data in table format with tag name from genome-tags.csv
    def PrintTagsAsTable(self, dataframeobj, maxRows = None):
        tagidname = self.ReadWholeCSV(filename = 'genome-tags.csv')
        tagidname = tagidname.rename(columns={'tagId': 'tagid'})
        columnstoprint = list(dataframeobj.columns.values)
        columnstoprint.insert(1, 'tag')
        mergeddata = dataframeobj.merge(tagidname, on='tagid')
        with pandas.option_context('display.max_rows', maxRows, 'display.max_columns', None):
            printdata = mergeddata[columnstoprint]
            print(tabulate(printdata.reset_index(drop=True), headers='keys', tablefmt='fancy_grid'))

class Tags():

#Base constructor for class Tag
    def __init__(self, actorid=None, genreid=None, userid=None):
        self.actorid= actorid
        self.genreid = genreid
        self.userid = userid

#Gets all the tags into data frame mltags.csv
    def GetAllTags(self):
        alltagsdata =super().ReadWholeCSV(filename="mltags.csv")
        return alltagsdata

#Gets all the tags into data frame only for specific tags specified in taglist
    def GetTagsDataByID(self, taglist=[]):
        alltagsdata = self.GetAllTags()
        tagsdata = alltagsdata[alltagsdata['tagid'].isin(taglist)]
        return tagsdata

#Gets all the tags into data frame for a list of movies specified in movielist
    def GetTagsDataByMovieID(self, movielist=[]):
        alltagsdata = self.GetAllTags()
        tagsdata = alltagsdata[alltagsdata['movieid'].isin(movielist)]
        return tagsdata

#Calculate weights for the timestamps of each tag within the entities(actor,genre,user).
    def CalculateTimestampWeights(self, row, min_timestamp, max_timestamp, inputcolumnname):
        return (pandas.to_datetime(row[inputcolumnname])-min_timestamp)/(max_timestamp-min_timestamp)

#Wrapper function for CalculateTimestampWeights
    def CalculatTimestampWeightsWrapper(self, combineddata, inputcolumnname='timestamp'):
        min_timestamp = pandas.to_datetime(min(combineddata[inputcolumnname]))
        max_timestamp = pandas.to_datetime(max(combineddata[inputcolumnname]))
        combineddata['timestamp_weightage'] = combineddata.apply(self.CalculateTimestampWeights, axis=1, args=(min_timestamp, max_timestamp, inputcolumnname))
        return combineddata

#Using various weights and computes TF.
    def ProcessWeightsToTF(self, combineddata, columnname ='timestamp_weightage'):
        combineddata['tag_weightage'] = combineddata.groupby('tagid')[columnname].transform('sum')
        tfdata = combineddata[['tagid', 'tag_weightage']].drop_duplicates(subset=['tagid'])
        total_weights=tfdata['tag_weightage'].sum()
        tfdata['tf'] = tfdata['tag_weightage']/total_weights
        return tfdata[['tagid', 'tf']].sort_values('tf', ascending=False)

#Computes TF-IDF using TF and IDF.
    def CalculateTFIDF(self, row):
        return row['tf']*row['idf']

#Merges TF and IDF data and calls TF-IDF for each row in a data frame.
    def ProcessTFandIDFtoTFIDF(self, tfdata, idfdata):
        tfidfdata = tfdata.merge(idfdata, on='tagid')
        tfidfdata['tfidf'] = tfidfdata.apply(self.CalculateTFIDF, axis=1)
        return tfidfdata


class Actor(Tags ,Utils):

#Base constructor for class Tag
    def __init__(self,actorid):
        super().__init__(actorid=actorid)

#Gets the total list of actors from imdb-actor-info
    def GetActorList(self):
        actordata = super().ReadWholeCSV(filename="imdb-actor-info.csv")
        return actordata

#Get all the movies data from movie-actor.csv
    def GetMoviesForAllActor(self):
        allactormoviesdata = super().ReadWholeCSV(filename="movie-actor.csv")
        return allactormoviesdata

#Get all the movies data for a actor.
    def GetMoviesForActor(self):
        allactormoviesdata = self.GetMoviesForAllActor()
        filteredactormoviesdata = allactormoviesdata[allactormoviesdata['actorid'] == self.actorid]
        return filteredactormoviesdata

#Get all the movies data for a actor with the rank of that actor under each movie.
    def GetMoviesForActorWithRank(self):

        allmoviesdata = super().ReadWholeCSV(filename="movie-actor.csv")
        #Computing the max and min actor rank in a movie.
        allmoviesdata['max_actor_rank'] = allmoviesdata.groupby(['movieid'])['actor_movie_rank'].transform(max)
        allmoviesdata['min_actor_rank'] = allmoviesdata.groupby(['movieid'])['actor_movie_rank'].transform(min)
        actormoviesdata = allmoviesdata[allmoviesdata['actorid'] == self.actorid]
        return actormoviesdata

#Gets all the tag data for a specific actor.
    def GetTagsDataForActor(self):

        alltagsdata = super().GetAllTags()
        actormoviesdata = self.GetMoviesForActorWithRank()
        return alltagsdata.merge(actormoviesdata, on='movieid')

#Computes the weightage based on the rank of that actor in a movie.
    def compute_actor_weightage(self, row):

        return 1-(row['actor_movie_rank'] - row['min_actor_rank'])/(row['max_actor_rank'] - row['min_actor_rank']) if row['max_actor_rank'] > 1 else 1

#Adds the weightage based on the rank of that actor in a movie and weightage of each taga based on timestamp.
    def aggregate_tf_weightages(self, row):

        return numpy.round(row['actor_rank_weightage'] + row['timestamp_weightage'], decimals=4)

#Generates Term Frequency for an Actor
    def GenerateTF(self, donotprint=False):

        combineddata = self.GetTagsDataForActor()
        combineddata['actor_rank_weightage'] = combineddata.apply(self.compute_actor_weightage, axis=1)
        combineddata = super().CalculatTimestampWeightsWrapper(combineddata)
        combineddata['combined_weights'] = combineddata.apply(self.aggregate_tf_weightages, axis=1)
        if donotprint==False:
            super().PrintTagsAsTable(super().ProcessWeightsToTF(combineddata, columnname='combined_weights'))
        return super().ProcessWeightsToTF(combineddata, columnname='combined_weights')

#Get the tag data related to all the actor for a specific tag list.
    def GetTagsDataForAllActors(self,taglist):

        requiredtagsdata = super().GetTagsDataByID(taglist=taglist)
        allmoviesdata = self.GetMoviesForAllActor()
        return requiredtagsdata.merge(allmoviesdata, on='movieid')

#Computes IDF for an tag that belongs to an actor.
    def CalculateActorIDF(self, row, total_actors):
        return math.log10(total_actors/row['actor_count'])

#Generates Inverse Document Frequency for an Actor
    def GenerateTFIDF(self):

        tfdata = self.GenerateTF(donotprint=True)
        taglist = tfdata['tagid'].tolist()
        combineddata = self.GetTagsDataForAllActors(taglist=taglist)
        combineddata.drop_duplicates(subset=['tagid', 'actorid'], inplace=True)
        combineddata['actor_count'] = combineddata.groupby('tagid')['actorid'].transform('count')
        combineddata.drop_duplicates(subset=['tagid'], inplace=True)
        total_actors = self.GetActorList().shape[0]
        combineddata['idf'] = combineddata.apply(self.CalculateActorIDF, axis=1, total_actors=total_actors)
        tfidfdata = super().ProcessTFandIDFtoTFIDF(tfdata, combineddata[['tagid', 'idf']])
        super().PrintTagsAsTable(tfidfdata.sort_values('tfidf', ascending=False))

class Genre(Tags, Utils):

#Constructor for the class Genre
    def __init__(self,genreid):
        super().__init__(genreid=genreid)

#Gets the movie data for all the genres using mlmovies.csv
    def GetMoviesForAllGenres(self):
        allgenremoviesdata =super().ReadWholeCSV(filename="mlmovies.csv")
        return allgenremoviesdata

#Gets the movie data for a specific genre.
    def GetMoviesForGenre(self):

        allgenremoviesdata = self.GetMoviesForAllGenres()
        filteredgenremoviesdata = allgenremoviesdata[allgenremoviesdata['genres'].str.contains(self.genreid)]
        return filteredgenremoviesdata

#Gets the movie list for a specific genre.
    def GetMoviesListForGenre(self):

        genremovieslist = self.GetMoviesForGenre()['movieid'].tolist()
        return genremovieslist

#Get tags data for a genre using the movies that belongs to the genre
    def GetTagsDataForGenre(self):

        genremovieslist = self.GetMoviesListForGenre()
        tagsdata = super().GetTagsDataByMovieID(movielist=genremovieslist)
        return tagsdata

#Generates Term Frequency for a Genre
    def GenerateTF(self, donotprint=False):

        combineddata = self.GetTagsDataForGenre()
        combineddata = super().CalculatTimestampWeightsWrapper(combineddata)
        if donotprint==False:
            super().PrintTagsAsTable(super().ProcessWeightsToTF(combineddata))
        return super().ProcessWeightsToTF(combineddata)

#Normalize the genre column to multiple rows
    def NormalizeGenreColumn(self, dataframeobj):

        col1 = dataframeobj['tagid']
        col2 = dataframeobj['genres'].str.split('|', expand=True).stack().reset_index(level=1, drop=True)
        sliceddataframe = pandas.concat([col1,col2], axis=1, keys=['tagid','genres'])
        return dataframeobj.drop(['tagid','genres'], axis=1).join(sliceddataframe).reset_index(drop=True)

#Gets the total Genres in the dataset
    def GetGenreList(self):
        allgenremoviesdata =super().ReadWholeCSV(filename="mlmovies.csv")
        allgenremoviesdata = allgenremoviesdata[['movieid', 'genres']]
        col1 = allgenremoviesdata['movieid']
        col2 = allgenremoviesdata['genres'].str.split('|', expand=True).stack().reset_index(level=1, drop=True)
        sliceddataframe = pandas.concat([col1,col2], axis=1, keys=['movieid','genres'])
        allgenredata = allgenremoviesdata.drop(['movieid','genres'], axis=1).join(sliceddataframe).reset_index(drop=True)
        allgenredata.drop_duplicates(subset=['genres'],inplace=True)
        return allgenredata

#Gets the tags data for all genres
    def GetTagsDataForAllGenres(self,taglist):
        requiredtagsdata = super().GetTagsDataByID(taglist=taglist)
        allmoviesdata = self.GetMoviesForAllGenres()
        return requiredtagsdata.merge(allmoviesdata, on= 'movieid')

#Calculates IDF across the genres
    def CalculateGenreIDF(self, row, total_genres):
        return math.log10(total_genres/row['genres_count'])

#Calculates IDF across the two genres using movies under each genre(used only for GenerateTFIDFDIFF).
    def CalculateSpecificGenreIDF(self, row, count):
        return math.log10(count/row['movie_count'])

#Generates Inverse Document Frequency for an Genre
    def GenerateTFIDF(self):
        tfdata = self.GenerateTF(donotprint=True)
        taglist = tfdata['tagid'].tolist()
        combineddata = self.GetTagsDataForAllGenres(taglist=taglist)
        combineddata = self.NormalizeGenreColumn(combineddata[['tagid','genres']])
        combineddata.drop_duplicates(subset=['tagid', 'genres'],inplace=True)
        combineddata['genres_count'] = combineddata.groupby('tagid')['genres'].transform('count')
        combineddata.drop_duplicates(subset=['tagid'],inplace=True)
        total_genres = self.GetGenreList().shape[0]
        combineddata['idf'] = combineddata.apply(self.CalculateGenreIDF, axis=1, total_genres=total_genres)
        tfidfdata = super().ProcessTFandIDFtoTFIDF(tfdata, combineddata[['tagid', 'idf']])
        super().PrintTagsAsTable(tfidfdata.sort_values('tfidf', ascending=False))

#Generates TF-IDF-DIFF which finds all the tags in genre that differentiates it from othergenre
    def GenerateTFIDFDIFF(self,othergenre):

        tfdata = self.GenerateTF(donotprint=True)

        taglist = tfdata['tagid'].tolist()

        combineddata = self.GetTagsDataForAllGenres(taglist=taglist)

        filteredgenremoviesdata = combineddata[combineddata['genres'].str.contains(self.genreid + "|" + othergenre.genreid)]

        # print(filteredgenremoviesdata)

        filteredgenremoviesdata.drop_duplicates(subset=['tagid', 'movieid'],inplace=True)

        filteredgenremoviesdata['movie_count'] = filteredgenremoviesdata.groupby('tagid')['movieid'].transform('count')

        filteredgenremoviesdata.drop_duplicates(subset=['tagid'],inplace=True)

        moviesingenre1 = self.GetMoviesForGenre()
        moviesingenre2 = othergenre.GetMoviesForGenre()

        combinedmovielist = moviesingenre1.append(moviesingenre2)

        combinedmovielist.drop_duplicates(subset=['movieid'],inplace=True)

        combinedmovie_count = combinedmovielist.shape[0]

        filteredgenremoviesdata['idf'] = filteredgenremoviesdata.apply(self.CalculateSpecificGenreIDF, axis=1 , count=combinedmovie_count)

        tfidfdata = super().ProcessTFandIDFtoTFIDF(tfdata, filteredgenremoviesdata[['tagid', 'idf']])

        super().PrintTagsAsTable(tfidfdata.sort_values('tfidf', ascending=False))

#Computes PDIFF1 using the varaiables needed for a calculation
    def CalculatePDIFF1(self, row, t1, t12):

        x = (row['movies_count1'] + (float(row['movies_count12'])/t12))/(t1+1)
        y = (row['movies_count12'] - row['movies_count1'] + (float(row['movies_count12'])/t12)) / (t12-t1 + 1)

        pdiff1 = math.log((x * (1 - y)) / (y * (1 - x)))
        return pdiff1 * (x - y)

#Computes PDIFF2 using the variables needed for the calculation.
    def CalculatePDIFF2(self, row, t2, t12):

        x = ((t2-row['movies_count2'])/(t2+1))
        y = ((t12 - row['movies_count12']) - (t2 - row['movies_count2'])) / (t12-t2 + 1)

        pdiff2 = math.log((x * (1 - y)) / (y * (1 - x)))

        return pdiff2

#Generates PDIFF1 which finds all the tags in genre that differentiates it from othergenre
    def GeneratePDIFF1(self, other):

        combineddata1 = self.GetTagsDataForGenre()
        # combineddata1 = combineddata1[combineddata1['movieid'].isin([8171,4155,5783])]#

        combineddata1.drop_duplicates(subset=['movieid', 'tagid'],inplace=True)

        totalmovies1 = len(numpy.unique(self.GetMoviesListForGenre()))

        combineddata2 = other.GetTagsDataForGenre()
        # combineddata2 = combineddata2[combineddata2['movieid'].isin([3963,4922,4214])]

        combineddata2.drop_duplicates(subset=['movieid', 'tagid'],inplace=True)

        combineddata12 = combineddata1.append(combineddata2)

        combineddata12.drop_duplicates(subset=['movieid', 'tagid'],inplace=True)

        totalmovies12 = len(numpy.unique(other.GetMoviesListForGenre() + self.GetMoviesListForGenre()))

        combineddata1['movies_count1'] = combineddata1.groupby('tagid')['movieid'].transform('count')

        filtereddata1 = combineddata1[['tagid','movies_count1']].drop_duplicates(subset=['tagid'])

        combineddata12['movies_count12'] = combineddata12.groupby('tagid')['movieid'].transform('count')

        filtereddata12 = combineddata12[['tagid','movies_count12']].drop_duplicates(subset=['tagid'])

        filtereddata112 = filtereddata1.merge(filtereddata12, how='left', on= 'tagid').fillna(0)

        filtereddata112['movies_count12'] = filtereddata112['movies_count12'].astype(int)

        filtereddata112['pdiff1'] = filtereddata112.apply(self.CalculatePDIFF1, axis=1, args=(totalmovies1,totalmovies12))

        super().PrintTagsAsTable(filtereddata112[['tagid','pdiff1']].sort_values('pdiff1', ascending=False))

#Generates PDIFF2 which finds all the tags in genre that differentiates it from othergenre.
    def GeneratePDIFF2(self, other):

        combineddata1 = self.GetTagsDataForGenre()
        # combineddata1 = combineddata1[combineddata1['movieid'].isin([8171,4155,5783])]#
        combineddata1.drop_duplicates(subset=['movieid', 'tagid'],inplace=True)

        combineddata2 = other.GetTagsDataForGenre()
        # combineddata2 = combineddata2[combineddata2['movieid'].isin([8171,3963,4922,4214])]
        combineddata2.drop_duplicates(subset=['movieid', 'tagid'],inplace=True)

        totalmovies2 = len(numpy.unique(other.GetMoviesListForGenre()))

        combineddata12 = combineddata1.append(combineddata2)
        combineddata12.drop_duplicates(subset=['movieid', 'tagid'],inplace=True)

        totalmovies12 = len(numpy.unique(other.GetMoviesListForGenre() + self.GetMoviesListForGenre()))

        combineddata1['movies_count1'] = combineddata1.groupby('tagid')['movieid'].transform('count')

        filtereddata1 = combineddata1[['tagid','movies_count1']].drop_duplicates(subset=['tagid'])

        combineddata2['movies_count2'] = combineddata2.groupby('tagid')['movieid'].transform('count')

        filtereddata2 = combineddata2[['tagid','movies_count2']].drop_duplicates(subset=['tagid'])

        combineddata12['movies_count12'] = combineddata12.groupby('tagid')['movieid'].transform('count')

        filtereddata12 = combineddata12[['tagid','movies_count12']].drop_duplicates(subset=['tagid'])

        temp = filtereddata1.merge(filtereddata2, how='left', on= 'tagid').fillna(0)

        filtereddata1212 = temp.merge(filtereddata12, how='left', on= 'tagid').fillna(0)

        filtereddata1212['movies_count12'] = filtereddata1212['movies_count12'].astype(int)

        filtereddata1212['pdiff2'] = filtereddata1212.apply(self.CalculatePDIFF2, axis=1, args=(totalmovies2 ,totalmovies12))

        super().PrintTagsAsTable(filtereddata1212[['tagid','pdiff2']].sort_values('pdiff2', ascending=False))

class User(Tags ,Utils):

#Constructor for a User class
    def __init__(self,userid):
        super().__init__(userid=userid)

#Gets the user list using mlusers.csv.
    def GetUserList(self):
        userrdata =super().ReadWholeCSV(filename="mlusers.csv")
        return userrdata

#Gets movies data for a all the users using mlratings.csv.
    def GetMoviesForAllUsers(self):
        allusermoviesdata =super().ReadWholeCSV(filename="mlratings.csv")
        return allusermoviesdata

#Gets movies data for a specific user.
    def GetMoviesForUser(self):
        allusermoviesdata = self.GetMoviesForAllUsers()
        filteredusermoviesdata = allusermoviesdata[allusermoviesdata['userid'] == self.userid]
        return filteredusermoviesdata

#Gets tag data for a users.
    def GetTagsDataForUser(self):
        alltagsdata = super().GetAllTags()
        usermoviesdata = self.GetMoviesForUser()
        return alltagsdata.merge(usermoviesdata, on='movieid')

#Generates Term Frequency for an Actor
    def GenerateTF(self, donotprint=False):
        combineddata = self.GetTagsDataForUser()
        combineddata = super().CalculatTimestampWeightsWrapper(combineddata, inputcolumnname='timestamp_x')
        if donotprint==False:
            super().PrintTagsAsTable(super().ProcessWeightsToTF(combineddata))
        return super().ProcessWeightsToTF(combineddata)

#Gets tag data for all the users using a taglist.
    def GetTagsDataForAllUsers(self,taglist):
        requiredtagsdata = super().GetTagsDataByID(taglist=taglist)
        allmoviesdata = self.GetMoviesForAllUsers()
        return requiredtagsdata.merge(allmoviesdata, on= 'movieid')

#Calculates IDF across users.
    def CalculateUserIDF(self, row, total_users):
        return math.log10(total_users/row['user_count'])

#Generates Inverse Document Frequency for a User
    def GenerateTFIDF(self):
        tfdata = self.GenerateTF(donotprint=True)
        taglist = tfdata['tagid'].tolist()
        combineddata = self.GetTagsDataForAllUsers(taglist=taglist)
        combineddata.drop_duplicates(subset=['tagid', 'userid_y'],inplace=True)
        combineddata['user_count'] = combineddata.groupby('tagid')['userid_y'].transform('count')
        combineddata.drop_duplicates(subset=['tagid'], inplace=True)
        total_users = self.GetUserList().shape[0]
        combineddata['idf'] = combineddata.apply(self.CalculateUserIDF, axis=1, total_users=total_users)
        tfidfdata = super().ProcessTFandIDFtoTFIDF(tfdata, combineddata[['tagid', 'idf']])
        super().PrintTagsAsTable(tfidfdata.sort_values('tfidf', ascending=False))
#
#
# if __name__ == "__main__" :
#
#     warnings.filterwarnings("ignore")
#
#     command = sys.argv[1]
#
#     if(command == "print_actor_vector"):
#         actorobj = Actor(int(sys.argv[2]))
#         if(sys.argv[3] == 'TF'):
#             actorobj.GenerateTF()
#         if(sys.argv[3] == 'TF-IDF'):
#             actorobj.GenerateTFIDF()
#
#     if(command == "print_genre_vector"):
#         genreobj = Genre(sys.argv[2])
#         if(sys.argv[3] == 'TF'):
#             genreobj.GenerateTF()
#         if(sys.argv[3] == 'TF-IDF'):
#             genreobj.GenerateTFIDF()
#
#     if(command == "print_user_vector"):
#         userobj = User(int(sys.argv[2]))
#         if(sys.argv[3] == 'TF'):
#             userobj.GenerateTF()
#         if(sys.argv[3] == 'TF-IDF'):
#             userobj.GenerateTFIDF()
#
#     if(command == "differentiate_genre"):MD
#         genreobj1 = Genre(sys.argv[2])
#         genreobj2 = Genre(sys.argv[3])
#         if(sys.argv[4] == 'TF-IDF-DIFF'):
#             genreobj1.GenerateTFIDFDIFF(genreobj2)
#         if(sys.argv[4] == 'P-DIFF1'):
#             genreobj1.GeneratePDIFF1(genreobj2)
#         if(sys.argv[4] == 'P-DIFF2'):
#             genreobj1.GeneratePDIFF2(genreobj2)
#

a = Actor(177901)

print ("Computing TF for actor 177901")

a.GenerateTF()

print ("Computing TF-IDF for actor 177901")

a.GenerateTFIDF()
