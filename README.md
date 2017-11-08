# Movie-Recommendation-Engine

PHASE 1:

• Task 1: Implement a program which considers all the movies an actor played and creates and stores a weighted tag vector for           each actor using (time-weighted) TF as well as TF-IDF models. When combining tag vectors under TF or TF-IDF models,           newer tags should be given higher weight than older tags. Similarly, movies where the given actor appears with a               lower rank should be given a relatively higher weight.

• Task 2: Implement a program which considers all movies of a given genre to create a combined tag vector for the genre. When           combining tag vectors under TF or TF-IDF models, newer tags should be given higher weight than older tags.

• Task 3: Implement a program which considers all movies watched by a user to create a combined tag vector for the user. When           combining tag vectors under TF or TF-IDF models, newer tags should be given higher weight than older tags.

• Task 4: Implement a program which considers two genres, g1 and g2, and explains in what ways the g1 differs from g2. You               will consider three models, TF-IDF-DIFF, P-DIFF1, P-DIFF2:
          
          – In the TF-IDF-DIFF model, you will consider the set of movies for the given genre to compute TF, but the set,      movies(g1) ∪ movies(g2), of all movies in genres, g1 and g2, to compute IDF.

          – In the P-DIFF1 model, you will identify the weight, wi,j, of the tag tj for genre g1 relying on a probabilistic   feedback mechanism 
        
DATASET : Download the MovieLens+IMDB data data.
          – mlmovies(movieId,movieName,genres)
          – mltags(userId,movieId,tagid,timestamp)
          – mlratings(movieId,userId,imdbId,rating,timestamp) – genome-tags(tagId,tag)
          – movie-actor (movieId, actorId, actorMovieRank)
          – imdb-actor-info (actorId, name, gender)
          – mlusers(userId)

PHASE 2:

• Task 1:
Task 1a: Implement a program which, given a genre, identifies and reports the top-4 latent semantics/topics using
                   
Task 1b: Implement a program which, given a genre, identifies and reports the top-4 latent semantics/topics using
                    
Task 1c: Implement a program which, given an actor, finds and ranks the 10 most similar actors by comparing actors
                  
Task 1d: Implement a program which, given a movie, finds and ranks the 10 most related actors who have not acted in the movie, leveraging the given movie’s
                   
