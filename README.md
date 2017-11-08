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

• TASK 1:

Task 1a: Implement a program which, given a genre, identifies and reports the top-4 latent semantics/topics using 
1. PCA in TF-IDF space of tags,
2. SVD in TF-IDF space of tags, and
3. LDA in the space of tags.
                   
Task 1b: Implement a program which, given a genre, identifies and reports the top-4 latent semantics/topics using 
1. PCA in TF-IDF space of actors,
2. SVD in TF-IDF space of actors, and ∗ LDA in the space of actors.
                    
Task 1c: Implement a program which, given an actor, finds and ranks the 10 most similar actors by comparing actors 
1. TF-IDF tag vectors,
2. top-5 latent semantics (PCA, SVD, or LDA) in the space of tags.
                  
Task 1d: Implement a program which, given a movie, finds and ranks the 10 most related actors who have not acted in the movie, leveraging the given movie’s 
1. TF-IDF tag vectors,
2. top-5 latent semantics (PCA, SVD, or LDA) in the space of tags.


• TASK 2:

Task 2a: Implement a program which 
1. creates an actor-actor similarity matrix (using tag vectors),
2. performs SVD on this actor-actor similarity matrix,
3. reports the top-3 latent semantics, in the actor space, underlying this actor-actor similarity matrix, and
4. partitionstheactorsinto3non-overlappinggroupsbasedontheirdegreesofmembershipstothese3semantics.

Task 2b: Implement a program which 
1. creates a coactor-coactor matrix based on co-acting relationships (recording the number of times two actors played acted in the same movie),
2. performs SVD on this coactor-coactor matrix,
3. reports the top-3 latent semantics, in the actor space, underlying this coactor-coactor matrix, and
4. partitionstheactorsinto3non-overlappinggroupsbasedontheirdegreesofmembershipstothese3semantics.

Task 2c: Implement a program which
1. creates an actor-movie-year tensor, where the tensor contains 1 for any actor-movie-year triple if the given actor played in the stated movie and the movie was released in the stated year (the tensor contains 0 for all other triples)
2. performs CP on this actor-movie-year tensor with target rank set to 5,
3. reports the top-5 latent
∗ actor ∗ movie ∗ year
semantics underlying this tensor, and
4. partitions
∗ actors ∗ movies ∗ years
into 5 non-overlapping groups based on their degree of memberships to these 5 semantics.

Task 2d: Implement a program which 
1. createstag-movie-ratingtensor,wherethetensorcontains1foranytag-movie-ratingtripleifthegiventagwas assigned to a movie by at least one user and the movie has received an average rating lower than or equal to the given rating value (the tensor contains 0 for all other triples)
2. performs CP on this actor-movie-rating tensor with target rank set to 5,
3. reports the top-5 latent semantics in terms of
∗ tag
∗ movie ∗ rating
memberships underlying this tensor, and
4. partitions
∗ tag
∗ movies ∗ ratings
into 5 non-overlapping groups based on their degree of memberships to these 5 semantics.


• TASK 3:

Task 3a: Implement a program which
1. creates an actor-actor similarity matrix (using tag vectors),
2. given a set, S, of “seed” actors (indicating the user’s interest), identifies the 10 most related actors to the actors
in the given seed set using Random Walk with ReStarts (RWR, or Personalized PageRank, PPR) score. See
J.-Y. Pan, H.-J. Yang, C. Faloutsos, and P. Duygulu. Au- tomatic multimedia cross-modal correlation discovery. In KDD, pages 653658, 2004.

Task 3b: Implement a program which
1. creates a coactor-coactor matrix based on the number of movies two actors acted in together,
2. given a set, S, of “seed” actors (indicating the user’s interest), identifies the 10 most related actors to the actors
in the given seed set using RWR.


• TASK 4: 

Implement a program which, given all the information available about the set of movies a given user has watched, recommends the user 5 more movies to watch.

