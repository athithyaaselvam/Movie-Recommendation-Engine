import random
import sys
from collections import defaultdict
from operator import itemgetter
from task3input import VectorSpace

class ImplementLHS:

    def __init__(self,hash_family,k,L,vector):
        self.hash_family = hash_family
        self.k = k
        self.L = 0
        self.hash_tables = []
        self.resize(L)
        self.vec = vector
        self.totalMovies = 0
        self.UniqueMovies = 0
        self.candidates = set()

    def resize(self,L):
        if L < self.L:
            self.hash_tables = self.hash_tables[:L]
        else:            
            #print "======================================================================"
            #print "k="+str(self.k)
            #print "l="+str(L)            
            hash_funcs = [[self.hash_family.create_hash_func() for h in xrange(self.k)] for l in xrange(self.L,L)]			
            #print "hash_funcs len = "+str(len(hash_funcs))+"\n hash_funcs = "+str(hash_funcs)			
            self.hash_tables.extend([(g,defaultdict(lambda:[])) for g in hash_funcs])
            #print "hash_tables len = "+str(len(self.hash_tables))+"\n hastables = "+str(self.hash_tables)
			
            #for g,table in self.hash_tables:
            #    print "set of functions = "+str(g)+" hashtable= "+str(table)			
			
				
				

    def hash(self,g,p):
        arr = [h.hash(p) for h in g]
        #print "set of hash keys "
        #print arr
        combined =  self.hash_family.combine(arr)
        #print "combined="+str(combined)
        return combined

    def index(self,points):
        self.points = points
        for g,table in self.hash_tables:
            for ix,p in enumerate(self.points):
                table[self.hash(g,p)].append(ix)
        # reset stats
		#print "table"
		#print table
        self.tot_touched = 0
        self.num_queries = 0

    def query(self,q,metric,max_results):        
        for g,table in self.hash_tables:
            matches = table.get(self.hash(g,q),[])
            self.totalMovies = self.totalMovies + len(matches)
            self.candidates.update(matches)
        
        self.tot_touched += len(self.candidates)
        self.num_queries += 1

        self.UniqueMovies = len(self.candidates)
        self.candidates = [(ix,metric(q,self.points[ix])) for ix in self.candidates]
        self.candidates.sort(key=itemgetter(1))
        self.candidates.pop(0)
        return self.candidates[:max_results]

    def get_avg_touched(self):
        """  """
        return self.tot_touched/self.num_queries


class L2DisthashFamily:

    def __init__(self,w,d):
        self.w = w
        self.d = d

    def create_hash_func(self):
        
        return L2Disthash(self.rand_vec(),self.rand_offset(),self.w)

    def rand_vec(self):
        return [random.gauss(0,1) for i in xrange(self.d)]

    def rand_offset(self):
        return random.uniform(0,self.w)

    def combine(self,hashes):
        """
        
        """
        return str(hashes)
def dot(u,v):
	return sum(ux*vx for ux,vx in zip(u,v))

class L2Disthash:

    def __init__(self,r,b,w):
        self.r = r
        self.b = b
        self.w = w

    def hash(self,vec):
        return int((dot(vec,self.r)+self.b)/self.w) # (dot product of the point with the randomized vector plus offset vector) / w

def L2_norm(u,v):
        return sum((ux-vx)**2 for ux,vx in zip(u,v))**0.5


class RunLHS:
    """
    
    """

    def __init__(self,points,num_neighbours,VectorSpace,movieName):
        self.points = points
        self.num_neighbours = num_neighbours
        self.vec = VectorSpace
        self.movie = movieName
        self.movieNameVsRelevance = {}
        self.candidates = set()
        

    def run(self,name,metric,hash_family,k_vals,L_vals,shouldPrint):
        
        for k in k_vals:        
            lsh = ImplementLHS(hash_family,k,0,self.vec)
            for L in L_vals:
                lsh.resize(L)
                lsh.index(self.points)               
                movieId = self.vec.getMovieID(self.movie)                
                queryPoint = self.vec.getQueryPoint(movieId)
                #print "Query Point"
                #print queryPoint

                self.candidates = lsh.query(queryPoint,metric,self.num_neighbours+1)
                
                ans =[moviepoint for moviepoint,dist in  self.candidates]
                if shouldPrint:
                	print "RELATED MOVIES FOR YOU"
                	for m in ans:
                		print self.vec.movieIndexVsName[m]
                		self.movieNameVsRelevance[self.vec.movieIndexVsName[m]]=0
                		#print "executing"
                	print "NUMBER OF UNIQUE MOVIES CONSIDERED "
                	print lsh.UniqueMovies
                	print "OVERALL NUMBER OF MOVIES CONSIDERED "
                	print lsh.totalMovies

    def linear(self,q,metric,max_results):
        
        self.candidates = [(ix,metric(q,p)) for ix,p in enumerate(self.points)]
        return sorted(self.candidates,key=itemgetter(1))[:max_results]


if __name__ == "__main__":
	task = int(sys.argv[1])
	d = 500
	L = int(raw_input('Enter number of layers L:'))
	K = int(raw_input('Enter number of hashes per layer k:'))
	num_neighbours = int(raw_input('Enter r:'))
	#num_neighbours = num_neighbours+1
	movie = raw_input('Enter Movie Name ')
	num_points = 4323
	vec = VectorSpace()
	points = vec.getLSHInput()
	radius = 0.1
	tester = RunLHS(points,num_neighbours-1,vec,movie)
	args = {'name':'L2','metric':L2_norm,'hash_family':L2DisthashFamily(10*radius,d),'k_vals':[K],'L_vals':[L],'shouldPrint':True}
	tester.run(**args)

	#print "check movieNameVsRelevance len = "
	#print len(tester.movieNameVsRelevance)
	#for movieName,relevance in tester.movieNameVsRelevance.items():
	#	print movieName
	if task == 3:
		exit()
	for movieName,relevance in tester.movieNameVsRelevance.items():
		res = raw_input("Is "+movieName+" relevant ")
		if res == 'Y' or res == 'y':
			tester.movieNameVsRelevance[movieName]=1
			#print "setting relevance 1"
	#print "relevance loop over"
	allCandidateMovies = set()
	for movieName,relevance in tester.movieNameVsRelevance.items():
		if relevance == 1:
			#print "\n Getting related movies for "+movieName
			tester2 = RunLHS(points,num_neighbours,vec,movieName)
			args = {'name':'L2','metric':L2_norm,'hash_family':L2DisthashFamily(10*radius,d),'k_vals':[K],'L_vals':[L],'shouldPrint':False}
			tester2.run(**args)
			allCandidateMovies.update(tester2.candidates)
	#print "out of loop"
	#allCandidateMovies.sort(key = itemgetter(1))
	finalRes = sorted(allCandidateMovies,key=lambda x:x[1])
	finalRes = finalRes[:num_neighbours]
	ans = [p for p,d in finalRes]
	print "Revision of Movie Suggestion after Relevance Feedback"
	for m in ans:
		print vec.movieIndexVsName[m]
