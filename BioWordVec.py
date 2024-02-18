'''
Reference implementation for joint learning word vector based on bio text and mesh knowledge.
The implementation is based on Fasttext and Node2vec.
'''
#coding=utf-8

import argparse
import networkx as nx
import node2vec
from gensim.models import FastText
import random
import gzip
import pickle as pkl


def parse_args():

	parser = argparse.ArgumentParser()

	parser.add_argument('--input_corpus', nargs='?', default='./data/pubmed_sample',
	                    help='Input biomedical corpus')

	parser.add_argument('--input_mesh', nargs='?', default='./data/MeSH_graph.edgelist',
						help='Input mesh knowledge')

	parser.add_argument('--input_dic', nargs='?', default='./data/MeSH_dic.pkl.gz',
						help='Input mesh dic')

	parser.add_argument('--output_model', nargs='?', default='./pubmed_mesh_test',
	                    help='output of word vector model')

	parser.add_argument('--output_bin', nargs='?', default='./pubmed_mesh_test.bin',
						help='output of word vector bin file')

	parser.add_argument('--dimensions', type=int, default=200,
	                    help='Number of dimensions. Default is 200.')

	parser.add_argument('--walk-length', type=int, default=50,
	                    help='Length of walk per source. Default is 100.')

	parser.add_argument('--num-walks', type=int, default=2,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--windows', type=int, default=5,
                    	help='Context size for optimization. Default is 5.')

	parser.add_argument('--iter', default=5, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--min_count', default=5, type=int,
						help='Number of ignores min_count')

	parser.add_argument('--sg', default=1, type=int,
						help='if 1, skip-gram is used, otherwise, CBOW')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=2,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

class MySentences(object): # this class is used to read the corpus and mesh knowledge
	def __init__(self, mesh_list,pubmed_file):
		self.mesh_list = mesh_list # a list of mesh terms
		self.pubmed_file=pubmed_file # the file of pubmed corpus, including the abstracts and titles
	def __iter__(self):

		for instance in self.mesh_list: # read the mesh knowledge

			yield instance

		for line in open(self.pubmed_file, 'r'): # read the pubmed corpus
			yield str(line).split() # each line is a cleaned sentence, all the words are lowercased, no punctuation


def main(args):
	f_pkl = gzip.open(args.input_dic, 'r') # read the mesh dictionary
	mesh_dict = pkl.load(f_pkl) # mesh_dict is a dictionary, the key is the mesh term, the value is the index of the mesh term
	f_pkl.close()

	# read the mesh knowledge, G is MeSH term graph
	G = nx.read_edgelist(args.input_mesh, nodetype=str, create_using=nx.DiGraph()) 
	for edge in G.edges(): # set the weight of the edges to 1
		G[edge[0]][edge[1]]['weight'] = 1 # the reason to set the weight to 1 is that the node2vec implementation requires the weight of the edges

	G = G.to_undirected() # convert the directed graph to undirected graph, then can randomly walk on the graph

	G = node2vec.Graph(G, args.directed, args.p, args.q) # create a node2vec graph

	G.preprocess_transition_probs() # preprocess the transition probabilities of the transition from one node to another node

	walks = G.simulate_walks(args.num_walks, args.walk_length) # simulate the walks on the graph

	walks = [list(map(str, walk)) for walk in walks] # convert the walks to string

	new_walks=[] # this list is used to store the walks, each walk is a list of mesh term index

	node_set=set([]) # this set is used to store the mesh term index

	for instance in walks: 
		temp_list=[]
		for node in instance:
			node_set.add(node)
			if node in mesh_dict:
				temp_list.append(mesh_dict[node]) # convert the walks to the index of the mesh term
		new_walks.append(temp_list)

	# model = FastText(MySentences(new_walks,args.input_corpus), size=args.dimensions, window=args.windows, min_count=args.min_count, workers=args.workers,
	# 				 sg=args.sg, iter=args.iter)
	
	# train the word vector model
	model = FastText(sentences=MySentences(new_walks,args.input_corpus), vector_size=args.dimensions, window=args.windows, min_count=args.min_count, 
				  workers=args.workers, sg=args.sg, epochs=args.iter)
	# new_walks is the mesh knowledge, args.input_corpus is the pubmed corpus
	# windows means the context window size
	# min_count means the minimum frequency of the word, 
	# workers means the number of threads, 
	# sg means the training algorithm, skip-gram or CBOW
	# iter means the number of iterations

	model.save(args.output_model) # save the word embedding model to pubmed_mesh_test

	print(model)

	model.wv.save_word2vec_format(args.output_bin, binary=True) # output of word vector bin file

if __name__ == "__main__":
	args = parse_args()
	main(args)
