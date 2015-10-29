
from multiplex import *
import networkx as nx
import utility
import random

print 'Reading metro network.'
metro = utility.read_metro(directory = 'data/metro', file_prefix = 'metro') # networkx graph

print 'Reading street network'
streets = utility.read_streets(directory = 'data/street', file_prefix = 'street') # networkx graph

layer_dict = {'metro' : metro, 'streets' : streets} # need a dict to add to multiplex.

multi = multiplex() # initialize empty multiplex
multi.add_layers(layer_dict) # add metro and street layers to multiplex
multi.spatial_join(layer1 = 'metro', layer2 = 'streets', transfer_speed = .1, base_cost = 0, both = True)

N = len(multi.as_graph().node)
norm = (N-1) * (N-2) / 1000

nx.set_edge_attributes(multi.G, 'weight', nx.get_edge_attributes(multi.G, 'dist_km'))
multi.add_epsilon(weight = 'weight', epsilon = .000001)

for beta in [.1, .5, .9, 1, 2]:
	multi.scale_edge_attribute(layer = 'metro', attribute = 'weight', beta = beta)
	multi.igraph_betweenness_centrality(layers = multi.get_layers(), weight = 'weight', attrname = str(beta) + 'bc')
	multi.streets_betweenness_plot(draw_metro = True, file_name = str(beta) + 'x' + 'betweenness.png', attrname = str(beta) + 'bc', norm = norm)
	multi.scale_edge_attribute(layer = 'metro', attribute = 'weight', beta = 1/beta)

multi.to_txt('data/multiplex', 'multi')
