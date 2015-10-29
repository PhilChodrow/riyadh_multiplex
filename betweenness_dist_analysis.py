
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

nx.set_edge_attributes(multi.G, 'weight', nx.get_edge_attributes(multi.G, 'dist_km'))
multi.add_epsilon(weight = 'weight', epsilon = .000001)


# multi.igraph_betweenness_centrality(layers = multi.get_layers(), weight = 'weight', attrname = 'bc')
# N = multi.as_graph()

# max_bc = np.max([N.node[n]['bc'] for n in N.node])
# print max_bc

vmax = 1.6*1e11
vmin = .1*1e11


betas = [.01, .1, .3, .5, .7, .9, 1]
betas = [.3]

for beta in betas:
	multi.scale_edge_attribute(layer = 'metro', attribute = 'weight', beta = beta)
	multi.igraph_betweenness_centrality(layers = multi.get_layers(), weight = 'weight', attrname = str(beta) + 'bc')
	multi.betweenness_plot_interpolated(layer1 = 'streets', 
	                                    layer2 = 'metro', 
	                                    title = r"Riyadh Streets: Betweenness (distance weighting, $\beta =" + str(beta) + "$)", 
	                                    measure = str(beta) + 'bc', 
	                                    file_name = 'dist_betweenness_beta' + str(beta) + '.png',
	                                    vmin = None,
	                                    vmax = None)

	multi.scale_edge_attribute(layer = 'metro', attribute = 'weight', beta = 1/beta)

multi.to_txt('data/multiplex', 'multi')
