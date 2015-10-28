
from multiplex import *
import networkx as nx
import utility

# print 'Reading metro network.'
# metro = utility.read_metro(directory = 'data/metro', file_prefix = 'metro') # networkx graph




# print 'Reading street network'
# streets = utility.read_streets(directory = 'data/street', file_prefix = 'street') # networkx graph

# add = ['dist_km', 'cost_time_m']
# streets = utility.remove_flow_through(streets, add)
# streets = utility.remove_flow_through_2(streets, add)

# layer_dict = {'metro' : metro, 'streets' : streets} # need a dict to add to multiplex.

# multi = multiplex() # initialize empty multiplex
# multi.add_layers(layer_dict) # add metro and street layers to multiplex
# multi.weight_layers(weight = 'cost_time_m', epsilon = .000001)


# E = len(multi.G.edges())

# print 'Before spatial join, the layers of the multiplex are ' + str(multi.get_layers()) + '. It has ' + str(E) + ' edges.'

# multi.spatial_join(layer1 = 'metro', layer2 = 'streets', transfer_speed = .1, base_cost = 2, both = True)

# F = len(multi.G.edges())

# print 'After spatial join, the layers of the multiplex are ' + str(multi.get_layers()) + '. It has ' + str(F) + ' edges, of which ' + str(F - E) + ' are transfers. Now we\'ll save the network in a .txt file.' 



# layers = ['metro', 'streets']

# multi.igraph_betweenness_centrality(layers = layers, calc_layers = layers, weight = 'cost_time_m')

# multi.to_txt('data/multiplex', 'multi')




# multi.summary(print_summary = True)

multi = utility.multiplex_from_txt(nodes_file_name = 'data/multiplex/multi_nodes.txt',
                           edges_file_name = 'data/multiplex/multi_edges.txt',
                           sep = '\t',
                           nid = 'id',
                           eidfrom = 'source',
                           eidto = 'target')

multi.summary(print_summary = True)


N = len(multi.as_graph().node)
norm = (N-1) * (N-2) / 1000

print norm

for beta in [.1, 1]:
	multi.weight_layers(weight = 'dist_km', epsilon = 0.000001)
	multi.scale_edge_attribute(layer = 'metro', attribute = 'dist_km', beta = beta)

	weight_name = str(beta) + 'x' + 'dist_km'
	print weight_name
	 
	multi.igraph_betweenness_centrality(layers = ['streets', 'metro'], weight = weight_name, attrname = str(beta) + 'bc')
# 	multi.streets_betweenness_plot(draw_metro = True, file_name = str(beta) + 'x' + 'betweenness.png', attrname = str(beta) + 'bc', norm = norm)

multi.to_txt('data/multiplex', 'multi2')


# multi2.streets_betweenness_plot(draw_metro = True, file_name = 'test.png')


# multi2.summary(print_summary = True)
