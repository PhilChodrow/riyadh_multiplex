import multiplex as mx
import utility
import networkx as nx

def main():
	metro_dir = '1. data/metro'
	metro_prefix = 'metro'

	street_dir = '1. data/street'
	street_prefix = 'street'

	taz_dir = '1. data/taz'
	taz_file = '/taz_nodes.txt'

	print 'Reading TAZ connector nodes'
	taz = utility.graph_from_txt(nodes_file_name = taz_dir + taz_file,
	                             nid = 'id',
	                             sep = '\t')
	pos = {n : (taz.node[n]['lat'], taz.node[n]['lon']) for n in taz}
	nx.set_node_attributes(taz, 'pos', pos)

	print 'Reading metro network.'
	metro = utility.read_metro(directory = metro_dir, file_prefix = metro_prefix) # networkx graph

	print 'Reading street network'
	streets = utility.read_streets(directory = street_dir, file_prefix = street_prefix) # networkx graph

	layer_dict = {'metro' : metro, 'streets' : streets, 'taz' : taz} # need a dict to add to multiplex.

	multi = mx.multiplex() # initialize empty multiplex
	multi.add_layers(layer_dict) # add metro and street layers to multiplex
	print multi.get_layers()
	multi.spatial_join(layer1 = 'metro', layer2 = 'streets', transfer_speed = 100000000000, base_cost = 0, both = True)
	multi.spatial_join(layer1 = 'taz', layer2 = 'streets', transfer_speed = 10000000000000000, base_cost = 0, both = True)

	multi.to_txt('2. multiplex', 'multiplex')

if __name__ == "__main__":
    main()