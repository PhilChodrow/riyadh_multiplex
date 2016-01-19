from metro import analysis
from metro import utility 
from metro import multiplex as mx
import networkx as nx
import numpy as np


def main():

	taz = read_taz('1_data/taz', 'taz')
	taz = clean_taz(taz)

	streets = read_streets('1_data/street', 'street')
	streets = clean_streets(streets)

	metro = read_metro('1_data/metro', 'metro')
	metro = clean_metro(metro)

	layer_dict = {'metro'   : metro, 
	              'streets' : streets, 
	              'taz'     : taz}
	multi = mx.multiplex()
	multi.add_layers(layer_dict)

	multi.spatial_join(layer1 = 'metro', 
	                   layer2 = 'streets', 
	                   transfer_speed = 1e10,
	                   base_cost = 0,
	                   capacity = 1e10,
	                   both = True)

	multi.spatial_join(layer1 = 'taz', 
	                   layer2 = 'streets', 
	                   transfer_speed = 1e10, 
	                   base_cost = 0, 
	                   capacity = 1e10, 
	                   both = True, )

	multi.to_txt('2_multiplex', 'multiplex_unscaled')

def read_metro(directory, file_prefix):
	"""convenience function to quickly read in and clean the metro network
	
	Args:
		directory (str): the location in which to find the node and edge files
		file_prefix (TYPE): the prefix of the node and edge files
	
	Returns:
		networkx.DiGraph(): the metro network
	"""
	metro = mx.graph_from_txt(nodes_file_name = directory + '/' + file_prefix +'_nodes.txt', 
						   edges_file_name = directory + '/' + file_prefix +'_edges.txt', 
						   sep = '\t', 
						   nid = 'Station', 
						   eidfrom = 'From', 
						   eidto = 'To')

	print str(len(metro.nodes())) + ' nodes added to metro network'
	print str(len(metro.edges())) + ' edges added to metro network.'
	
	return metro

def clean_metro(metro):
	# Rename some attributes
	utility.rename_node_attribute(metro, old = 'Latitude', new = 'lat')
	utility.rename_node_attribute(metro, old = 'Longitude', new = 'lon')
	utility.rename_edge_attribute(metro, old = 'Time (s)', new = 'time_s')
	
	# delete extraneous attributes
	
	# utility.del_edge_attribute(metro, 'To')
	# utility.del_edge_attribute(metro, 'From')
	# utility.del_node_attribute(metro, 'Station')

	# compute time in minutes
	time_m = {(e[0], e[1]) : metro.edge[e[0]][e[1]]['time_s'] / 60 for e in metro.edges_iter()}

	# mark whether a given edge is a transfer edge to another metro line.  
	transfer = {key : 'transfer' for key in time_m if time_m[key] == 5.0}
	nx.set_edge_attributes(metro, 'transfer', transfer)

	nx.set_edge_attributes(metro, 'free_flow_time_m', time_m)
	nx.set_edge_attributes(metro, 'uniform_time_m', nx.get_edge_attributes(metro, 'free_flow_time_m'))

	# -----------------------------------------------------------
	# ZEYAD: please delete the below three lines when you update the metro data set. Replace them with whatever is necessary to appropriate set a distance attribute in kilometers. 
	dists = {(e[0], e[1]) : analysis.distance((metro.node[e[0]]['lat'],metro.node[e[0]]['lon']) , 
							  (metro.node[e[1]]['lat'],metro.node[e[1]]['lon'])) for e in metro.edges_iter()}
	nx.set_edge_attributes(metro, 'dist_km', dists)
	# -----------------------------------------------------------

	# assume metro has unlimited capacity
	nx.set_edge_attributes(metro, 'capacity', 100000000000000000000000)
	
	# don't need time_s anymore
	utility.del_edge_attribute(metro, 'time_s')
	
	return metro

def read_streets(directory, file_prefix):
	"""convenience function to quickly read in the street network 
	
	Args:
		directory (str): the directory in which to find the street network node and edge files
		file_prefix (str): the file prefix of the node and edge files
	
	Returns:
		networkx.DiGraph(): the street network. 
	"""
	streets = mx.graph_from_txt(nodes_file_name = directory + '/' + file_prefix +'_nodes.txt', 
						   edges_file_name = directory + '/' + file_prefix +'_edges.txt', 
						   sep = ' ', 
						   nid = 'id', 
						   eidfrom = 'source', 
						   eidto = 'target')

	print str(len(streets.nodes())) + ' nodes added to street network'
	print str(len(streets.edges())) + ' edges added to street network.'
	
	return streets

def clean_streets(streets):

	# Rename attributes
	utility.rename_edge_attribute(streets,'cost_time_m', 'free_flow_time_m')
	utility.rename_edge_attribute(streets, 'len_km', 'dist_km')

	utility.rename_node_attribute(streets, old = 'st_x', new = 'lon')
	utility.rename_node_attribute(streets, old = 'st_y', new = 'lat')

	# Delete some extraneous attributes
	utility.del_edge_attribute(streets, 'gid')
	utility.del_edge_attribute(streets, 'source')
	utility.del_edge_attribute(streets, 'target')

	# compute uniform time
	dists = nx.get_edge_attributes(streets, 'dist_km')
	total_dist = np.array(dists.values()).sum()
	total_time_free = np.array(nx.get_edge_attributes(streets, 'free_flow_time_m').values()).sum()
	uniform_speed = total_time_free/total_dist
	uniform_times = {key : dists[key] * uniform_speed  for key in dists}
	nx.set_edge_attributes(streets, 'uniform_time_m', uniform_times)

	# Delete edges with zero capacity. to impute capacity for them instead, uncomment block below. 
	for e in streets.copy().edges_iter():
		if streets.edge[e[0]][e[1]]['capacity'] == 0:
			streets.remove_edge(*e)

	# impute capacity -- just use the mean of all the other capacities. 
	# cap = [streets.edge[e[0]][e[1]]['capacity'] for e in streets.edges_iter()]
	# cap = np.array(cap)
	# mean = cap.mean()
	# for e in streets.edges_iter():
	# 	if streets.edge[e[0]][e[1]]['capacity'] == 0:
	# 		streets.edge[e[0]][e[1]]['capacity'] = mean
	
	return streets

def read_taz(directory, file_prefix):
	taz = mx.graph_from_txt(nodes_file_name = directory + '/' + file_prefix +'_nodes.txt',
	                             nid = 'id',
	                             sep = '\t')
	print str(len(taz.nodes())) + ' nodes added to TAZ connector network'
	return taz

def clean_taz(taz):
	return taz


if __name__ == "__main__":
    main()