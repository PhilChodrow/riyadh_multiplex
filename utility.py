import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import math as math
import os
import multiplex
from ast import literal_eval

def d(pos1,pos2):
	LAT_DIST = 110766.95237186992 / 1000.0 # in km. See http://www.csgnetwork.com/degreelenllavcalc.html
	LON_DIST = 101274.42720366278 / 1000.0 # in km. See http://www.csgnetwork.com/degreelenllavcalc.html
	return math.sqrt((LAT_DIST*(pos1[0]- pos2[0]))**2 + (LON_DIST*(pos1[1] - pos2[1]))**2)

def graph_from_txt(nodes_file_name, edges_file_name, sep, nid, eidfrom, eidto):
	'''
		Docs
	'''

	edges = pd.read_table(edges_file_name, sep = sep, index_col=False)
	nodes = pd.read_table(nodes_file_name, sep = sep, index_col=False)

	N = nx.DiGraph()

	for n in range(len(nodes)):
		attr = {nid: nodes[nid][n]}
		attr2 = {col: nodes[col][n] for col in list(nodes)}
		attr.update(attr2)
		N.add_node(nodes[nid][n], attr)

	for e in range(len(edges)):
		attr = {eidfrom : edges[eidfrom][e], eidto : edges[eidto][e]}
		attr2 = {col: edges[col][e] for col in list(edges)}
		attr.update(attr2)
		N.add_edge(edges[eidfrom][e], edges[eidto][e], attr)

	return N

def write_nx_nodes(N, directory, file_name):
	'''
	N: a networkx graph.
	file_name: a string with the name of the file to be saved. 
	OUT: No return value. Writes a file with name file_name of N's node ids and attributes, tab-separated. 
	'''
	col_names = set([])
	for n in N.node:
		col_names = col_names.union(N.node[n].keys())

	if not os.path.exists(directory):
		os.makedirs(directory)

	nodes = open(directory + '/' + file_name, 'w')

	nodes.write('id')
	for col in col_names:
		if col == 'id':
			pass
		else:
			nodes.write('\t' + col)

	for n in N.node:
		nodes.write('\n'  + str(n))
		for col in col_names:
			if col == 'id':
				pass
			else:
				try:
					nodes.write('\t' + str(N.node[n][col]))
				except KeyError:
					nodes.write('\tNone')

	nodes.close()

def write_nx_edges(N, directory, file_name):
	'''
	N: a networkx graph.
	file_name: a string with the name of the file to be saved. 
	OUT: No return value. Writes a file with name file_name of N's node ids and attributes, tab-separated. 
	'''
	col_names = set([])
	for e in N.edges_iter():
		col_names = col_names.union(N.edge[e[0]][e[1]].keys())

	if not os.path.exists(directory):
		os.makedirs(directory)

	edges = open(directory + '/' + file_name, 'w')

	edges.write('source' + '\t' + 'target')
	for col in col_names:
		if col == 'source' or col == 'target':
			pass
		else: 
			edges.write('\t' + col)

	for e in N.edges_iter():
		edges.write('\n' + str(e[0]) + '\t' + str(e[1]))
		for col in col_names:
			if col == 'source' or col == 'target':
				pass
			else:
				try:
					edges.write('\t' + str(N.edge[e[0]][e[1]][col]))
				except KeyError:
					edges.write('\tNone')

	edges.close()


def rename_node_attribute(N, old, new):
	nx.set_node_attributes(N, new, nx.get_node_attributes(N, old))

	for n in N.node:
		del N.node[n][old]

def rename_edge_attribute(N, old, new):
	nx.set_edge_attributes(N, new, nx.get_edge_attributes(N, old))
	for e in N.edges_iter():
		del N.edge[e[0]][e[1]][old]


def find_nearest(n, N1, N2):
	'''
	n, a node in network N1
	N1, a networkx graph whose nodes have lat and lon attributes
	N2, a networkx graph whose nodes have lat and lon attributes
	RETURNS: a 2-tuple containing: the node in network N2 spatially nearest to n,the distance from n to that node in km.
	'''

	dists = {m: d(N1.node[n]['pos'], N2.node[m]['pos']) for m in N2}
	nearest = min(dists, key=dists.get)
	nearest_dist = dists[nearest]
	return nearest, nearest_dist

def spatial_multiplex_join(N1, N2, TRANSFER_SPEED):
	'''
	N1, a networkx graph with a graph mode attribute and whose nodes have lat, lon, and mode attributes equal to the graph mode attribute.  
	N2, a networkx graph with a graph mode attribute and whose nodes have lat, lon, and mode attributes equal to the graph mode attribute. Mode attributes must be different than N1.
	TRANSFER_SPEED, the speed in meters per second of transfer from N1 to N2. E.g. walking time from parking lot to subway station.
	RETURNS: a networkx graph containing each of N1 and N2, joined by edges between the nodes of N1 and the nearest nodes of N2. These edges have the edge attribute mode = 'transfer', dist = length of transfer, and weight = dist/TRANSFER_SPEED. 
	'''
	multiplex = nx.disjoint_union(N1, N2)

	N1_nodes = [n for n in multiplex.node if multiplex.node[n]['mode'] == N1.mode]
	N2_nodes = [n for n in multiplex.node if multiplex.node[n]['mode'] == N2.mode]

	N1_sub = multiplex.subgraph(N1_nodes)
	N2_sub = multiplex.subgraph(N2_nodes)

	for n in N1_sub.node:
		nearest, nearest_dist = find_nearest(n, N1_sub, N2_sub)
		multiplex.add_edge(n, nearest, {'dist_km' : nearest_dist, 'mode' : 'transfer', 'cost_time_m' : nearest_dist / TRANSFER_SPEED, 'weight' : nearest_dist / TRANSFER_SPEED})
		multiplex.add_edge(nearest, n, {'dist_km' : nearest_dist, 'mode' : 'transfer', 'cost_time_m' : nearest_dist / TRANSFER_SPEED, 'weight' : nearest_dist / TRANSFER_SPEED})
		
		print 'Added transfer between ' + str(n) + ' and ' + str(nearest) + ' of length ' + str(round(nearest_dist, 2)) + 'km.'

	return multiplex

def read_metro(directory, file_prefix):
	metro = graph_from_txt(nodes_file_name = directory + '/' + file_prefix +'_nodes.txt', 
	                       edges_file_name = directory + '/' + file_prefix +'_edges.txt', 
	                       sep = '\t', 
	                       nid = 'Station', 
	                       eidfrom = 'From', 
	                       eidto = 'To')

	rename_node_attribute(metro, old = 'Latitude', new = 'lat')
	rename_node_attribute(metro, old = 'Longitude', new = 'lon')
	rename_edge_attribute(metro, old = 'Time (s)', new = 'time_s')
	
	pos = {n : (metro.node[n]['lat'], metro.node[n]['lon']) for n in metro}
	nx.set_node_attributes(metro, 'pos', pos)

	dists = {(e[0], e[1]) : d(metro.node[e[0]]['pos'], metro.node[e[1]]['pos']) for e in metro.edges_iter()}
	nx.set_edge_attributes(metro, 'dist_km', dists)


	time_m = {(e[0], e[1]) : metro.edge[e[0]][e[1]]['time_s'] / 60 for e in metro.edges_iter()}
	nx.set_edge_attributes(metro, 'cost_time_m', time_m)
	nx.set_edge_attributes(metro, 'weight', time_m)

	print str(len(metro.nodes())) + ' nodes added to metro network.'
	print str(len(metro.edges())) + ' edges added to metro network.'
	nx.set_node_attributes(metro, 'layer', 'metro')
	nx.set_edge_attributes(metro, 'layer', 'metro')
	metro.mode = 'metro'


	return metro

def read_streets(directory, file_prefix):

	streets = graph_from_txt(nodes_file_name = directory + '/' + file_prefix +'_nodes.txt', 
	                       edges_file_name = directory + '/' + file_prefix +'_edges.txt', 
	                       sep = ' ', 
	                       nid = 'id', 
	                       eidfrom = 'source', 
	                       eidto = 'target')
	print 'constructed graph'


	nx.set_edge_attributes(streets, 'weight', nx.get_edge_attributes(streets, 'cost_time_m'))
	nx.set_edge_attributes(streets, 'dist_km', nx.get_edge_attributes(streets, 'len_km'))
	pos = {n : (streets.node[n]['st_y'], streets.node[n]['st_x']) for n in streets}	
	nx.set_node_attributes(streets, 'pos', pos)

	print str(len(streets.nodes())) + ' nodes added to street network'
	print str(len(streets.edges())) + ' edges added to street network.'
	nx.set_node_attributes(streets, 'layer', 'streets')
	nx.set_edge_attributes(streets, 'layer', 'streets')

	streets.mode = 'streets'

	return streets

def remove_flow_through(N, add):
	'''
	Removes flow-through nodes in a directed graph. We should probably implement some checks for this to ensure that all the additive measures work out right. 
	'''
	edges_before = len(N.edges())
	nodes_before = len(N.nodes())
	for n in N.nodes():
		pred = N.predecessors(n)
		succ = N.successors(n)
		if len(pred) == len(succ) == 2 and set(pred) == set(succ):
			attr = {}
			for measure in add: 
				total =  N[pred[0]][n][measure] + N[n][pred[1]][measure]
				attr.update({measure : total})
			N.add_edge(pred[0], pred[1], attr)
			N.add_edge(pred[1], pred[0], attr)
			N.remove_node(n)
	edges_after = len(N.edges())
	nodes_after = len(N.nodes())
	print 'Removed ' + str(nodes_before - nodes_after) + ' nodes and ' + str(edges_before - edges_after) + ' edges.' 
	nx.convert_node_labels_to_integers(N)
	return N

def remove_flow_through_2(N, add):
	'''
	Removes flow-through nodes in a directed graph. We should probably implement some checks for this to ensure that all the additive measures work out right. 
	'''
	edges_before = len(N.edges())
	nodes_before = len(N.nodes())
	for n in N.nodes():
		pred = N.predecessors(n)
		succ = N.successors(n)
		if len(pred) == len(succ) == 1 and set(pred) != set(succ):
			attr = {}
			for measure in add: 
				total =  N[pred[0]][n][measure] + N[n][succ[0]][measure]
				attr.update({measure : total})
			N.add_edge(pred[0], succ[0], attr)
			N.remove_node(n)
	edges_after = len(N.edges())
	nodes_after = len(N.nodes())
	print 'Removed ' + str(nodes_before - nodes_after) + ' nodes and ' + str(edges_before - edges_after) + ' edges.' 
	nx.convert_node_labels_to_integers(N)
	return N

def nx_2_igraph(graph):
	
	ig_graph = ig.Graph()
	ig_graph = ig_graph.as_directed(mutual = False)

	nodes = graph.node.keys()
	edges = graph.edges_iter()

	for n in nodes:
		attr = graph.node[n]
		ig_graph.add_vertex(str(n), **attr)

	for e in edges:
		attr = graph[e[0]][e[1]]
		attr.pop('source', None)
		attr.pop('target', None)
		ig_graph.add_edge(str(e[0]), str(e[1]), **attr)
		
	return ig_graph

def igraph_2_nx(ig_graph):
	print 'Converting to networkx format'
	nx_graph = nx.DiGraph()
	for v in ig_graph.vs:
		attr = v.attributes()
		nx_graph.add_node(v.index, attr)

	for e in ig_graph.es:
		attr = e.attributes()
		nx_graph.add_edge(e.source, e.target, attr)

	return nx_graph

def multiplex_from_txt(**kwargs):
	G = graph_from_txt(**kwargs)
	pos = {n : (literal_eval(G.node[n]['pos'])) for n in G}

	nx.set_node_attributes(G, 'pos', pos)

	multi = multiplex.multiplex()
	multi.add_graph(G)

	return multi
