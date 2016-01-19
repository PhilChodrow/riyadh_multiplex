import networkx as nx
import numpy as np
import igraph as ig
import pandas as pd
import os

def check_directory(directory):
	"""check for the existence of a directory and add if it's not there. 
	
	Args:
		directory (str): the directory to check
	
	Returns:
		None
	"""
	if not os.path.exists(directory):
		os.makedirs(directory)


def graph_from_txt(nodes_file_name = None, edges_file_name = None, sep = '\t', nid = None, eidfrom = None, eidto = None):
	"""Summary
	
	Args:
		nodes_file_name (str, optional): the file in which to find node ids and attributes
		edges_file_name (str, optional): the file in which to find edge ids and attributes
		sep (str, optional): the separator character used in the node and edge files
		nid (str, optional): the hashable attribute used to identify nodes
		eidfrom (str, optional): the hashable attribute used to identify sources of edges (must match nid)
		eidto (str, optional): the hashable attribute used to identify targets of edges (must match nid)
	
	Returns:
		a networkx.DiGraph() object
	"""
	nodes = pd.read_table(nodes_file_name, sep = sep, index_col=False)
	for col in nodes:
		nodes[col] = pd.to_numeric(nodes[col], errors = 'ignore')
	
	N = nx.DiGraph()
	for n in range(len(nodes)):
		attr = {nid: nodes[nid][n]}
		attr2 = {col: nodes[col][n] for col in list(nodes) if col != nid}
		attr.update(attr2)
		
		N.add_node(n = nodes[nid][n], attr_dict = attr)

	if edges_file_name is not None: 
		edges = pd.read_table(edges_file_name, sep = sep, index_col=False)
		for col in edges:
			edges[col] = pd.to_numeric(edges[col], errors = 'ignore')

		for e in range(len(edges)):
			attr = {eidfrom : edges[eidfrom][e], eidto : edges[eidto][e]}
			attr2 = {col: edges[col][e] for col in list(edges)}
			attr.update(attr2)
			N.add_edge(edges[eidfrom][e], edges[eidto][e], attr)

	return N

def write_nx_nodes(N, directory, file_name):
	'''
	Write the nodes of a networkx.DiGraph() object to a .txt file
	Args:
		N (networkx.DiGraph()): the graph to write
		directory (str): the directory in which to save the file
		file_name (str): the name under which to save the file
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
	"""Write the edges of a networkx.DiGraph() object to a .txt file.
	
	Args:
		N (networkx.DiGraph()): the networkx.DiGraph() object to write
		directory (str): The directory in which to save the file
		file_name (str): the name of the file
	
	Returns:
		None
	"""
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




def nx_2_igraph(graph):
	"""convert a networkx.DiGraph() object into an igraph.Graph() object. 
	
	Args:
		graph (networkx.DiGraph()): the network to convert
	
	Returns:
		igraph.Graph(): the converted network in igraph format
	"""
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
	"""convert an igraph.Graph() object into a networkx.DiGraph() object
	
	Args:
		ig_graph (igraph.Graph()): the graph to convert
	
	Returns:
		networkx.DiGraph(): the converted graph
	"""
	print 'Converting to networkx format'
	nx_graph = nx.DiGraph()
	for v in ig_graph.vs:
		attr = v.attributes()
		nx_graph.add_node(v.index, attr)

	for e in ig_graph.es:
		attr = e.attributes()
		nx_graph.add_edge(e.source, e.target, attr)

	return nx_graph

def edges_2_df(G, attrs):
	attrdict = {attr : [G.edge[e[0]][e[1]][attr] or None for e in G.edges_iter()] for attr in attrs}
	return pd.DataFrame(attrdict)

def nodes_2_df(G, attrs):
    attrs = attrs
    attrdict = {attr : [G.node[n][attr] or None for n in G.node] for attr in attrs}
    return pd.DataFrame(attrdict)

def rename_node_attribute(N, old, new):
	""" rename a node attribute in a networkx.DiGraph() object. 
	
	Args:
		N (networkx.DiGraph()): the networkx.DiGraph() containing an attribute to rename
		old (str): the old name of the attribute
		new (str): the new name of the attribute
	
	Returns:
		None
	"""
	nx.set_node_attributes(N, new, nx.get_node_attributes(N, old))

	for n in N.node:
		del N.node[n][old]

def rename_edge_attribute(N, old, new):
	"""rename an edge attribute in a networkx.DiGraph() object
	
	Args:
		N (networkx.DiGraph()): the networkx.DiGraph() object containing the edge attribute to rename
		old (str): the old name of the edge attribute
		new (str): the new name of the edge attribute
	
	Returns:
		None
	"""
	nx.set_edge_attributes(N, new, nx.get_edge_attributes(N, old))
	for e in N.edges_iter():
		del N.edge[e[0]][e[1]][old]

def spatial_multiplex_join(N1, N2, TRANSFER_SPEED):
	"""join nodes in N1 to their nearest neighbors in N2
	
	Args:
		N1 (networkx.DiGraph()): the network from which to search for nearest neighbors
		N2 (networkx.DiGraph()): the network in which to search for nearest neighbors
		TRANSFER_SPEED (float): the speed at which transfers are assumed traversed, in km/m
	
	Returns:
		networkx.DiGraph(): a graph including N1, N2, and the new links between them. 
	"""
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

def del_edge_attribute(N, a):
	for e in N.edges_iter():
		attr = N[e[0]][e[1]]
		if a in attr:
			del attr[a]

def del_node_attribute(N,a):
	for n in N.nodes_iter():
		attr = N.node[n]
		if a in attr:
			del attr[a]

def scale_edge_attribute_igraph(g, layer, attr, beta = 1):
    original = g.es.select(lambda v: v['layer'] == layer)[attr]
    scaled = [beta * v for v in original]
    g.es.select(lambda v: v['layer'] == layer)[attr] = scaled
