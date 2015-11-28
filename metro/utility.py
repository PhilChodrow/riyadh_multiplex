import networkx as nx
import numpy as np
import igraph as ig
import pandas as pd
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

def edges_2_df(N, col_names = ['layer']):
    if col_names == None:
        col_names = set([])
        for e in N.edges_iter():
            col_names = col_names.union(N.edge[e[0]][e[1]].keys())
    d = {}
    for col in col_names:
        attr = nx.get_edge_attributes(N, col)
        attr = [attr[e] if e in attr.keys() else None for e in N.edges_iter()]
        d[col] = attr
    return pd.DataFrame(d)

def nodes_2_df(N, col_names = ['layer', 'lon', 'lat']):
	"""
	Convert the nodes of a networkx.DiGraph() object into a pandas.DataFrame

	Args:
		N (networkx.DiGraph()): the graph to convert

	Returns: 
		a pandas.DataFrame with node attributes as columns
	"""
	if col_names == None:
		col_names = set([])
		for n in N.node:
			col_names = col_names.union(N.node[n].keys())

	d = {}
	for col in col_names: 
		attr = nx.get_node_attributes(N, col)
		attr = [attr[n] if n in attr.keys() else None for n in N.node]
		d[col] = attr
	
	return pd.DataFrame(d)

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
