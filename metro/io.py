import numpy as np
import pandas as pd
import networkx as nx
import multiplex as mx
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



def read_multi(nodes_file_name = '2_multiplex/multiplex_nodes.txt', edges_file_name = '2_multiplex/multiplex_edges.txt', sep = '\t', nid = 'id', eidfrom = 'source', eidto = 'target'):
	"""A convenience function for easily reading in pipeline's multiplex. 
	
	Returns:
		multiplex.multiplex(): the pipeline's multiplex from make_multiplex.py
	"""
	multi = multiplex_from_txt(nodes_file_name = nodes_file_name,
									   edges_file_name = edges_file_name,
									   sep = sep,
									   nid = nid,
									   eidfrom = eidfrom,
									   eidto = eidto)
	return multi

def multiplex_from_txt(**kwargs):
	"""Convenience function to quickly read a multiplex object from a pair of node and edge files. 
	
	Args:
		**kwargs: kwargs passed down to utility.graph_from_txt()
	
	Returns:
		multiplex.multiplex(): a multiplex object with appropriate attributes, etc. 
	"""
	G = graph_from_txt(**kwargs)
	

	cap = {(e[0], e[1]) : float(G.edge[e[0]][e[1]]['capacity']) for e in G.edges_iter()}
	nx.set_edge_attributes(G, 'capacity', cap)

	multi = mx.multiplex()
	multi.add_graph(G)

	return multi

def multiplex_to_txt(multi, directory, file_name):
		'''
		save the multiplex to a pair of .txt documents for later processing. 
		
		args: 
			directory -- (str) the directory in which to save the file_name
			file_name -- (str) the file prefix, will have '_nodes.txt' and _edges.txt' suffixed. 
		'''
		write_nx_nodes(multi.G, directory, file_name + '_nodes.txt')
		write_nx_edges(multi.G, directory, file_name + '_edges.txt')
