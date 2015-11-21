
import numpy as np
import pandas as pd
import networkx as nx
import math as math
import os
import multiplex
from ast import literal_eval
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def d(pos1,pos2):
	"""Compute geographical distance between two points
	
	Args:
		pos1 (tuple): a tuple of the form (lat, lon)
		pos2 (tuple): a tuple of the form (lat, lon)
	
	Returns:
		float: the geographical distance between points, in kilometers
	"""
	LAT_DIST = 110766.95237186992 / 1000.0 # in km. See http://www.csgnetwork.com/degreelenllavcalc.html
	LON_DIST = 101274.42720366278 / 1000.0 # in km. See http://www.csgnetwork.com/degreelenllavcalc.html
	return math.sqrt((LON_DIST*(pos1[0]- pos2[0]))**2 + (LAT_DIST*(pos1[1] - pos2[1]))**2)

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
		nodes[col] = nodes[col].convert_objects(convert_numeric = True)
	
	N = nx.DiGraph()
	for n in range(len(nodes)):
		attr = {nid: nodes[nid][n]}
		# print nodes[nid][n]
		attr2 = {col: nodes[col][n] for col in list(nodes) if col != nid}
		attr.update(attr2)
		
		N.add_node(n = nodes[nid][n], attr_dict = attr)

	if edges_file_name is not None: 
		edges = pd.read_table(edges_file_name, sep = sep, index_col=False)
		for col in edges:
			edges[col] = edges[col].convert_objects(convert_numeric = True)

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

def find_nearest(n, N1, N2):
	"""for a fixed node n in network N1, find th nearest node in network N2
	
	Args:
		n (str): the node from which to start
		N1 (networkx.DiGraph()): the network in which n lies. 
		N2 (networkx.DiGraph()): the network in which to find the node nearest to N1.  
	
	Returns:
		str, int: the nearest neighbor in N2 and the distance to that neighbor
	"""

	dists = {m: d( (N1.node[n]['lon'], N1.node[n]['lat']), (N2.node[m]['lon'], N2.node[m]['lat']) ) for m in N2}
	nearest = min(dists, key=dists.get)
	nearest_dist = dists[nearest]
	return nearest, nearest_dist

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

def read_metro(directory, file_prefix):
	"""convenience function to quickly read in and clean the metro network
	
	Args:
		directory (str): the location in which to find the node and edge files
		file_prefix (TYPE): the prefix of the node and edge files
	
	Returns:
		networkx.DiGraph(): the metro network
	"""
	metro = graph_from_txt(nodes_file_name = directory + '/' + file_prefix +'_nodes.txt', 
						   edges_file_name = directory + '/' + file_prefix +'_edges.txt', 
						   sep = '\t', 
						   nid = 'Station', 
						   eidfrom = 'From', 
						   eidto = 'To')

	rename_node_attribute(metro, old = 'Latitude', new = 'lat')
	rename_node_attribute(metro, old = 'Longitude', new = 'lon')
	rename_edge_attribute(metro, old = 'Time (s)', new = 'time_s')
	
	dists = {(e[0], e[1]) : d((metro.node[e[0]]['lat'],metro.node[e[0]]['lon']) , 
							  (metro.node[e[1]]['lat'],metro.node[e[1]]['lon'])) for e in metro.edges_iter()}

	nx.set_edge_attributes(metro, 'dist_km', dists)
	nx.set_edge_attributes(metro, 'capacity', 100000000000000000000000)

	time_m = {(e[0], e[1]) : metro.edge[e[0]][e[1]]['time_s'] / 60 for e in metro.edges_iter()}
	nx.set_edge_attributes(metro, 'free_flow_time_m', time_m)
	nx.set_edge_attributes(metro, 'cost_time_m', time_m)
	nx.set_edge_attributes(metro, 'weight', time_m)

	print str(len(metro.nodes())) + ' nodes added to metro network.'
	print str(len(metro.edges())) + ' edges added to metro network.'
	nx.set_node_attributes(metro, 'layer', 'metro')
	nx.set_edge_attributes(metro, 'layer', 'metro')
	metro.mode = 'metro'


	return metro

def read_streets(directory, file_prefix):
	"""convenience function to quickly read in the street network 
	
	Args:
		directory (str): the directory in which to find the street network node and edge files
		file_prefix (str): the file prefix of the node and edge files
	
	Returns:
		networkx.DiGraph(): the street network. 
	"""
	streets = graph_from_txt(nodes_file_name = directory + '/' + file_prefix +'_nodes.txt', 
						   edges_file_name = directory + '/' + file_prefix +'_edges.txt', 
						   sep = ' ', 
						   nid = 'id', 
						   eidfrom = 'source', 
						   eidto = 'target')
	print 'constructed graph'

	nx.set_edge_attributes(streets, 'weight', nx.get_edge_attributes(streets, 'cost_time_m'))
	nx.set_edge_attributes(streets, 'free_flow_time_m', nx.get_edge_attributes(streets, 'cost_time_m'))
	nx.set_edge_attributes(streets, 'dist_km', nx.get_edge_attributes(streets, 'len_km'))
	
	rename_node_attribute(streets, old = 'st_x', new = 'lon')
	rename_node_attribute(streets, old = 'st_y', new = 'lat')

	print str(len(streets.nodes())) + ' nodes added to street network'
	print str(len(streets.edges())) + ' edges added to street network.'
	nx.set_node_attributes(streets, 'layer', 'streets')
	nx.set_edge_attributes(streets, 'layer', 'streets')

	streets.mode = 'streets'

	cap = [streets.edge[e[0]][e[1]]['capacity'] for e in streets.edges_iter()]
	cap = np.array(cap)
	mean = cap.mean()

	for e in streets.edges_iter():
		if streets.edge[e[0]][e[1]]['capacity'] == 0:
			streets.edge[e[0]][e[1]]['capacity'] = mean

	return streets

def remove_flow_through(N, add):
	'''
	NOT USED
	'''
	print 'You can use me, but it\'s probably a bad idea right now'
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
	NOT USED
	'''
	print 'You can use me, but it\'s probably a bad idea right now'
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

	multi = multiplex.multiplex()
	multi.add_graph(G)

	return multi

def distance_matrix(x0, y0, x1, y1):
	obs = np.vstack((x0, y0)).T
	interp = np.vstack((x1, y1)).T

	# Make a distance matrix between pairwise observations

	d0 = np.subtract.outer(obs[:,0], interp[:,0])
	d1 = np.subtract.outer(obs[:,1], interp[:,1])

	return np.hypot(d0, d1)

def simple_idw(x, y, z, xi, yi, threshhold):
	dist = distance_matrix(x,y, xi,yi)

	# In IDW, weights are 1 / distance
	weights = 1.0 / dist**.5

	# Make weights sum to one
	weights /= weights.sum(axis=0)

	# Multiply the weights for each interpolated point by all observed Z-values
	zi = np.dot(weights.T, z)
	# gap = zi[dist.min(axis = 0) > threshhold].max()
	# zi[dist.min(axis = 0) > threshhold] = 0
	# zi = zi - gap
	# zi[zi < 0] = 0
	return zi

def plot(x,y,z,grid):
	# plt.figure(figsize = (15,15), dpi = 500)
	plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()), cmap=cm.Blues)
	plt.hold(True)
	# plt.colorbar()

def idw_smoothed_plot(layer, measure): # broken after removing pos attributes
	N = multi.layers_as_subgraph([layer])

	x = np.array([N.node[n]['pos'][1] for n in N.node])
	y = np.array([- N.node[n]['pos'][0] for n in N.node])
	z = np.array([float(N.node[n][measure]) for n in N.node])

	mx, my = 100, 100
	xi = np.linspace(x.min(), x.max(), mx)
	yi = np.linspace(y.min(), y.max(), my)

	xi, yi = np.meshgrid(xi, yi)
	xi, yi = xi.flatten(), yi.flatten()

	grid1 = simple_idw(x,y,z,xi,yi, threshhold = .2)
	grid1 = grid1.reshape((my, mx))

	plot(x,y,z,grid1)

def read_multi(nodes_file_name = '2. multiplex/multiplex_nodes.txt', edges_file_name = '2. multiplex/multiplex_edges.txt', sep = '\t', nid = 'id', eidfrom = 'source', eidto = 'target'):
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

def check_directory(directory):
	"""check for the existence of a directory and add if it's not there. 
	
	Args:
		directory (str): the directory to check
	
	Returns:
		None
	"""
	if not os.path.exists(directory):
		os.makedirs(directory)

def gini_coeff(x):
	'''
	compute the gini coefficient from an array of floats
	
	From http://www.ellipsix.net/blog/2012/11/the-gini-coefficient-for-distribution-inequality.html
	
	Args:
		x (np.array()): an array of floats
	'''
	# requires all values in x to be zero or positive numbers,
	# otherwise results are undefined
	n = len(x)
	s = x.sum()
	r = np.argsort(np.argsort(-x)) # calculates zero-based ranks
	return 1 - (2.0 * (r*x).sum() + s)/(n*s)

def spatial_plot(G, attr, ax, title = 'plot!'):
	import scipy.ndimage as ndimage

	cols = ['layer', 'lon', 'lat', attr]
	df = nodes_2_df(G, cols)

	n = 2000
	grid_x, grid_y = np.mgrid[df.lon.min():df.lon.max():n * 1j, 
					  df.lat.min():df.lat.max():n * 1j]
	zj = np.zeros(grid_x.shape)
	
	lonmax = df.lon.max()
	lonmin = df.lon.min()
	latmax = df.lat.max()
	latmin = df.lat.min()

	for i in range(len(df)):
		x = int((df.loc[i]['lon'] - lonmin) / (lonmax - lonmin)*n) - 1
		y = int((df.loc[i]['lat'] - latmin) / (latmax - latmin)*n) - 1 
		zj[x][y] += df.loc[i][attr]
		
	zi = ndimage.gaussian_filter(zj, sigma=12.0, order=0)
	ax.contourf(grid_x, grid_y, zi, 100, linewidths=0.1, cmap=plt.get_cmap('afmhot'), alpha = 1, vmax = 1./1. * zi.max())
	
	G.position = {n : (G.node[n]['lon'], G.node[n]['lat']) for n in G}
	nx.draw(G, G.position,
			edge_color = 'white', 
			edge_size = 0.01,
			node_color = 'white',
			node_size = 0,
			alpha = .15,
			with_labels = False,
			arrows = False)
	plt.title(title)


