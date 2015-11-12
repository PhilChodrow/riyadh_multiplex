import networkx as nx
from numpy import sqrt
from heapq import heappush, heappop
from time import clock
from sys import stdout
import utility
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate
import numpy as np
import matplotlib.cm as cm
import random

class multiplex:
	'''

	multiplex class is a thin wrapper for a the networkx.DiGraph() object, for 
	cases in which that object is composed of distinct layers. 

	attributes: 
		self.layers -- (list) list of strings
		self.G -- a networkx.DiGraph object, all of whose nodes and edges have a 
		'layer' attribute. Many methods also assume that the nodes have a 'pos' 
		attribute of the form (lat, lon) 

	'''
	def __init__(self):
		self.layers = []
		self.G = nx.DiGraph()

	def add_layers(self, layer_dict):
		'''
		Add layers from a dict of networkx graphs.
		
		args:
		    layer_dict: (dict) a dict of layer names and graphs, e.g. {'metro' : metro, 'street' : street}
		'''
		for layer in layer_dict:
			if layer in self.layers: 
				print "ERROR: The layer" + layer + "is already defined in the multiplex, did not overwrite"
			
			else:
				self.layers.append(layer)
				nx.set_node_attributes(layer_dict[layer], 'layer', layer)
				nx.set_edge_attributes(layer_dict[layer], 'layer', layer)
				self.G = nx.disjoint_union(self.G, layer_dict[layer])
				self.label_nodes()
				

	def add_epsilon(self, weight, epsilon):
		'''
		Add an infinitesimal positive number to a numeric attribute of self.G.edges()

		args:
			weight  -- (str) the numeric attribute to increment
			epsilon -- (float) the amount by which to increment, typically very small.  
		
		Use case: some edges have zero cost_time_m, but the betweenness
		algorithm used requires nonzero weights. 

		'''
		d = {e : float(self.G.edge[e[0]][e[1]][weight] or 0) + epsilon for e in self.G.edges_iter()}
		nx.set_edge_attributes(self.G, weight, d)

	def label_nodes(self):
		'''
		Relabel the nodes in the format layer_int, e.g. 'streets_214'. 
		'''
		nx.convert_node_labels_to_integers(self.G)
		new_labels = {n : self.G.node[n]['layer'] + '_' + str(n) for n in self.G.node} 
		nx.relabel_nodes(self.G, mapping = new_labels, copy = False)

	def add_graph(self, H):
		'''
		Add a single graph to self.G and update layers. 

		args:
			H -- a networkx.DiGraph() object whose nodes and edges all have a 'layer' attribute. 
		'''
		self.G = nx.disjoint_union(self.G, H)
		self.update_layers()
		self.label_nodes()

	def get_layers(self):
		'''Return the current layer list.'''
		return self.layers
		
	def remove_layer(self, layer):
		'''
		Delete a layer from the multiplex. All nodes and edges in that layer are
		removed.

		args: 
			layer -- (str) the name of an element of self.layers
		 
		'''
		if layer not in self.layers:
			print "Sorry, " + layer + ' is not current in the multiplex.'
		else:
			self.layers.remove(layer)
			self.G.remove_nodes_from([n for n,attrdict in self.G.node.items() if attrdict['layer'] == layer])

	def check_layer(self, layer_name):
		'''
		Check to see if G contains layer_name as a layer.

		args: 
			layer_name -- (str) the layer to check
		'''
		return layer_name in self.layers 
	
	def spatial_join(self, layer1, layer2, transfer_speed, base_cost, both = True):
		'''
		Add edges to between ALL nodes of layer1 and the nodes of layer2 spatially nearest to the nodes of layer1. 
		New edges are labelled 'layer1_layer2_T' and 'layer1_layer2_T' is added to self.layers.  
		Requires that each node in each G have a 'pos' tuple of format (latitude, longitude). 
		
		args: 
			layer1         -- (str) base layer, all nodes joined to one node in layer2
			layer2         -- (str) layer to which layer1 will be joined
			transfer_speed -- (float) assumed speed at which transfer distance can be traversed, e.g. walking speed from street to metro. 
			base_cost      -- (float) base cost associated with transfer, e.g. mean time spent waiting for metro.
			both           -- (bool) if true, transfer is bidirectional.   		

		Example: spatial_join(layer1 = 'metro', layer2 = 'street')
		'''	
		transfer_layer_name = layer1 + '--' + layer2
		self.layers.append(transfer_layer_name)

		layer1_copy = self.layers_as_subgraph([layer1])	
		layer2_copy = self.layers_as_subgraph([layer2])

		for n in layer1_copy.node:
			nearest, nearest_dist = utility.find_nearest(n, layer1_copy, layer2_copy)
			self.G.add_edge(n, nearest, 
							layer = transfer_layer_name,
							weight = 0,
							dist_km = nearest_dist, 
							cost_time_m = nearest_dist / transfer_speed + base_cost)
			
			bidirectional = ""
			if both: 
				self.G.add_edge(nearest, n, 
								layer = transfer_layer_name,
								weight = 0,
								dist_km = nearest_dist, 
								cost_time_m = nearest_dist / transfer_speed + base_cost) # assumes bidirectional
				bidirectional = "bidirectional "

			print 'Added ' + bidirectional + 'transfer between ' + str(n) + ' in ' + layer1 + ' and ' + str(nearest) + ' in ' + layer2 + ' of length ' + str(round(nearest_dist, 2)) + 'km.'

	def layers_as_subgraph(self, layers):
		'''
		return a subset of the layers of self.G as a networkx.DiGraph() object. 
		args: 
			layers -- (list) a list of layers to return
		'''
		return self.G.subgraph([n for n,attrdict in self.G.node.items() if attrdict['layer'] in layers])

	def sub_multiplex(self, sublayers):
		'''
		return a subset of the layers of self.G as a multiplex() object. 
		
		args:
			sublayers -- (list) a list of layers, all of which must be elements of self.layers
		
		'''
		sub_multiplex = multiplex()        
		sublayer_dict = {layer : self.zlayer_as_subgraph(layer) for layer in sublayers}
		sub_multiplex.add_layers(sublayer_dict)
		return subMultiplex

	def as_graph(self):
		'''
		Return self.multiplex as a networkx.DiGraph() object. 
		'''
		return self.G

	def update_node_attributes(self, attr):
		'''
		set the attributes of self.G.node

		args:
			attr -- (dict) a dict with nodenames as keys. Values are attribute dicts. 
		'''

		for n in attr:
			for att in attr[n]: self.G.node[n][att] = attr[n][att]

	def update_edge_attributes(self, attr):
		'''
		set the attributes of self.G.node

		args:
			attr -- (dict) a dict with edgenames (or node 2-tuples) as keys. Values are attribute dicts. 
		'''
		for e in attr:
			for att in attr[e]: self.G.edge[e[0]][e[1]] = attr[e][att]

	def summary(self):
		'''
		view a summary of self, printed to the terminal
		'''
		layers = {layer: (len([n for n,attrdict in self.G.node.items() if attrdict['layer'] == layer]), 
						  len([(u,v,d) for u,v,d in self.G.edges(data=True) if d['layer'] == layer])) for layer in self.layers} 
		
		print "Layer \t N \t E "
		for layer in layers:
			print layer, "\t", layers[layer][0], "\t", layers[layer][1]  

	def to_txt(self, directory, file_name):
		'''
		save the multiplex to a pair of .txt documents for later processing. 
		
		args: 
			directory -- (str) the directory in which to save the file_name
			file_name -- (str) the file prefix, will have '_nodes.txt' and _edges.txt' suffixed. 
		'''
		utility.write_nx_nodes(self.G, directory, file_name + '_nodes.txt')
		utility.write_nx_edges(self.G, directory, file_name + '_edges.txt')

	def update_layers(self):
		'''
		Check that layers includes all values of 'layer' attributes in self.G
		'''
		new_layers = set([attrdict['layer'] for n, attrdict in self.G.node.items()])
		new_layers.update(set([d['layer'] for u,v,d in self.G.edges(data=True)]))
		self.layers = list(new_layers)

	def igraph_betweenness_centrality(self, layers = None, weight = None, attrname = 'bc'):
		'''
		compute the (weighted) betweenness centrality of one or more layers and save to self.G.node attributes. 

		args: 
			thru_layers -- the layers on which to calculate betweenness. 
			source_layers -- the layers to use as sources in betweenness calculation.
			target_layers -- the layers to use as targets in the betweenness calculation.  

		'''
		print 'Computing betweenness centrality -- this could take a while.' 

		g = utility.nx_2_igraph(self.layers_as_subgraph(layers))

		bc = g.betweenness(directed = True,
						  cutoff = 300,
						  weights = weight)
		print 'betweenness calculated'
		d = dict(zip(g.vs['name'], bc))
		d = {key:d[key] for key in d.keys()}

		nx.set_node_attributes(self.G, attrname, d)


	def scale_edge_attribute(self, layer = None, attribute = None, beta = 1):
		"""
		multiply specified edge attributes by a specified constant

		Args:
		    layer (str, optional): the layer to scale
		    attribute (TYPE, optional): attribute to scale
		    beta (int, optional): constant by which to scale attribute
		"""
		d = {e: self.G.edge[e[0]][e[1]][attribute] * beta for e in self.layers_as_subgraph([layer]).edges_iter()}
		nx.set_edge_attributes(self.G, attribute, d)

	def spatial_plot_interpolated(self, layer1, layer2, measure, title, file_name, show = False):
		""" Create an interpolated plot of a spatially-distributed measure.
		
		Args:
		    layer1 (str): base layer, nodes must have 'measure' as a numeric attribute. 
		    layer2 (str): other layer to draw, e.g. 'metro'
		    measure (str): measure to plot, e.g. betweenness centrality
		    title (str): plot title
		    file_name (str): name under which to save file
		    show (bool, optional): if True, call plt.show() to interactively view plot. 
		
		Returns:
		    None
		"""
		N = self.layers_as_subgraph([layer1])
		M = self.layers_as_subgraph([layer2])

		x = np.array([N.node[n]['pos'][1] for n in N.node])
		y = np.array([- N.node[n]['pos'][0] for n in N.node])
		z = np.array([float(N.node[n][measure]) for n in N.node])
	   
		mx, my = 100, 100
		xi = np.linspace(x.min(), x.max(), mx)
		yi = np.linspace(y.min(), y.max(), my)

		xi, yi = np.meshgrid(xi, yi)
		xi, yi = xi.flatten(), yi.flatten()

		# Calculate IDW
		grid1 = utility.simple_idw(x,y,z,xi,yi, threshhold = .2)
		grid1 = grid1.reshape((my, mx))

		utility.plot(x,y,z,grid1)
		plt.title(title)
		N.position = {n : (N.node[n]['pos'][1], - N.node[n]['pos'][0]) for n in N}

		nx.draw(N,N.position,
				edge_color = 'grey',
				edge_size = .01,
				node_color = 'black',
				node_size = 0,
				alpha = .2,
				with_labels=False,
				arrows = False)

		M.position = {m : (M.node[m]['pos'][1], - M.node[m]['pos'][0]) for m in M}
		nx.draw(M, 
				M.position,
				edge_color = '#5A0000',
				edge_size = 60,
				node_size = 0,
				arrows = False,
				with_labels = False)

		if show == True:
			plt.show()
		
	def betweenness_plot_scatter(self, layer1, layer2, measure, title, file_name):
		"""Create a basic spatially distributed scatter plot.
		
		Args:
		    layer1 (str): base layer, nodes must have 'measure' as a numeric attribute. 
		    layer2 (str): other layer to draw, e.g. 'metro'
		    measure (str): measure to plot, e.g. betweenness centrality
		    title (str): plot title
		    file_name (str): name under which to save file
		
		Returns:
		    None
		"""
		N = self.layers_as_subgraph([layer1])
		M = self.layers_as_subgraph([layer2])

		N.position = {n : (N.node[n]['pos'][1], N.node[n]['pos'][0]) for n in N}
		N.size = [float(N.node[n][measure]) / 20000 for n in N]

		nx.draw(N,N.position,
			edge_color = 'grey',
			edge_size = .01,
			node_size = N.size,
			node_color = '#003399',
			linewidths = 0,
			alpha = .1,
			with_labels=False,
			arrows = False)

		M.position = {m : (M.node[m]['pos'][1], M.node[m]['pos'][0]) for m in M}
		nx.draw(M, 
			M.position,
			edge_color = '#5A0000',
			edge_size = 60,
			node_size = 0,
			arrows = False,
			with_labels = False)
		plt.savefig(file_name)

	def random_nodes_in(self, layers = [], n_nodes = 1):
		"""Generate a list of random node names in a specified set of layers
		
		Args:
		    layers (list, optional): the layers from which to sample
		    n_nodes (int, optional): the number of random nodes to return
		
		Returns:
		    list: a list of randomly sampled nodes
		"""
		H = self.layers_as_subgraph(layers = layers)
		nodes = [n for n in H.node]
		nodes = random.sample(nodes, n_nodes)
		
		for n in nodes:
			assert self.G.node[n]['layer'] in layers  
		
		return nodes

	def layers_of(self, nbunch = []):
		"""retrieve the layers included in a given set of nodes. 
		[probably slow due to repeated queries to G, a cached version might be faster]
		
		Args:
		    nbunch (list, optional): a list of node ids
		
		Returns:
		    list: a list of layers assocated with the specified node ids
		"""
		layers_found = set([self.G.node[n]['layer'] for n in nbunch])
		
		return layers_found

	def shortest_path(self, source, target, weight = 'weight'):
		"""basic single-path routing. Fine for basic use-cases, too slow for mass computations. 
		
		Args:
		    source (list): a list of source nodes
		    target (list): a list of target nodes
		    weight (str, optional): the edge weight by which to compute shortest paths. 
		
		Returns:
		    list: a list of the nodes comprising the shortest path. 
		"""
		return nx.shortest_path(self.G, source = source, target = target, weight = weight)

	def mean_edge_attr_per(self, layers = [], attr = 'weight', weight_attr = None):
		"""compute the optionally weighted mean of a specified edge attribute 
		
		Args:
		    layers (list, optional): the layers across which to compute the average
		    attr (str, optional): the numeric attribute of which to compute the mean 
		    weight_attr (str, optional): the numeric attribute to use as weights 
		
		Returns:
		    None
		"""
		H = self.layers_as_subgraph(layers = layers)
		attr_array = np.array([H.edge[e[0]][e[1]][attr] for e in H.edges_iter()])

		if weight_attr is not None: 
			weight_array = np.array([H.edge[e[0]][e[1]][weight_attr] for e in H.edges_iter()])
		else: 
			weight_array = np.array([1 for e in H.edges_iter()])

		return np.dot(attr_array.T, weight_array) / (weight_array**2).sum()


	def local_intermodality(self, layer = None, thru_layer = None, weight = None):
		"""Compute the local intermodality of a set of nodes and save as a node attribute. 
		
		Args:
		    layer (str, optional): the layer for which to compute intermodality
		    thru_layer (str, optional): the layer through which a path couns as 'intermodal'
		    weight (str, optional): the numeric edge attribute used to weight paths
		
		Returns:
		    None
		"""
		g = utility.nx_2_igraph(self.G)
		nodes = g.vs.select(layer=layer)

		def intermodality(v, g, nodes = nodes, weight = weight):
			paths = g.get_shortest_paths(v, nodes, weights = weight)
			total = len(nodes)
			intermodal = 0
			for p in paths: 
				if thru_layer in [g.vs[u]['layer'] for u in p]:
					intermodal += 1
			return intermodal * 1.0 / total

		d = {v['name'] : intermodality(v = v, g = g, nodes = nodes, weight = weight) for v in nodes}
		
		nx.set_node_attributes(self.G, 'intermodality', d)

	def spatial_outreach(self, layer = None, weight = None, cost = None, attrname = 'outreach'):
		'''
		Compute the spatial outreach of all nodes in a layer according to a specified edge weight (e.g. cost_time_m). 
		Currently uses area of convex hull to measure outreach.
		
		Args:
		    layer (TYPE, optional): the layer in which to compute spatial outreach
		    weight (TYPE, optional): the numeric edge attribute by which to measure path lengths
		    cost (TYPE, optional): the maximum path length 
		    attrname (str, optional): the base name to use when saving the computed outreach
		'''
		from shapely.geometry import MultiPoint
		
		def distance_matrix(nodes, weight):
			N = len(nodes)
			lengths = g.shortest_paths_dijkstra(weights = weight, source = nodes, target = nodes)
			d = {nodes[i] : {nodes[j] : lengths[i][j] for j in range(N) } for i in range(N)}
			return d

		def ego(n, cost):
			return [j for j in nodes if d[n][j] <= cost]
	
		def area(n, cost):
			points = [pos[n] for n in ego(n, cost)]
			return MultiPoint(points).convex_hull.area
			
		print 'converting to igraph'
		g = utility.nx_2_igraph(self.G)
		nodes = g.vs.select(lambda vertex: vertex['layer'] == layer)['name']
		pos = {v['name'] : v['pos'] for v in g.vs.select(lambda v: v['name'] in nodes)}
		print 'computing distance matrix, this could take a while'
		d = distance_matrix(nodes, weight)
		print 'computing outreach'
		outreach = {n : area(n, cost) for n in nodes}
		nx.set_node_attributes(self.G, attrname, outreach)

	def proximity_to(self, layers, to_layer):
		"""Calculate how close nodes in one layer are to nodes in another. Closeness 
		is measured as Euclidean distance, not graph distance. 
		
		Args:
		    layers (TYPE): base layer from which to compute proximity
		    to_layer (TYPE): layer to which to calculate proximity 
		
		Returns:
		    TYPE: Description
		"""
		layers_copy = self.layers_as_subgraph(layers)	
		to_layer_copy = self.layers_as_subgraph([to_layer])
		d = {n : utility.find_nearest(n, layers_copy, to_layer_copy)[1] for n in layers_copy.node}
		nx.set_node_attributes(self.G, 'proximity_to_' + to_layer, d)
		


