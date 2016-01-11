import networkx as nx
from metro import utility
from metro import analysis
from numpy import sqrt
from time import clock
import pandas as pd
import os

class multiplex:
	'''
	multiplex class is a thin wrapper for a the networkx.DiGraph() object, for 
	cases in which that object is composed of distinct layers. 
	attributes: 
		self.layers -- (list) list of strings
		self.G -- a networkx.DiGraph object, all of whose nodes and edges have a 
		'layer' attribute.  
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
		Add a small positive number to a numeric attribute of self.G.edges()
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
		self.set_node_labels('id')
		
	def set_node_labels(self, label):
		new_labels = {n : self.G.node[n][label] for n in self.G.node}
		nx.relabel_nodes(self.G, mapping = new_labels, copy = False)

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
	
	def spatial_join(self, layer1, layer2, transfer_speed, base_cost, capacity, both = True):
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
		def find_nearest(n, N1, N2):
			dists = {m: analysis.distance( (N1.node[n]['lon'], N1.node[n]['lat']), (N2.node[m]['lon'], N2.node[m]['lat']) ) for m in N2}
			nearest = min(dists, key=dists.get)
			nearest_dist = dists[nearest]
			return nearest, nearest_dist

		transfer_layer_name = layer1 + '--' + layer2
		self.layers.append(transfer_layer_name)

		layer1_copy = self.layers_as_subgraph([layer1])	
		layer2_copy = self.layers_as_subgraph([layer2])

		edges_added = 0
		for n in layer1_copy.node:
			nearest, nearest_dist = find_nearest(n, layer1_copy, layer2_copy)
			self.G.add_edge(n, nearest, 
							layer = transfer_layer_name,
							weight = 0,
							dist_km = nearest_dist, 
							free_flow_time_m = nearest_dist / transfer_speed + base_cost,
							uniform_time_m = nearest_dist / transfer_speed + base_cost,
							capacity = capacity)
			
			bidirectional = ""
			if both: 
				self.G.add_edge(nearest, n, 
								layer = transfer_layer_name,
								weight = 0,
								dist_km = nearest_dist, 
								free_flow_time_m = nearest_dist / transfer_speed + base_cost,
								uniform_time_m = nearest_dist / transfer_speed + base_cost,
								capacity = capacity) # assumes bidirectional
				bidirectional = "bidirectional "
			edges_added += 1

		print 'Added ' + str(edges_added) + ' ' + bidirectional + 'transfers between '  + layer1 + ' and ' + layer2 + '.'
                    
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
		sublayer_dict = {layer : self.layer_as_subgraph(layer) for layer in sublayers}
		sub_multiplex.add_layers(sublayer_dict)
		return sub_multiplex

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
		set the attributes of self.G.edge
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
		io.write_nx_nodes(self.G, directory, file_name + '_nodes.txt')
		io.write_nx_edges(self.G, directory, file_name + '_edges.txt')

	def update_layers(self):
		'''
		Check that layers includes all values of 'layer' attributes in self.G
		'''
		new_layers = set([attrdict['layer'] for n, attrdict in self.G.node.items()])
		new_layers.update(set([d['layer'] for u,v,d in self.G.edges(data=True)]))
		self.layers = list(new_layers)

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

	def layers_of_path(self, nbunch = []):
		"""retrieve the layers included in a given set of nodes. 
		[probably slow due to repeated queries to G, a cached version might be faster]
		
		Args:
		    nbunch (list, optional): a list of node ids
		
		Returns:
		    list: a list of layers assocated with the specified node ids
		"""
		layers_found = set([self.G.node[n]['layer'] for n in nbunch])
		
		return layers_found

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

	def nodes_2_df(self, layers, attrs):
		attrs = attrs + ['layer']
		return utility.nodes_2_df(self.layers_as_subgraph(layers), attrs)

