import networkx as nx
from metro import utility
from metro import analysis
from numpy import sqrt
from time import clock
import pandas as pd
import os
import numpy as np
import ita

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
		self.od = None

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
		
	def read_od(self, layer, key, od_file, sep, **kwargs):

		from itertools import product
		

		K = self.layers_as_subgraph([layer])
		cons = {n : int(K.node[n][key]) for n in K}
		

		o, d = zip(*product(cons.keys(), cons.keys()))

		df = pd.DataFrame({'o' : o, 'd' : d})

		df['o_tract'] = df['o'].map(cons.get) 
		df['d_tract'] = df['d'].map(cons.get)
		

		od = pd.read_table(od_file, sep = sep, **kwargs)
		

		od.rename(columns = {'o' : 'o_tract', 'd' : 'd_tract'}, inplace = True)

		df = df.merge(od, how = 'inner')

		norms = pd.DataFrame(df.groupby(['o_tract','d_tract']).size())
		norms.rename(columns = {0 : 'norm'}, inplace = True)

		df = df.merge(norms, left_on= ['o_tract', 'd_tract'], right_index=True)
		
		    
		df['flow'] = df['flow'] / df['norm']
		
		
		mat = df.pivot(index = 'o', columns = 'd', values = 'flow')
		mat[np.isnan(mat)] = 0
		
		od = mat.to_dict('index')
		

		from copy import deepcopy
		od_copy = deepcopy(od)
		for origin in od_copy:
		    for destination in od_copy[origin]:
		        if od[origin][destination] < .00000000001:
		            del od[origin][destination]
	

		self.od = od

	def re_key_od(self, key_map):
		self.od = re_key_od(self.od, key_map)

		

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

	def label_nodes(self, old_label = 'old_label'):
		'''
		Relabel the nodes in the format layer_int, e.g. 'streets_214'. 
		'''
		self.G = nx.convert_node_labels_to_integers(self.G, label_attribute = old_label)
		new_labels = {n : self.G.node[n]['layer'] + '_' + str(n) for n in self.G.node} 
		self.G = nx.relabel_nodes(self.G, mapping = new_labels, copy = False)

		if self.od is not None:
			key_map = {self.G.node[n][old_label] : n for n in self.G}
			self.re_key_od(key_map)

	def add_graph(self, H):
		'''
		Add a single graph to self.G and update layers. 
		args:
			H -- a networkx.DiGraph() object whose nodes and edges all have a 'layer' attribute. 
		'''
		self.G = nx.disjoint_union(self.G, H)
		self.update_layers()
		self.label_nodes('id')
		
	# def set_node_labels(self, label):
	# 	new_labels = {n : self.G.node[n][label] for n in self.G.node}
	# 	nx.relabel_nodes(self.G, mapping = new_labels, copy = False)

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
		
		if self.od is not None:
			print 'OD: loaded\n'
		else: 
			print 'OD: none \n'

		print '{0: <16}'.format('layer') + '\tnodes \tedges'
		print '-'*40
		for layer in layers:
			print '{0: <16}'.format(layer), "\t", layers[layer][0], "\t", layers[layer][1]  

	def to_txt(self, directory, file_name):
		'''
		save the multiplex to a pair of .txt documents for later processing. 
		
		args: 
			directory -- (str) the directory in which to save the file_name
			file_name -- (str) the file prefix, will have '_nodes.txt' and _edges.txt' suffixed. 
		'''
		write_nx_nodes(self.G, directory, file_name + '_nodes.txt')
		write_nx_edges(self.G, directory, file_name + '_edges.txt')

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

	def to_igraph(self):
		g = utility.nx_2_igraph(self.G)
		d = {v['name'] : v.index for v in g.vs}
		od_ig = re_key_od(self.od, d)
		return g, od_ig

	def run_ita(self, n_nodes = None, summary = False, attrname = 'congested_time_m', flow_name = 'flow', P = [.4, .3, .2, .1], scale = 1):

		g, od = self.to_igraph()
		if n_nodes is not None:
			sub_od = {key : od[key] for key in od.keys()[:n_nodes]}
			df = ita.ITA(g, sub_od, P = P, details = summary, scale = scale)
		else:
			df = ita.ITA(g, od, P = P, details = summary, scale = scale)	

		d = {(g.vs[g.es[i].source]['name'], g.vs[g.es[i].target]['name']) : g.es[i]['congested_time_m'] for i in range(len(g.es))}
		f = {(g.vs[g.es[i].source]['name'], g.vs[g.es[i].target]['name']) : g.es[i]['flow'] for i in range(len(g.es))}

		nx.set_edge_attributes(self.G, attrname, nx.get_edge_attributes(self.G, 'free_flow_time_m'))
		nx.set_edge_attributes(self.G, flow_name, 0)

		nx.set_edge_attributes(self.G, attrname, d)
		nx.set_edge_attributes(self.G, flow_name, f)

		return df

	def edges_2_df(self, layers, attrs):
		attrs = attrs  ['layer']
		return utility.edges_2(self.layers_as_subgraph(layers), attrs)


# Helper FUNCTIONS ------
# --------------------------------------------------------------------------------------------------------------

def re_key_od(od, key_map):
	new_od = {key_map[o] : {key_map[d] : od[o][d] for d in od[o]} for o in od}
	return new_od

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



def read_multi(nodes_file_name = '2_multiplex/mx_nodes.txt', edges_file_name = '2_multiplex/mx_edges.txt', sep = '\t', nid = 'id', eidfrom = 'source', eidto = 'target'):
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
	

	cap = {(e[0], e[1]) : float(G.edge[e[0]][e[1]]['capacity']) for e in G.edges_iter()} # huh, what is this doing here? 
	nx.set_edge_attributes(G, 'capacity', cap)

	multi = multiplex()
	multi.add_graph(G)

	return multi
