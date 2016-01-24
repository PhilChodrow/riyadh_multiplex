import networkx as nx
from metro.utility import *
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
		self.od -- a dict of dicts ....
	'''
	def __init__(self):
		self.layers = []
		self.G = nx.DiGraph()
		self.od = None

	# -------------------------------------------------------------------------
	# NETWORK CONSTRUCTION	
	# -------------------------------------------------------------------------

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
	def label_nodes(self, old_label = 'old_label'):
		"""
		Summary:
			Generate new labels for the nodes of the multiplex. 
		
		Args:
		    old_label (str, optional): Name of attribute under which to save the old labels
		
		Returns:
		    None 
		"""
		self.G = nx.convert_node_labels_to_integers(self.G, 
		                                            label_attribute = old_label)
		new_labels = {n : self.G.node[n]['layer'] + '_' + str(n) 
					  for n in self.G.node} 

		self.G = nx.relabel_nodes(self.G, mapping = new_labels, copy = False)

		if self.od is not None:
			key_map = {self.G.node[n][old_label] : n for n in self.G}
			self.re_key_od(key_map)

	def add_graph(self, H):
		"""
		Summary: 
			Add a graph H to the multiplex and update labels. 
		
		Args:
		    H (networkx.DiGraph): The graph to add 
		
		Returns:
		    None
		"""
		self.G = nx.disjoint_union(self.G, H)
		self.update_layers()
		self.label_nodes('id')

	# -------------------------------------------------------------------------
	# OD CONSTRUCTION
	# -------------------------------------------------------------------------

	def read_od(self, layer, key, od_file, sep, **kwargs):
		"""
		Summary:
			Read an OD matrix formatted with columns 'o', 'd', and 'flow', where 'o' and 'd' match a node attribute in a layer of self. 
		
		Args:
		    layer (str): the layer of nodes whose attributes match the 'o' and 'd' columns
		    key (str): the node attribute matching the 'o' and 'd' columns
		    od_file (str): the path of the OD matrix to read in
		    sep (str): The separator used in the OD file
		    **kwargs: Additional arguments passed to pd.read_table
		
		Returns:
		    None
		"""
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
		"""
		Re-key the OD matrix according to a user-supplied mapping of keys. 
		
		Args:
		    key_map (dict): a dictionary with the old keys as keys and the new keys as values.  
		
		Returns:
		    None
		"""
		self.od = re_key_od(self.od, key_map)

		
	# -------------------------------------------------------------------------
	# NETWORK MANIPULATION	
	# -------------------------------------------------------------------------

	def add_epsilon(self, weight, epsilon):
		'''
		Summary:
			Add a small positive number epsilon to an edge attribute. 
		Args:
		    weight (TYPE): The edge attribute to modify
		    epsilon (TYPE): The number to add to each attribute
		'''
		d = {e : float(self.G.edge[e[0]][e[1]][weight] or 0) + epsilon for e in self.G.edges_iter()}
		nx.set_edge_attributes(self.G, weight, d)

	def remove_layer(self, layer):
		"""
		Summary:
			Delete a layer of the multiplex. All nodes in that layer are deleted, as well as any edges that begin or end at a deleted node. 
		
		Args:
		    layer (str): the layer to remove 
		
		Returns:
		    None
		"""
		if layer not in self.layers:
			print "Sorry, " + layer + ' is not current in the multiplex.'
		else:
			self.layers.remove(layer)
			self.G.remove_nodes_from([n for n,attrdict in self.G.node.items() if attrdict['layer'] == layer])
	
	def spatial_join(self, layer1, layer2, transfer_speed, base_cost, capacity, both = True):
		'''
		Summary: 
			Add edges to between ALL nodes of layer1 and the nodes of layer2 spatially nearest to the nodes of layer1. New edges are labelled 'layer1_layer2_T' and 'layer1_layer2_T' is added to self.layers.  Requires that each node in each G have a 'pos' tuple of format (latitude, longitude). 
		
		Args: 
			layer1 (str): base layer, all nodes joined to one node in layer2
			layer2 (str): layer to which layer1 will be joined
			transfer_speed (float): assumed speed at which transfer distance can be traversed, e.g. walking speed from street to metro. 
			base_cost (float): base cost associated with transfer, e.g. mean time spent waiting for metro.
			both (bool): if true, transfer is bidirectional. 

		Returns:
			None  		
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
	
	def update_node_attributes(self, attr):
		'''
		Summary:
			Set the attributes of self.G.node
		
		Args:
			attr (dict): a dict with nodenames as keys. Values are attribute dicts. 

		Returns:
			None
		'''

		for n in attr:
			for att in attr[n]: self.G.node[n][att] = attr[n][att]

	def update_edge_attributes(self, attr):
		'''
		Summary: 
			Set the attributes of self.G.edge

		Args:
			attr (dict): a dict with edgenames (or node 2-tuples) as keys. Values are attribute dicts. 

		Returns:
			None
		'''
		for e in attr:
			for att in attr[e]: self.G.edge[e[0]][e[1]] = attr[e][att]
	
	def update_layers(self):
		"""
		Summary:
			Check that the layers of self include all layers present in self.G

		Args:
			None
		
		Returns:
		    None 
		"""
		new_layers = set([attrdict['layer'] 
		                 for n, attrdict in self.G.node.items()])
		new_layers.update(set([d['layer'] 
		                  for u,v,d in self.G.edges(data=True)]))

		self.layers = list(new_layers)

	def scale_edge_attribute(self, layer = None, attribute = None, beta = 1):
		"""
		Summary:
			Multiply specified edge attributes by a specified constant
		
		Args:
		    layer (str, optional): the layer to scale
		    attribute (str, optional): attribute to scale
		    beta (int, optional): constant by which to scale attribute

		Returns:
			None
		"""
		d = {e: self.G.edge[e[0]][e[1]][attribute] * beta 
		     for e in self.layers_as_subgraph([layer]).edges_iter()}
		nx.set_edge_attributes(self.G, attribute, d)
	
	# -------------------------------------------------------------------------
	# NETWORK QUERIES	
	# -------------------------------------------------------------------------

	def get_layers(self):
		"""
		Summary: Get a list of layers currently included in the multiplex. 
		
		Returns:
		    list: a list of layers  
		"""
		return self.layers
		

	def check_layer(self, layer_name):
		"""
		Summary: Check for the presence of a layer in the multiplex. 
		
		Args:
		    layer_name (str): the name of the layer to check for.  
		
		Returns:
		    bool: True iff layer_name is the name of a layer in the multiplex.  
		"""
		return layer_name in self.layers 
	
                    
	def layers_as_subgraph(self, layers):
		'''
		Summary:
			return a subset of the layers of self.G as a networkx.DiGraph() object. 
		args: 
			layers (list): a list of layers to return

		Returns:
			None
		'''
		return self.G.subgraph([n for n,attrdict in self.G.node.items() 
		                       if attrdict['layer'] in layers])

	def sub_multiplex(self, sublayers):
		'''
		Summary:
			Return a subset of the layers of self.G as a multiplex() object. 
		
		Args:
			sublayers (list): a list of layers, all of which must be elements of self.layers

		Returns:
			None
		
		'''
		sub_multiplex = multiplex()        
		sublayer_dict = {layer : self.layer_as_subgraph(layer) 
						 for layer in sublayers}
		sub_multiplex.add_layers(sublayer_dict)
		return sub_multiplex

	def as_graph(self):
		'''
		Summary: 
			Return self.multiplex as a networkx.DiGraph() object. 

		Args:
			None

		Returns:
			None
		'''
		return self.G


	def summary(self):
		'''
		Summary: 
			View a summary of self, printed to the terminal

		Args: 
			None

		Returns:
			None

		'''
		layers = {layer: (len([n for n,attrdict in self.G.node.items() 
		                  if attrdict['layer'] == layer]), 
						  len([(u,v,d) for u,v,d in self.G.edges(data=True) 
						      if d['layer'] == layer])) for layer in self.layers} 
		
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
		Summary:
			Save self to node and edge text files for further processing. 
		
		Args: 
			directory (str): the directory in which to save the file_name
			file_name (str): the file prefix, will have '_nodes.txt' and _edges.txt' suffixed. 

		Returns:
			None
		'''
		write_nx_nodes(self.G, directory, file_name + '_nodes.txt')
		write_nx_edges(self.G, directory, file_name + '_edges.txt')

	def nodes_2_df(self, layers, attrs):
		"""
		Summary:
			Create a pandas.DataFrame in which each row is a node and each column a node attribute, for a specified set of layers. 
		
		Args:
		    layers (list): a list of strings giving the layers to include in the returned df.  
		    attrs (list): a list of strings giving the node attributes to include as columns in the returned df .
		
		Returns:
		    pandas.DataFrame: a df in which each row is a node and each specified column is a node attribute.  
		"""
		attrs = attrs + ['layer']
		return nodes_2_df(self.layers_as_subgraph(layers), attrs)

	def to_igraph(self):
		"""
		Summary: 
			Retrieve self.G and self.od in igraph form. 
		
		Returns:
		    igraph.Graph: an igraph-formatted copy of G for use in computationally-intensive operations. 
		    od_ig: a copy of self.od keyed to the returned igraph graph. 
		"""
		g = nx_2_igraph(self.G)
		d = {v['name'] : v.index for v in g.vs}
		od_ig = re_key_od(self.od, d)
		return g, od_ig
		
	def edges_2_df(self, layers, attrs):
		"""
		Summary: 
			Create a pandas.DataFrame in which each row is an edge and each column an edge attribute. 
		
		Args:
		    layers (list): a list of strings indicating the layers to be included in the df 
		    attrs (list): a list of attributes to include as columns
		
		Returns:
		    pandas.DataFrame: a df in which each row is an edge and each column is an edge attribute.  
		"""
		attrs = attrs + ['layer']
		return edges_2_df(self.layers_as_subgraph(layers), attrs)
	# -------------------------------------------------------------------------
	# ANALYSIS
	# -------------------------------------------------------------------------
	
	def mean_edge_attr_per(self, layers = [], attr = 'dist_km', weight_attr = None):
		""" 
		Summary:
			Compute the (optionally weighted) mean of a specified edge attribute over a specified set of layers
		
		Args:
		    layers (list, optional): the layers across which to compute the average
		    attr (str, optional): the numeric attribute of which to compute the mean 
		    weight_attr (str, optional): the numeric attribute to use as weights 
		
		Returns:
		    The weighted average of attr over the specified layer set. 
		"""
		H = self.layers_as_subgraph(layers = layers)
		attr_array = np.array([H.edge[e[0]][e[1]][attr] for e in H.edges_iter()])

		if weight_attr is not None: 
			weight_array = np.array([H.edge[e[0]][e[1]][weight_attr] 
			                        for e in H.edges_iter()])
		else: 
			weight_array = np.array([1 for e in H.edges_iter()])

		return np.average(attr_array, weights = weight_array)


	def run_ita(self, n_nodes = None, summary = False, base_cost = 'free_flow_time_m', attrname = 'congested_time_m', flow_name = 'flow', P = [.4, .3, .2, .1], scale = 1):
		"""
		Summary: 
			Run Iterated Traffic Assignment on self.G, using self.od as the OD matrix. 
		
		Args:
		    n_nodes (TYPE, optional): the number of nodes on which to run analysis; only n_nodes = None should be used for publication. 
		    summary (bool, optional): whether to construct a route-by-route summary of key metrics. EXTREMELY EXPENSIVE in time and memory. 
		    base_cost (str, optional): the cost to use as the base in ITA. 
		    attrname (str, optional): the name of the new edge attribute to reflect congested travel time
		    flow_name (str, optional): the name of the new edge attribute to reflect congested flow. 
		    P (list, optional): the iteration levels to use. 
		    scale (int, optional): the fraction of flow to assign. 
		
		Returns:
		    pd.DataFrame: if summary = True, return a df with route-by-route metrics. Otherwise None.  
		"""
		g, od = self.to_igraph()
		if n_nodes is not None:
			sub_od = {key : od[key] for key in od.keys()[:n_nodes]}
			df = ita.ITA(g, sub_od, base_cost, P = P, details = summary, scale = scale)
		else:
			df = ita.ITA(g, od, base_cost, P = P, details = summary, scale = scale)	

		d = {(g.vs[g.es[i].source]['name'], g.vs[g.es[i].target]['name']) : g.es[i]['congested_time_m'] for i in range(len(g.es))}
		f = {(g.vs[g.es[i].source]['name'], g.vs[g.es[i].target]['name']) : g.es[i]['flow'] for i in range(len(g.es))}

		nx.set_edge_attributes(self.G, attrname, nx.get_edge_attributes(self.G, 'free_flow_time_m'))
		nx.set_edge_attributes(self.G, flow_name, 0)

		nx.set_edge_attributes(self.G, attrname, d)
		nx.set_edge_attributes(self.G, flow_name, f)

		return df


	def route_summary(self, n_nodes = None, cost = 'congested_time_m', layer = 'streets', funs = None):
		'''
		Summary: 
			Compute route-wise metrics over shortest paths using flexibly-defined functions. 

		Args:
		    n_nodes (int, optional): the number of nodes over which to compute -- only n_nodes = None should be used for final analysis. 
		    cost (str, optional): edge cost for computing shortest paths. 
		    layer (str, optional): the layer over which to compute. 
		    funs (dict, optional): a dict in which the keys are column names and the values are functions applied to each edge. Each function should retrieve a scalar value from an igraph edge instance. Example:

			funs = {'dist' : lambda e : e['dist_km'],
			        'free_flow_time' : lambda e : e['free_flow_time_m'],
			        'weighted_demand' : lambda e : e['flow_100'] * e['dist_km'],
			        'weighted_capacity' : lambda e : e['capacity'] * e['dist_km']}

		Returns:
			A pandas.DataFrame with the routes, flows, and summarised metrics.  
		
		'''
		g, od = self.to_igraph()
		if n_nodes is not None:
			sub_od = {key : od[key] for key in od.keys()[:n_nodes]}
			df = igraph_route_summary(g, sub_od, cost, layer, funs)

		else:
			df = igraph_route_summary(g, od, cost, layer, funs)

		def get_flow(row):
			return od[row['o']][row['d']]

		df['flow'] = df.apply(get_flow, axis = 1)
		return df

	def path_lengths(self, n_nodes, weight, mode = 'array'):
		"""
		Summary:
			Compute shortest path lengths under a given weight. 
		
		Args:
		    n_nodes (int): the number of nodes for which to compute; only n_nodes = None should be used for final analysis.  
		    weight (str): the edge attribute to use as cost for shortest paths.  
		    mode (str, optional): the mode in which to return the results; see analysis.path_lengths_igraph() for options. 
		
		Returns:
		    TYPE: 
		"""
		g, od = self.to_igraph()
		nodes = np.array([v.index for v in g.vs 
		                 if g.vs[v.index]['layer'] == 'streets'])

		if n_nodes is not None:
			nodes = np.random.choice(nodes, size = n_nodes, replace = False) 
		lengths = analysis.path_lengths_igraph(g, nodes, weight, mode)
		lengths = lengths[~np.isinf(lengths)]
		return lengths


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------


def re_key_od(od, key_map):
	"""
	Summary:
		Re-key an od matrix according to a mapping from old keys to new ones. 
	
	Args:
	    od (dict): a dict of dicts giving ods 
	    key_map (dict): a dict in which keys are old labels and values are new labels. 
	
	Returns:
	    (dict): the re-keyed od matrix. 
	"""
	new_od = {key_map[o] : {key_map[d] : od[o][d] for d in od[o]} for o in od}
	return new_od

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
	

	cap = {(e[0], e[1]) : float(G.edge[e[0]][e[1]]['capacity']) 
	        for e in G.edges_iter()} 
	nx.set_edge_attributes(G, 'capacity', cap)

	multi = multiplex()
	multi.add_graph(G)

	return multi

def igraph_route_summary(g, od, cost, layer, funs):
    """
    Summary:
    	Compute a flexible summary of route information over shortest paths. 
    
    Args:
        g (igraph.Graph): the graph over which to compute shortest paths
        od (dict): a dict of dicts containing OD information keyed to nodes of g
        cost (str): the edge attribute to use as cost for shortest paths
        layer (str): the layer in which the edge attributes are to be computed
        funs (dict): a dict in which the keys are column names and the values are functions applied to each edge. Each function should retrieve a scalar value from an igraph edge instance. Example:

			funs = {'dist' : lambda e : e['dist_km'],
			        'free_flow_time' : lambda e : e['free_flow_time_m'],
			        'weighted_demand' : lambda e : e['flow_100'] * e['dist_km'],
			        'weighted_capacity' : lambda e : e['capacity'] * e['dist_km']}
    
    Returns:
        pd.DataFrame: a data frame including columns for origin, destination, and specified metrics.  
    """
    summary = []
    es = g.es
    
    def entries(o, d, path, funs):
        labs = {'o' : o, 'd' : d} 
        metrics = {lab : sum([funs[lab](es[e]) 
                             for e in path if es[e]['layer'] == layer]) 
        		   for lab in funs}
        labs.update(metrics)
        return labs
    
    for o in od:
        ds = od[o]
        if len(ds) > 0:
            targets = ds.keys()
            paths = g.get_shortest_paths(o, 
                                         to=targets, 
                                         weights=cost, 
                                         mode='OUT', 
                                         output='epath')
            update = [entries(o, targets[i], paths[i], funs) 
            		  for i in range(len(targets))]
            summary += update
            
    return pd.DataFrame(summary)
    
