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
	
		
	#NOTE: I'm using my old function for now because I need to preserve the original node names to keep them consistent with imported data sets (ODs, connectors, etc)
	#The problem is that the nx.disjoint_union function uses sequential node labeling but this prevents us from using any data sets that refer to the original node names
	       #If you know a way to map from old to new names, I could convert the imported data sets to the new names and my code could just use the sequentially labeled nodes exclusively 
	       #Alternatively, we could keep the original names in the initial setup and convert to sequential labeling when preprocessing for the igraph functions
        def add_layers2(self, layer_dict):
                '''
                layer_dict: a dict of layer names and graphs, e.g. {'metro' : metro, 'street' : street}
                
                Adds layer_dict.keys() to self.layers and layer_dict.values() to multiplex, with all nodes and edges having attributes in layer_dict.keys()
                '''
                for layer in layer_dict:
                    if layer in self.layers: print "ERROR: The layer", layer, "is already defined in the multiplex, did not overwrite"
                    else:
                        self.layers.append(layer)
                        for n in layer_dict[layer].nodes(): 
                            n2 = str(layer) + "_" + str(n)
                            self.G.add_node(n2, layer = layer)
                            for attribute in layer_dict[layer].node[n]: self.G.node[n2][attribute] = layer_dict[layer].node[n][attribute]
                        for start, end in layer_dict[layer].edges():
                            start2, end2 = str(layer) + "_" + str(start), str(layer) + "_" + str(end)
                            self.G.add_edge( start2, end2, layer = layer )
                            for attribute in layer_dict[layer].edge[start][end]: self.G.edge[start2][end2][attribute] = layer_dict[layer].edge[start][end][attribute]

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
							free_flow_time_m = nearest_dist / transfer_speed + base_cost,
							cost_time_m = nearest_dist / transfer_speed + base_cost,
							capacity = capacity)
			
			bidirectional = ""
			if both: 
				self.G.add_edge(nearest, n, 
								layer = transfer_layer_name,
								weight = 0,
								dist_km = nearest_dist, 
								free_flow_time_m = nearest_dist / transfer_speed + base_cost,
								cost_time_m = nearest_dist / transfer_speed + base_cost,
								capacity = capacity) # assumes bidirectional
				bidirectional = "bidirectional "

			print 'Added ' + bidirectional + 'transfer between ' + str(n) + ' in ' + layer1 + ' and ' + str(nearest) + ' in ' + layer2 + ' of length ' + str(round(nearest_dist, 2)) + 'km.'
        
        #NOTE: This is a simpler version of spatial join that avoids using distance/time calculations and uses fixed edge attributes instead
            #While the original method would be quite useful as a general class feature, the problem with our data sets is that the metro and road networks aren't aligned well enough  
            #As a result, the Euclidean distance between nodes in different layers (especially at a scale of tens of meters) will be dominated by the noise and won't produce a meaningful result 
            #Essentially, we would be going beyond the resolution of the data and adding complexity without necessarily improving accuracy or realism
       

	def manual_join(self, layer1, layer2, joinDict, time_cost, distance = 0., both = True):
	   '''
	   Adds edges to multiplex between two layers using a given dictionary
	   '''	
	   transfer_layer_name = layer1 + '--' + layer2
	   self.layers.append(transfer_layer_name)
	   for start in joinDict:
	       end = joinDict[start]
	       self.G.add_edge(start, end, dist_km = distance, cost_time_m = time_cost, layer = transfer_layer_name)
	       if both: self.G.add_edge(end, start, dist_km = distance, cost_time_m = time_cost, layer = transfer_layer_name) 
	            
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

	def routify(self, path):
	       """
	       Takes a list of nodes and returns the corresponding list of edges
	       
	       """
	       return [ (path[i], path[i+1]) for i in range(len(path)-1) ]

	def dijkstra(self, origin, destination, weight):
	        '''
	        Finds the shortest path from a single origin to a single destination 
	        Possibly redundant, I would expect this to return the same result as your shortest_path function
	        
	        Args:
	    origin (str): the initial node in the path
	    destination (str): the intended destination of the path
	    weight (str): the edge weight by which to compute shortest paths. 

	Returns:
	    list: a list of the nodes comprising the shortest path. 
	        '''
	        q = [ (0, origin, None) ]
	        seen = {}
	        while q:
	            dist, current, parent = heappop(q)
	            if current == destination:
	                path = [current]
	                node = parent
	                while node != None:
	                    path.append(node)
	                    node = seen[node]
	                path.reverse()
	                return path
	            if current in seen:
	                continue
	            seen[current] = parent
	            for nextNode, edge in self.G[current].items():
	                if nextNode in seen: continue
	                heappush(q, (dist + edge[weight], nextNode, current) )  
	            
	def multi_dijkstra(self, origin, weight, OD):
	        '''
	        Finds the shortest path from the origin to all destinations in the OD dictionary
	        Returns the OD-weighted betweenness (traffic volume) in a dictionary indexed by the corresponding edge
	        
	        Args:
	    origin (str): the initial node for all the paths
	    weight (str): the edge weight by which to compute shortest paths.
	    OD (dict): a nested dictionary with the structure OD[origin] = { destination: flow }
	       The dictionary OD[origin] should contain all the destinations with flows from this origin
	                
	Returns:
	    dict: a dictionary of edge betweenness values indexed by edge tuples, ie (source, target)  
	        '''
	        volume = {}
	        dest = { d: False for d in OD[origin] }
	        N, size = 0, len(dest)
	        q = [ (0, origin, None) ]
	        seen = {}
	        while q:
	            dist, current, parent = heappop(q)
	            if current in dest:
	                path = [current]
	                node = parent
	                while node != None:
	                    path.append(node)
	                    node = seen[node]
	                path.reverse()
	                for edge in self.routify(path):
	                    if edge in volume: volume[edge] += OD[origin][current]
	                    else: volume[edge] = OD[origin][current]
	                del dest[current]
	                N += 1
	                if N == size or len(dest) == 0: 
	                    return volume
	            if current in seen:
	                continue
	            seen[current] = parent
	            for nextNode, edge in self.G[current].items():
	                if nextNode in seen: continue
	                heappush(q, (dist + edge[weight], nextNode, current) )  
	        return volume


	def multi_dijkstra_length(self, origin, weight, OD):
	        '''
	        Finds the shortest path from the origin to all destinations in the OD dictionary
	        Returns a dictionary containing the total path length for each OD pair in the given weight 
	        
	        Args:
	    origin (str): the initial node for all the paths
	    weight (str): the edge weight by which to compute shortest paths.
	    OD (dict): a nested dictionary with the structure OD[origin] = { destination: flow }
	       The dictionary OD[origin] should contain all the destinations with flows from this origin
	                
	Returns:
	    dict: a dictionary of path lengths (in the given weight) indexed by the destination
	        '''
	        lengths = {}
	        dest = { d: False for d in OD[origin] }
	        N, size = 0, len(dest)
	        q = [ (0, origin, None) ]
	        seen = {}
	        while q:
	            dist, current, parent = heappop(q)
	            if current in dest:
	                path = [current]
	                node = parent
	                while node != None:
	                    path.append(node)
	                    node = seen[node]
	                path.reverse()
	                lengths[current] = 0.
	                for e1,e2 in self.routify(path): lengths[current] += self.G.edge[e1][e2][weight]
	                del dest[current]
	                N += 1
	                if N == size or len(dest) == 0: 
	                    return lengths
	            if current in seen:
	                continue
	            seen[current] = parent
	            for nextNode, edge in self.G[current].items():
	                if nextNode in seen: continue
	                heappush(q, (dist + edge[weight], nextNode, current) )  
	        return lengths

	def multi_dijkstra_topo(self, origin, OD):
	        '''
	        Finds the topological shortest path form the origin to all destinations in the OD dictionary (treats every edge as having a weight of 1)
	    
	        Args:
	    origin (str): the initial node for all the paths
	    OD (dict): a nested dictionary with the structure OD[origin] = { destination: flow }
	       The dictionary OD[origin] should contain all the destinations with flows from this origin
	                
	Returns:
	    dict: a dictionary of edge betweenness values indexed by edge tuples, ie (source, target)  
	        '''
	        volume = {}
	        dest = { d: False for d in OD[origin] }
	        N, size = 0, len(dest)
	        q = [ (0, origin, None) ]
	        seen = {}
	        while q:
	            dist, current, parent = heappop(q)
	            if current in dest:
	                path = [current]
	                node = parent
	                while node != None:
	                    path.append(node)
	                    node = seen[node]
	                path.reverse()
	                for edge in self.routify(path):
	                    if edge in volume: volume[edge] += OD[origin][current]
	                    else: volume[edge] = OD[origin][current]
	                del dest[current]
	                N += 1
	                if N == size or len(dest) == 0: 
	                    return volume
	            if current in seen:
	                continue
	            seen[current] = parent
	            for nextNode, edge in self.G[current].items():
	                if nextNode in seen: continue
	                heappush(q, (dist + 1, nextNode, current) ) 

	def geo_betweenness(self, weight, OD = None):
	        '''
	        Calculates the betweenness of the multiplex network using the minimum-weight shortest path (volumes can be weighted by OD flows)
	        
	        Args:
	    weight (str): the edge weight by which to compute shortest paths.
	    OD (dict, optional): a nested dictionary with the structure OD[origin] = { destination: flow }
	       The dictionary OD[origin] should contain all the destinations with flows from this origin
	       If nothing is given, it constructs a uniform OD that corresponds to the standard betweenness calculation (ie without travel demand) 
	                
	Returns:
	    dict: a dictionary of edge betweenness values indexed by edge tuples, ie (source, target)  
	        '''
	        volume = {}
	        start = clock()
	        routed, nones = 0, 0
	        if OD == None:
	            OD = {}
	            for n in self.G.node: OD[n] = { n2: 1 for n2 in self.G.node if n != n2 }
	        for origin in OD:
	            volume0 = self.multi_dijkstra(origin, weight, OD)
	            for v in volume0: 
	                if v in volume: volume[v] += volume0[v]
	                else: volume[v] = volume0[v]
	        total = clock() - start
	        ODL = sum( [ len(OD[o]) for o in OD ] )
	        print "Time to calculate", ODL, "routes:", total, "(", total/ODL, ")"
	        return volume

	def geo_betweenness_ITA(self, volumeScale, OD = None, pathOD = None, P = (0.4, 0.3, 0.2, 0.1), a = 0.15, b = 4., layer = 'streets', base_cost = 'free_flow_time_m', attrname = 'congested_time_m'):
	        '''
	        Calculates the betweenness of the multiplex network using the minimum-weight shortest path (volumes can be weighted by OD flows)
	        This method updates travel times for congestion by using the BPR function and ITA to iteratively update the time cost of edges
	        
	        Args:
	    volumeScale (float): a multiplicative scaling factor for the OD flows
	    OD (dict, optional): a nested dictionary with the structure OD[origin] = { destination: flow }
	       The dictionary OD[origin] should contain all the destinations with flows from this origin
	       If nothing is given, it constructs a uniform OD that corresponds to the standard betweenness calculation (ie without travel demand)        
	            pathOD (dict, optional): if used, computes the shortest path lengths for all the OD pairs in this dictionary
	                This is used for validation with Google's results, it doesn't affect the betweenness calculation
	            P (list/tuple, optional): the increments for the ITA algorithm, the sum total should be 1 or it won't fully map the OD flows
	            
	Returns:
	    dict: a dictionary of edge betweenness values indexed by edge tuples, ie (source, target)  
	        '''
	        volume = {}
	        lengths = [ {}, {} ]
	        start = clock()
	        if OD == None: OD = { n: { n2: 1 for n2 in self.G.node if n != n2 } for n in self.G.node }
	        for e1,e2 in self.G.edges(): # set congested flow equal to free flow at base calculation
	        	self.G.edge[e1][e2][attrname] = self.G.edge[e1][e2][base_cost]
	        if pathOD != None:
	            print "Calculating free flow travel times"
	            for origin in pathOD:
	                lengths[0][origin] = self.multi_dijkstra_length(origin, base_cost, pathOD)
	        for p in P:
	            print "Starting traffic assignment for p =", p
	            stdout.flush()
	            for origin in OD:
	                volume0 = self.multi_dijkstra(origin, attrname, OD)
	                for v in volume0: 
	                    if v in volume: volume[v] += p*volume0[v]*volumeScale
	                    else: volume[v] = p*volume0[v]*volumeScale
	            for e1,e2 in volume: 
	                if self.G.edge[e1][e2]['layer'] == layer: 
	                    self.G.edge[e1][e2][attrname] = self.G.edge[e1][e2][base_cost]*( 1. + a*( (volume[(e1,e2)]/self.G.edge[e1][e2]['capacity'])**b ) )
	        if pathOD != None:
	            print "Calculating congested travel times"
	            for origin in pathOD:
	                lengths[1][origin] = self.multi_dijkstra_length(origin, attrname, pathOD)
	        total = clock() - start
	        ODL = sum( [ len(OD[o]) for o in OD ] )
	        print "Time to calculate", ODL, "routes:", total, "(", total/ODL, ")"
	        return volume, lengths

	def accessible_nodes(self, origin, weight, limit):
	        '''
	        Returns a dictionary of all nodes that can be accessed within the given limit according to the specified weight
	        
	        Args:
	    origin (str): the source node
	    weight (str): the edge weight by which to compute shortest paths.
	    limit (float): the upper bound for accessible shortest paths
	                
	Returns:
	    dict: a dictionary of shortest path lengths (in the given weight) indexed by the destination node
	        '''
	        q = [ (0, origin, None) ]
	        seen = {}
	        while q:
	            dist, current, parent = heappop(q)
	            if dist > limit: break
	            seen[current] = dist
	            for nextNode, edge in self.G[current].items():
	                if nextNode in seen: continue
	                heappush(q, (dist + edge[weight], nextNode, current) )  
	        return seen