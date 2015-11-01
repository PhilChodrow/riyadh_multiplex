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


class multiplex:
	
	def __init__(self):
		self.layers = []
		# self.layers = []
		self.G = nx.DiGraph()

	def add_layers(self, layer_dict):
		'''
		layer_dict: a dict of layer names and graphs, e.g. {'metro' : metro, 'street' : street}

		Adds layer_dict.keys() to self.layers and layer_dict.values() to multiplex, with all nodes and edges having attributes in layer_dict.keys()
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
		All layers in the multiplex must have nonzero cost for betweenness calculations. 
		'''
		d = {e : float(self.G.edge[e[0]][e[1]][weight] or 0) + epsilon for e in self.G.edges_iter()}
		nx.set_edge_attributes(self.G, weight, d)

	def label_nodes(self):
		'''
		Relabel the nodes in the format layer_int
		'''
		nx.convert_node_labels_to_integers(self.G)
		new_labels = {n : self.G.node[n]['layer'] + '_' + str(n) for n in self.G.node} 
		nx.relabel_nodes(self.G, mapping = new_labels, copy = False)

	def add_graph(self, H):
		self.G = nx.disjoint_union(self.G, H)
		self.update_layers()
		self.label_nodes()

	def get_layers(self):
		return self.layers
		
	def remove_layer(self, layer):
		'''
		layer: the name of an element of self.layers
		removes layer from self.layers and deletes all nodes + edges with attribute layer. 
		'''
		if layer not in self.layers:
			print "Sorry, " + layer + ' is not current in the multiplex.'
		else:
			self.layers.pop(layer)
			G.remove_nodes_from([n for n,attrdict in self.G.node.items() if attrdict['layer'] == layer])

	def check_layer(self, layer_name):
		'''
		Quick boolean check to see whether a given layer is actually an element of the G. 
		'''
		return layer_name in self.layers 
	
	def spatial_join(self, layer1, layer2, transfer_speed, base_cost, both = True):
		'''
		Adds edges to multiplex between ALL nodes of layer1 and the nodes of layer2 spatially nearest to layer1. 
		New edges are labelled 'layer1_layer2_T' and 'layer1_layer2_T' is added to self.layers.  
		Example: spatial_join(layer1 = 'metro', layer2 = 'street')
		Requires that each node in each G have a 'pos' tuple of format (latitude, longitude). 
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
		docs of course! 
		'''
		return self.G.subgraph([n for n,attrdict in self.G.node.items() if attrdict['layer'] in layers])

	def sub_multiplex(self, sublayers):
		'''
		sublayers: a list of layers, all of which must be elements of self.layers
		returns: a multiplex object consisting only of those layers and any connections between them. 
		'''
		subMultiplex = multiplex()        
		sublayer_dict = {layer : self.zlayer_as_subgraph(layer) for layer in sublayers}
		subMultiplex.add_layers(sublayer_dict)
		return subMultiplex

	def as_graph(self):
		'''
		Return self.multiplex as a standard graph object. self.sub_multiplex(sublayers).as_graph() to get a graph consisting only of certain layers. 
		'''
		return self.G

	def update_node_attributes(self, attr):
		'''
		attr: a dict with nodenames as keys. Values are attribute dicts. 
		'''
		print 'Not implemented.'

		for n in attr:
			for att in attr[n]: self.G.node[n][att] = attr[n][att]

	def update_edge_attributes(self, attr):
		'''
		attr: a dict with edgenames (or node 2-tuples) as keys. Values are attribute dicts. 
		'''
		for e in attr:
			for att in attr[e]: self.G.edge[e[0]][e[1]] = attr[e][att]

	def summary(self, print_summary = False):
		'''
		Return a dict of the form {'layer_name' : (num_layer_nodes, num_layer_edges)}
		'''
		layers = {layer: (len([n for n,attrdict in self.G.node.items() if attrdict['layer'] == layer]), 
		                  len([(u,v,d) for u,v,d in self.G.edges(data=True) if d['layer'] == layer])) for layer in self.layers} 
		if print_summary:
			print "Layer \t N \t E "
			for layer in layers:
				print layer, "\t", layers[layer][0], "\t", layers[layer][1] 
		return layers 

	def to_txt(self, directory, file_name):
		'''
		saves file_name_nodes.txt and file_name_edges.txt in a readable format to the working directory. 
		'''
		utility.write_nx_nodes(self.G, directory, file_name + '_nodes.txt')
		utility.write_nx_edges(self.G, directory, file_name + '_edges.txt')

	def update_layers(self):
		new_layers = set([attrdict['layer'] for n, attrdict in self.G.node.items()])
		new_layers.update(set([d['layer'] for u,v,d in self.G.edges(data=True)]))
		self.layers = list(new_layers)

	def igraph_betweenness_centrality(self, layers = None, weight = None, attrname = 'bc'):
		'''
			thru_layers -- the layers on which to calculate betweenness. 
			source_layers -- the layers to use as sources in betweenness calculation.
			target_layers -- the layers to use as targets in the betweenness calculation.  

		'''
		print 'Computing betweenness centrality -- this could take a while.' 

		g = utility.nx_2_igraph(self.layers_as_subgraph(layers))

		bc = g.betweenness(directed = True,
		                  cutoff = 300,
		                  weights = weight)

		d = dict(zip(g.vs['name'], bc))
		d = {key:d[key] for key in d.keys()}

		nx.set_node_attributes(self.G, attrname, d)


	def scale_edge_attribute(self, layer = None, attribute = None, beta = 1):
		d = {e: self.G.edge[e[0]][e[1]][attribute] * beta for e in self.layers_as_subgraph([layer]).edges_iter()}
		nx.set_edge_attributes(self.G, attribute, d)


	def streets_betweenness_plot(self, draw_metro = True, file_name = None, attrname = 'bc', norm = 1):
		'''
		Draw a simple betweenness plot, nothing too fancy. 
		'''
		streets = self.layers_as_subgraph(['streets'])
		
		streets.position = {n : (streets.node[n]['pos'][1], streets.node[n]['pos'][0]) for n in streets}
		N = len(streets.node)
		streets.size = [streets.node[n][attrname] for n in streets.node]
		streets.size = [n / norm for n in streets.size]
		
		fig = plt.figure(figsize = (15,15), dpi = 500)
		ax = fig.add_subplot(111)		
		nx.draw(streets,streets.position,
		        edge_color = 'grey',
		        edge_size = .01,
		        node_color = 'blue',
		        node_size=streets.size,
		        alpha = .1,
		        with_labels=False,
		        arrows = False,
		        cmap=plt.cm.Blues)

		if draw_metro:
			metro = self.layers_as_subgraph(['metro'])
			metro.position = {n:(metro.node[n]['pos'][1], metro.node[n]['pos'][0])for n in metro}
			nx.draw(metro, metro.position,
			        node_size = 2,
			        edge_size = 400,
			        arrows = False,
			        alpha = 1,
			        edge_color = 'red')

		plt.savefig(file_name)

	def betweenness_plot_interpolated(self, layer1, layer2, measure, title, file_name, vmin = None, vmax = None):

	    def distance_matrix(x0, y0, x1, y1):
	        obs = np.vstack((x0, y0)).T
	        interp = np.vstack((x1, y1)).T

	        # Make a distance matrix between pairwise observations
	        # Note: from <http://stackoverflow.com/questions/1871536>
	        # (Yay for ufuncs!)
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
	        gap = zi[dist.min(axis = 0) > threshhold].max()
	        zi[dist.min(axis = 0) > threshhold] = 0
	        zi = zi - gap
	        zi[zi < 0] = 0
	        return zi

	    def plot(x,y,z,grid):
	        plt.figure(figsize = (15,15), dpi = 500)
	        plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()), cmap=cm.Blues, vmin = vmin, vmax = vmax)
	        plt.hold(True)
	        # plt.colorbar()
	        

	    N = self.layers_as_subgraph([layer1])
	    M = self.layers_as_subgraph([layer2])

	    x = np.array([N.node[n]['pos'][1] for n in N.node])
	    y = np.array([- N.node[n]['pos'][0] for n in N.node])
	    z = np.array([N.node[n][measure] for n in N.node])

	    mx, my = 100, 100
	    xi = np.linspace(x.min(), x.max(), mx)
	    yi = np.linspace(y.min(), y.max(), my)

	    xi, yi = np.meshgrid(xi, yi)
	    xi, yi = xi.flatten(), yi.flatten()

	    # Calculate IDW
	    grid1 = simple_idw(x,y,z,xi,yi, threshhold = .2)
	    grid1 = grid1.reshape((my, mx))

	    plot(x,y,z,grid1)
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

	    plt.savefig(file_name)

	def betweenness_plot_scatter(self, layer1, layer2, measure, title, file_name, vmin = None, vmax = None):

		plt.figure(figsize = (15,15), dpi = 500)
		N = self.layers_as_subgraph([layer1])
		M = self.layers_as_subgraph([layer2])

		N.position = {n : (N.node[n]['pos'][1], N.node[n]['pos'][0]) for n in N}
		N.size = [N.node[n][measure] / 20000 for n in N]

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


