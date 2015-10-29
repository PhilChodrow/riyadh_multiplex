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
				

	def weight_layers(self, weight, epsilon):
		'''
		All layers in the multiplex must have nonzero cost for betweenness calculations. 
		'''
		d = {e : self.G.edge[e[0]][e[1]][weight] + epsilon for e in self.G.edges_iter()}
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
							dist_km = nearest_dist, 
							cost_time_m = nearest_dist / transfer_speed + base_cost)
			
			bidirectional = ""
			if both: 
				self.G.add_edge(nearest, n, 
								layer = transfer_layer_name,
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
		print 'current bug, working on it'
		print 'Computing betweenness centrality -- this could take a while.' 

		g = utility.nx_2_igraph(self.layers_as_subgraph(layers))
		
		print max(g.es[weight])

		bc = g.betweenness(directed = True,
		                  cutoff = 300,
		                  weights = weight)

		d = dict(zip(g.vs['name'], bc))
		d = {key:d[key] for key in d.keys()}

		print d
		nx.set_node_attributes(self.G, attrname, d)


	def scale_edge_attribute(self, layer = None, attribute = None, beta = 1):
		d = {e: self.G.edge[e[0]][e[1]][attribute] * beta for e in self.G.edges_iter()}
		nx.set_edge_attributes(self.G, str(beta) + 'x' + attribute, d)

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








