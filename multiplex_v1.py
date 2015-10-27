import networkx as nx
from numpy import sqrt
from heapq import heappush, heappop
from time import clock
from sys import stdout

class multiplex:
    
    def __init__(self):
	self.layers = {}
	self.network = nx.DiGraph()
	self.join = {}

    def add_layers(self, layer_dict):
	'''
	layer_dict: a dict of layer names and graphs, e.g. {'metro' : metro, 'street' : street}

	Adds layer_dict.keys() to self.layers and layer_dict.values() to multiplex, with all nodes and edges having attributes in layer_dict.keys()
	'''
	for layer in layer_dict:
	    if layer in self.layers: print "ERROR: The layer", layer, "is already defined in the multiplex, did not overwrite"
	    else:
	        self.layers[layer] = layer_dict[layer]
	        for n in layer_dict[layer].nodes(): 
	            n2 = str(layer) + str(n)
	            self.network.add_node(n2, layer = layer)
	            for attribute in layer_dict[layer].node[n]: self.network.node[n2][attribute] = layer_dict[layer].node[n][attribute]
                for start, end in layer_dict[layer].edges():
                    start2, end2 = str(layer) + str(start), str(layer) + str(end)
                    self.network.add_edge( start2, end2 )
                    for attribute in layer_dict[layer].edge[start][end]: self.network.edge[start2][end2][attribute] = layer_dict[layer].edge[start][end][attribute]

    def get_layers(self):
	return self.layers.keys()

    def remove_layer(self, layer):
	'''
	layer: the name of an element of self.layers
	removes layer from self.layers and deletes all nodes + edges with attribute layer. 
	'''
	if layer in self.layers:
	    for n in self.layers[layer]['nodes']: self.network.remove_node(n)
	    del( self.layers[layer] )
	    return True
	return False

    def check_layer(self, layer_name):
	'''
	Quick boolean check to see whether a given layer is actually an element of the network. 
	'''
	return layer_name in self.layers 
    
    def spatial_join(self, layer1, layer2, time, distance = 0., both = True):
	'''
	Adds edges to multiplex between ALL nodes of layer1 and the nodes of layer2 spatially to layer1. 
	New edges are labelled 'layer1_layer2_T' and 'layer1_layer2_T' is added to self.layers.  
	Example: spatial_join(layer1 = 'metro', layer2 = 'street')
	'''	
	nodes1, nodes2 = self.layers[layer1].nodes(), self.layers[layer2].nodes()
	self.join[ (layer1,layer2) ] = {}
	if both: self.join[ (layer2, layer1) ] = {}
	for n1 in nodes1:
	    closest = (1e10, None)
	    start = self.layers[layer1].node[n1]['pos']
	    for n2 in nodes2:
	        dist = sqrt( (start[0]-self.layers[layer2].node[n2]['pos'][0])**2 + (start[1]-self.layers[layer2].node[n2]['pos'][1])**2 )
                if dist < closest[0]: closest = (dist, n2)
            start, end = str(layer1) + str(n1), str(layer2) + str(closest[1])
            self.network.add_edge(start, end, dist = distance, time = time)
            self.join[ (layer1,layer2) ][ (start, end) ] = self.network.edge[start][end]
            if both: 
                self.network.add_edge(end, start, dist = distance, time = time) 
                self.join[ (layer2,layer1) ][ (end, start) ] = self.network.edge[end][start]
            

    def sub_multiplex(self, sublayers):
        '''
        sublayers: a list of layers, all of which must be elements of self.layers
        returns: a multiplex object consisting only of those layers and any connections between them. 
        '''
        subMultiplex = multiplex()
        layers = { layer: self.layers[layer] for layer in sublayers }
        subMultiplex.add_layers(layers)
        for layer1 in sublayers:
            for layer2 in sublayers:
                if layer1 != layer2:
                    if (layer1, layer2) in self.join: 
                        subMultiplex.join[ (layer1,layer2) ] = {}
                        for e1, e2 in self.join[(layer1, layer2)]:
                            subMultiplex.network.add_edge(e1, e2)
                            for attribute in self.network.edge[e1][e2]: subMultiplex.network.edge[e1][e2][attribute] = self.network.edge[e1][e2][attribute] 
                            subMultiplex.join[ (layer1,layer2) ][ (e1,e2) ] = subMultiplex.network.edge[e1][e2]
        return subMultiplex

    def as_graph(self):
        '''
        Return self.multiplex as a standard graph object. self.sub_multiplex(sublayers).as_graph() to get a graph consisting only of certain layers. 
	'''
	return self.network

    def update_node_attributes(self, attr):
        '''
        attr: a dict with nodenames as keys. Values are attribute dicts. 
        '''
        print 'Not implemented.'
        for n in attr:
            for att in attr[n]: self.network.node[n][att] = attr[n][att]

    def update_edge_attributes(self, attr):
        '''
        attr: a dict with edgenames (or node 2-tuples) as keys. Values are attribute dicts. 
        '''
        for e in attr:
            for att in attr[e]: self.network.edge[e[0]][e[1]] = attr[e][att]

    def summary(self, print_summary = False):
        '''
        Return a dict of the form {'layer_name' : (num_layer_nodes, num_layer_edges)}
        '''
        print 'Not implemented.'
        layers = {  layer: ( len(self.layers[layer].nodes()), len(self.layers[layer].edges()) ) for layer in self.layers }
        transfers, layers['Transfers'] = [ len(self.join[t]) for t in self.join ], [0, 0]
        for t in transfers: layers[ 'Transfers' ][1] += t
        if print_summary:
            print "Layer \t N \t U "
            for layer in layers:
                print layer, "\t", layers[layer][0], "\t", layers[layer][1] 
        return layers 

    def to_txt(file_name):
	'''
	saves file_name_nodes.txt and file_name_edges.txt in a readable format to the working directory. 
	'''
