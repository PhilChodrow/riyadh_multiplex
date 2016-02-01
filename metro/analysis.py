from math import sqrt
from metro import utility
import networkx as nx
import numpy as np
from collections import defaultdict
from shapely.geometry import MultiPolygon, Point, shape
import pandas as pd


def distance(pos1,pos2):
	"""
	Summary:
		Compute geographical distance between two points
	
	Args:
		pos1 (tuple): a tuple of the form (lat, lon)
		pos2 (tuple): a tuple of the form (lat, lon)
	
	Returns:
		float: the geographical distance between points, in kilometers
	"""
	LAT_DIST = 110766.95237186992 / 1000.0 # in km. See http://www.csgnetwork.com/degreelenllavcalc.html
	LON_DIST = 101274.42720366278 / 1000.0 # in km. See http://www.csgnetwork.com/degreelenllavcalc.html

	return sqrt((LON_DIST*(pos1[0]- pos2[0]))**2 + 
	            (LAT_DIST*(pos1[1] - pos2[1]))**2)

def gini_coeff(x):
	'''
	Summary:
		ompute the gini coefficient from an array of floats
	
	From http://www.ellipsix.net/blog/2012/11/the-gini-coefficient-for-distribution-inequality.html
	
	Args:
		x (np.array()): an array of floats

	Returns:
		float: the gini coefficient of the array x. 
	'''
	# requires all values in x to be zero or positive numbers,
	# otherwise results are undefined
	n = len(x)
	s = x.sum()
	r = np.argsort(np.argsort(-x)) # calculates zero-based ranks
	return 1 - (2.0 * (r*x).sum() + s)/(n*s)



def local_intermodality(self, layer = None, thru_layer = None, weight = None):
	"""
	Summary:
		Compute the local intermodality of a set of nodes and save as a node attribute. 
	
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

	d = {v['name'] : intermodality(v = v, g = g, nodes = nodes, weight = weight) 
		 for v in nodes}
	
	nx.set_node_attributes(self.G, 'intermodality', d)

def spatial_outreach(multi, node_layer = 'taz', thru_layers = ['streets'], weight = None, cost = None, attrname = 'outreach'):
	'''
	Summary:
		Compute the spatial outreach of all nodes in a layer according to a specified edge weight (e.g. cost_time_m). 
		Currently uses area of convex hull to measure outreach.
	
	Args:
		layer (TYPE, optional): the layer in which to compute spatial outreach
		weight (TYPE, optional): the numeric edge attribute by which to measure path lengths
		cost (TYPE, optional): the maximum path length 
		attrname (str, optional): the base name to use when saving the computed outreach

	Returns: 
		None
	'''
	from shapely.geometry import MultiPoint
	from math import pi
	
	LAT_DIST = 110766.95237186992 / 1000.0 # in km. See http://www.csgnetwork.com/degreelenllavcalc.html
	LON_DIST = 101274.42720366278 / 1000.0 # in km. See http://www.csgnetwork.com/degreelenllavcalc.html

	def distance_matrix(nodes, weight):
		N = len(nodes)

		lengths = g.shortest_paths_dijkstra(weights = weight, 
		                                    source = nodes, 
		                                    target = nodes)

		d = {nodes[i] : {nodes[j] : lengths[i][j] for j in range(N) } 
			 for i in range(N)}

		return d

	def ego(n, cost, d):
		return [j for j in nodes if d[n][j] <= cost]

	def area(n, cost, d):
		points = [pos[n] for n in ego(n, cost, d)]
		return MultiPoint(points).convex_hull.area
		
	g = utility.nx_2_igraph(multi.layers_as_subgraph(thru_layers + [node_layer]))
	nodes = g.vs.select(lambda vertex: vertex['layer'] == node_layer)['name']
	pos = {v['name'] : (v['lon'] * LON_DIST, v['lat'] * LAT_DIST) 
		   for v in g.vs.select(lambda v: v['name'] in nodes)}
	
	d = distance_matrix(nodes, weight)
	
	outreach = {n : sqrt(area(n, cost, d)/pi) for n in nodes}
	nx.set_node_attributes(multi.G, attrname, outreach)

def proximity_to(multi, layers, to_layer):
	"""
	Summary:
		Calculate how close nodes in one layer are to nodes in another. Closeness 
		is measured as Euclidean distance, not graph distance. 
	
	Args:
		layers (list): list of strings, base layers from which to compute proximity
		to_layer (str): layer to which to calculate proximity 
	
	Returns:
		None
	"""
	layers_copy = multi.layers_as_subgraph(layers)	
	to_layer_copy = multi.layers_as_subgraph([to_layer])
	d = {n : utility.find_nearest(n, layers_copy, to_layer_copy)[1] 
		 for n in layers_copy.node}
	nx.set_node_attributes(multi.G, 'proximity_to_' + to_layer, d)

def accessible_nodes(self, origin, weight, limit):
	'''
	Summary:
		Return a dictionary of all nodes that can be accessed within the given limit according to the specified weight
	
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

def path_lengths_igraph(g, nodes, weight, mode = 'array'):
	'''
	Summary: 
		quick finding of shortest path lengths between nodes. 
		If as_df, returns as a pd dataframe with columns for origin and destination. 
		If not as_df, returns a 1d np.array(). This is significantly faster in situations when we don't need
		to keep track of o and d.
		
	Args:
	    g (igraph.Graph()): the graph over which to compute shortest paths
	    nodes (list): the nodes to use as sources and sinks
	    weight (str): the edge attribute used to compute costs
	    mode (str, optional): the format in which to return the results; options include 
	    'array' and 'df'

	returns:
		the shortest path lengths as either an array or a pandas.DataFrame
	'''
	lengths = g.shortest_paths_dijkstra(weights = weight, 
	                                    source = nodes, 
	                                    target = nodes)
	if mode == 'df':
		q = [(nodes[i],nodes[j],lengths[i][j]) for i in range(len(nodes)) 
		for j in range(len(nodes))]

		o = [tup[0] for tup in q]
		d = [tup[1] for tup in q]
		p = [tup[2] for tup in q]

		df = pd.DataFrame({'o' : o, 'd' : d, weight + '_length' : p})
		return lengths
	elif mode == 'array':
		lengths = np.array(lengths)
		return lengths.ravel()
	else:
		return lengths

def standardize(array):
	"""
	Summary:
		Standardize an array
	
	Args:
	    array (np.array): the array to standardize 
	
	Returns:
	    array : the standardized array 
	"""
	return (array - array.mean()) / array.std()



def congestion_gradient(free_flow_time_m, flow, capacity, a = .15, b = 4, components = 'both'): 
	"""
	Summary: 
		Compute the derivative of the BPR congestion function for specified values of free flow time, flow, and capacity. 
		The derivative has two terms, either of which can be returned, or the sum of the two.  
	
	Args:
	    free_flow_time_m (float): the free flow time of the edge
	    flow (float): the flow through the edge
	    capacity (float): the capacity of the edge
	    a (float, optional): BPR tuning parameter
	    b (int, optional): BPR tuning parameter
	    components (str, optional): which component of the derivative to return; 
	    options include 'selfish', 'social', or 'both'
	
	Returns:
	    TYPE: 
	"""
	selfish = free_flow_time_m *     a * (flow / capacity) ** b
	social  = free_flow_time_m * b * a * (flow / capacity) ** b

	if components == 'both':
		return selfish + social
	elif components == 'selfish':
		return selfish
	elif components == 'social':
		return social



def weighted_avg_and_std(values, weights):
    """
    
    Summary:
    	Return the weighted average and standard deviation.
    Args:
    	values, weights -- Numpy ndarrays with the same shape.
    Returns:
    	The average and standard deviation as a 2-tuple
    	
    From http://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  
    return (average, sqrt(variance))


def construct_tract_getter(tracts, id_field):
    """
    Summary:
    	Construct a function to look up the tract of a row in a df based on its latitude and longitude
    
    Args:
        tracts: a collection of Shapely geometry polygons
        id_field (str): the attribute of tracts that gives the 'name' of the tract 
    
    Returns:
        function that reads in a row of a df and returns the tract in which the corresponding node lies 
    """
    tract_dict = defaultdict(str)
    for tract in tracts:
        tract_id = tract['properties'][id_field]
        tract = shape(tract['geometry'])
        if tract.geom_type == 'MultiPolygon':
            for p in list(tract):
                tract_dict[p] = tract
        elif tract.geom_type == 'Polygon':
            tract_dict[tract] = tract_id
    
    def get_tract(row):
        p = Point(row['lon'], row['lat'])
        for tract in tract_dict:
            if p.within(tract):
                return tract_dict[tract]
    return get_tract

    def edge_wise_cor(multi, attr1, attr2, weight):
	    """
	    Summary:
	    	Compute the weighted edge-wise correlation coefficient of two scalar edge attributes. 
	    
	    Args:
	        multi (multiplex.multiplex): the multiplex on which to compute
	        attr1 (str): the first attribute 
	        attr2 (str): the second attribute
	        weight (str): the attribute by which to weight
	    
	    Returns:
	        float: the correlation between attr1 and attr2, weighted by weight.  
	    """
	    df = multi.edges_2_df(['streets'], [attr1, attr2, weight])
	    for attr in [attr1, attr2]:
	        df[attr + '_weighted'] = df[attr] * df[weight] 
	    df = df.dropna(thresh = 5)
	    return np.corrcoef(df[attr1 + '_weighted'], df[attr2 + '_weighted'])[0][1]
