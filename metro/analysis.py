from math import sqrt
from metro import utility
import networkx as nx
import numpy as np
from collections import defaultdict
from shapely.geometry import MultiPolygon, Point, shape
import pandas as pd




def distance(pos1,pos2):
	"""Compute geographical distance between two points
	
	Args:
		pos1 (tuple): a tuple of the form (lat, lon)
		pos2 (tuple): a tuple of the form (lat, lon)
	
	Returns:
		float: the geographical distance between points, in kilometers
	"""
	LAT_DIST = 110766.95237186992 / 1000.0 # in km. See http://www.csgnetwork.com/degreelenllavcalc.html
	LON_DIST = 101274.42720366278 / 1000.0 # in km. See http://www.csgnetwork.com/degreelenllavcalc.html
	return sqrt((LON_DIST*(pos1[0]- pos2[0]))**2 + (LAT_DIST*(pos1[1] - pos2[1]))**2)

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

def igraph_betweenness_centrality(multi, layers = None, weight = None, attrname = 'bc'):
		'''
		compute the (weighted) betweenness centrality of one or more layers and save to self.G.node attributes. 
		args: 
			thru_layers -- the layers on which to calculate betweenness. 
			source_layers -- the layers to use as sources in betweenness calculation.
			target_layers -- the layers to use as targets in the betweenness calculation.  
		'''

		g = utility.nx_2_igraph(multi.layers_as_subgraph(layers))

		bc = g.betweenness(directed = True,
						  cutoff = 300,
						  weights = weight)

		d = dict(zip(g.vs['name'], bc))
		d = {key:d[key] for key in d.keys()}

		nx.set_node_attributes(multi.G, attrname, d)

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

def spatial_outreach(multi, node_layer = 'taz', thru_layers = ['streets'], weight = None, cost = None, attrname = 'outreach'):
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

	def ego(n, cost, d):
		return [j for j in nodes if d[n][j] <= cost]

	def area(n, cost, d):
		points = [pos[n] for n in ego(n, cost, d)]
		return MultiPoint(points).convex_hull.area
		
	
	g = utility.nx_2_igraph(multi.layers_as_subgraph(thru_layers + [node_layer]))
	nodes = g.vs.select(lambda vertex: vertex['layer'] == node_layer)['name']
	pos = {v['name'] : (v['lon'], v['lat']) for v in g.vs.select(lambda v: v['name'] in nodes)}
	
	d = distance_matrix(nodes, weight)
	
	outreach = {n : sqrt(area(n, cost, d)) for n in nodes}
	nx.set_node_attributes(multi.G, attrname, outreach)

def proximity_to(multi, layers, to_layer):
	"""Calculate how close nodes in one layer are to nodes in another. Closeness 
	is measured as Euclidean distance, not graph distance. 
	
	Args:
		layers (TYPE): base layer from which to compute proximity
		to_layer (TYPE): layer to which to calculate proximity 
	
	Returns:
		TYPE: Description
	"""
	layers_copy = multi.layers_as_subgraph(layers)	
	to_layer_copy = multi.layers_as_subgraph([to_layer])
	d = {n : utility.find_nearest(n, layers_copy, to_layer_copy)[1] for n in layers_copy.node}
	nx.set_node_attributes(multi.G, 'proximity_to_' + to_layer, d)

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

def weighted_betweenness(g, od, weight = 'free_flow_time_m',scale = .25, attrname = 'weighted_betweenness'):

	import collections
	
	vs = g.vs
	es = g.es
	
	# initialize graph attributes for collecting later
	vs[attrname] = 0
	
	# collects flows
	node_dict = collections.defaultdict(int)

	# main assignment loop
	
	for o in od:
		ds = od[o]
		if len(ds) > 0:
			targets = ds.keys()
			paths = g.get_shortest_paths(o, 
										 to=targets, 
										 weights=weight, 
										 mode='OUT', 
										 output="vpath") # compute paths
			for path in paths:
				if len(path) > 0:
					flow = ds[path[-1:][0]]
					for v in path: 
						node_dict[v] += scale * flow
					

	for key in node_dict:
		vs[key][attrname] = node_dict[key]

def path_lengths_igraph(g, nodes, weight, mode = 'array'):
	'''
	quick finding of shortest path lengths between nodes. 
	If as_df, returns as a pd dataframe with columns for origin and destination. 
	If not as_df, returns a 1d np.array(). This is significantly faster in situations when we don't need
	to keep track of o and d. 
	'''
	lengths = g.shortest_paths_dijkstra(weights = weight, source = nodes, target = nodes)
	if mode == 'df':
		q = [(nodes[i],nodes[j],lengths[i][j]) for i in range(len(nodes)) for j in range(len(nodes))]
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
	return (array - array.mean()) / array.std()


def congestion_gradient(free_flow_time_m, flow, capacity, a = .15, b = 4): # based on BPR function
    return free_flow_time_m * a * b * ((flow / capacity) ** b)

def traffic_summary(g, od, weight, flow):
    from collections import defaultdict
    df = defaultdict(int)
    
    di = {e.index : (g.es[e.index]['layer'],
                g.es[e.index]['capacity'],
                g.es[e.index][flow], 
                float(g.es[e.index]['free_flow_time_m']),
                float(g.es[e.index][weight]),
                float(g.es[e.index]['dist_km'])) 
         for e in g.es}

    o = []
    d = []
    flow = []
    traffic = []
    capacity = []
    free_flow_time_m = []
    congested_time_m = []
    dist_km = []
    gradient = []
    
    for origin in od:
        ds = od[origin]
        if len(ds) > 0: 
            targets = ds.keys()
            paths = g.get_shortest_paths(origin,
                                        to = targets,
                                        weights = weight,
                                        mode = 'OUT',
                                        output = 'epath')
            for i in range(len(targets)):
                street_path = [e for e in paths[i] if di[e][0] == 'streets']
                metro_path = [e for e in paths[i] if di[e][0] == 'metro']
                
                o += [origin]
                d += [targets[i]]
                flow += [od[origin][targets[i]]]
                traffic += [np.sum([di[e][2]*di[e][5] for e in street_path])]
                capacity += [np.sum([di[e][1]*di[e][5] for e in street_path])]
                free_flow_time_m += [np.sum([di[e][3] for e in street_path])]
                congested_time_m += [np.sum([di[e][4] for e in street_path])]
                dist_km += [np.sum([di[e][5] for e in street_path])]
                gradient += [np.sum([congestion_gradient(di[e][3], di[e][2], di[e][1]) 
                                            for e in street_path])]
    
    df = pd.DataFrame({'o' : np.array(o), 
                       'd' : np.array(d), 
                       'traffic' : np.array(traffic),
                       'capacity' : np.array(capacity),
                       'free_flow_time_m' : np.array(free_flow_time_m),
                       weight : np.array(congested_time_m),
                       'dist_km' : np.array(dist_km),
                       'flow' : np.array(flow),
                       'gradient' : np.array(gradient)})
    
    df = df[df['capacity'] != 0]
    
    df['ratio'] = df['traffic']/df['capacity']
    df['time_lost'] = df[weight]/df['free_flow_time_m']
    df['mean_speed'] = df['dist_km'] / df[weight]*60
    df['congestion_impact'] = df['gradient'] * df['flow']
    
    return df

def street_level_gradients(multi):
    def compute_gradients(sources, targets, graph):
        o = []
        d = []
        gradient = []
        for v in sources:
            paths = graph.get_shortest_paths(v, to = targets,weights = 'congested_time_m', output = 'epath')
            for i in range(len(targets)):
                grad = sum([graph.es[e]['gradient'] for e in paths[i]])
                o.append(v.index)
                d.append(targets[i].index)
                gradient.append(grad)
        df = pd.DataFrame({'o' : o, 'd' : d, 'gradient' : gradient})
        return df[['o','d','gradient']]
    
    g = utility.nx_2_igraph(multi.layers_as_subgraph(['taz','streets']))
    taz_nodes = g.vs.select(lambda v : v['layer'] == 'taz')
    	
    df = compute_gradients(taz_nodes, taz_nodes, g)
    
    node_lookup = node_lookup = {v.index : v['name'] for v in g.vs}
    df['o'] = df['o'].map(node_lookup.get)
    df['d'] = df['d'].map(node_lookup.get)
    
    return df

def to_metro_gradients(multi):
    def route_gradient(u,v,g):
        path = g.get_shortest_paths(u,v,weights='congested_time_m',output='epath')[0]
        gradient = sum([g.es[e]['gradient'] for e in path])
        return gradient

    def rowwise_gradient(row):
        return route_gradient(row['taz'], row['closest_metro'], g)

    G = multi.layers_as_subgraph(['metro', 'taz', 'streets'])
    g = utility.nx_2_igraph(G)

    taz_nodes = g.vs.select(lambda v : v['layer'] == 'taz')
    metro_nodes = g.vs.select(lambda v : v['layer'] == 'metro')

    x = g.shortest_paths_dijkstra(source = taz_nodes, target = metro_nodes, weights = 'congested_time_m') 
    x = np.array(x)
    nearest = np.argmin(x, axis=1)
    
    taz_nodes = [taz_nodes[i].index for i in range(len(taz_nodes))]
    closest_metro = [metro_nodes[nearest[i]].index for i in range(len(taz_nodes))]
    dg = pd.DataFrame({'taz' : taz_nodes, 'closest_metro' : closest_metro})
    dg['gradient'] = dg.apply(rowwise_gradient, axis = 1)

    node_lookup = {v.index : v['name'] for v in g.vs}
    dg['closest_metro'] = dg['closest_metro'].map(node_lookup.get)
    dg['taz'] = dg['taz'].map(node_lookup.get)
    
    return dg

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.

    From http://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, sqrt(variance))


def construct_tract_getter(tracts, id_field):
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
