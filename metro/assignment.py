from metro import utility
from metro import io
import time
import pandas as pd
import numpy as np
import networkx as nx
import itertools
import sys
import collections 
from copy import deepcopy

def readOD(directory, purpose, time):
	if purpose < 0 or purpose > 3 or time < 0 or time > 3:
		print 'No file exists: choose parameters between 0 and 3'
		return 
	ODpath = directory +  str(purpose) + '_' + str(time) + '.txt'
	with open(ODpath, 'r') as f:
		rows = f.read().splitlines()
		size = len(rows)-1
		ODdata = {}
		for i in range(size):
			row = rows[i+1].split()
			r =  [ int(row[0]), int(row[1]), float(row[2])]
			if r[0] in ODdata: ODdata[r[0]][r[1]] = r[2]
			else: ODdata[r[0]] = { r[1]: r[2] } 
	return ODdata

def connectorOD(multi, filePath, layer = 'taz'):
	OD  = readOD(filePath, 0,1)
	con2 = [ [ 0, [] ] for i in range(1493) ]
	for c in multi.G.nodes():
		print c
		if multi.G.node[c]['layer'] == layer:
			taz = multi.G.node[c]['taz']
			con2[taz][0] += 1
			con2[taz][1].append(c)
	allCon = [ ]
	for c in con2: allCon += c[1]
	OD2 = { o: { d: 0. for d in allCon } for o in allCon}
	for origin in OD:
		for dest in OD[origin]:
			if con2[origin][0]*con2[dest][0] > 0: 
				flow = OD[origin][dest]/(con2[origin][0]*con2[dest][0])
				for i in con2[origin][1]:
					for j in con2[dest][1]: 
						OD2[i][j] += flow
	OD3 = {}
	for origin in OD2:
		destinations = {}
		for destination in OD2[origin]:
			if OD2[origin][destination] > 0: destinations[destination] = OD2[origin][destination]
		if len(destinations) > 0: OD3[origin] = destinations
	return OD3

def routify(path):
		   """
		   Takes a list of nodes and returns the corresponding list of edges
		   
		   """
		   return [ (path[i], path[i+1]) for i in range(len(path)-1) ]

def dijkstra(multi, origin, destination, weight):
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
		for nextNode, edge in multi.G[current].items():
			if nextNode in seen: continue
			heappush(q, (dist + edge[weight], nextNode, current) )  
			
def multi_dijkstra(multi, origin, weight, OD):
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
	from heapq import heappop, heappush
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
			for edge in routify(path):
				if edge in volume: volume[edge] += OD[origin][current]
				else: volume[edge] = OD[origin][current]
			del dest[current]
			N += 1
			if N == size or len(dest) == 0: 
				return volume
		if current in seen:
			continue
		seen[current] = parent
		for nextNode, edge in multi.G[current].items():
			if nextNode in seen: continue
			heappush(q, (dist + edge[weight], nextNode, current) )  
	return volume

def multi_dijkstra_length(multi, origin, weight, OD):
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
	from heapq import heappop
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
			for e1,e2 in routify(path): lengths[current] += multi.G.edge[e1][e2][weight]
			del dest[current]
			N += 1
			if N == size or len(dest) == 0: 
				return lengths
		if current in seen:
			continue
		seen[current] = parent
		for nextNode, edge in multi.G[current].items():
			if nextNode in seen: continue
			heappush(q, (dist + edge[weight], nextNode, current) )  
	return lengths

def multi_dijkstra_topo(multi, origin, OD):
	'''
	Finds the topological shortest path form the origin to all destinations in the OD dictionary (treats every edge as having a weight of 1)

	Args:
	origin (str): the initial node for all the paths
	OD (dict): a nested dictionary with the structure OD[origin] = { destination: flow }
	The dictionary OD[origin] should contain all the destinations with flows from this origin
			
	Returns:
	dict: a dictionary of edge betweenness values indexed by edge tuples, ie (source, target)  
	'''
	from heapq import heappop
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
			for edge in routify(path):
				if edge in volume: volume[edge] += OD[origin][current]
				else: volume[edge] = OD[origin][current]
			del dest[current]
			N += 1
			if N == size or len(dest) == 0: 
				return volume
		if current in seen:
			continue
		seen[current] = parent
		for nextNode, edge in multi.G[current].items():
			if nextNode in seen: continue
			heappush(q, (dist + 1, nextNode, current) ) 

def geo_betweenness(multi, weight, OD = None):
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
		for n in multi.G.node: OD[n] = { n2: 1 for n2 in multi.G.node if n != n2 }
	for origin in OD:
		volume0 = multi.multi_dijkstra(origin, weight, OD)
		for v in volume0: 
			if v in volume: volume[v] += volume0[v]
			else: volume[v] = volume0[v]
	total = clock() - start
	ODL = sum( [ len(OD[o]) for o in OD ] )
	print "Time to calculate", ODL, "routes:", total, "(", total/ODL, ")"
	return volume

def geo_betweenness_ITA(multi, volumeScale = .25, OD = None, pathOD = None, P = (0.4, 0.3, 0.2, 0.1), a = 0.15, b = 4., exclude_layers = ['metro'], base_cost = 'free_flow_time_m', attrname = 'congested_time_m'):
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
	from sys import stdout
	

	volume = {}
	lengths = [ {}, {} ]
	start = time.clock()
	if OD == None: OD = { n: { n2: 1 for n2 in multi.G.node if n != n2 } for n in multi.G.node }
	for e1,e2 in multi.G.edges(): # set congested flow equal to free flow at base calculation
		multi.G.edge[e1][e2][attrname] = multi.G.edge[e1][e2][base_cost]
	if pathOD != None:
		print "Calculating free flow travel times"
		for origin in pathOD:
			lengths[0][origin] = multi_dijkstra_length(multi, origin, base_cost, pathOD)
	for p in P:
		multi2 = deepcopy(multi)
		for layer in exclude_layers:
			multi2.remove_layer(layer)	
		print "Starting traffic assignment for p =", p
		stdout.flush()
		for origin in OD:
			volume0 = multi_dijkstra(multi2, origin, attrname, OD)
			for v in volume0: 
				if v in volume: volume[v] += p*volume0[v]*volumeScale
				else: volume[v] = p*volume0[v]*volumeScale
		for e1,e2 in volume: 
			multi.G.edge[e1][e2][attrname] = multi.G.edge[e1][e2][base_cost]*( 1. + a*( (volume[(e1,e2)]/multi.G.edge[e1][e2]['capacity'])**b ) )
	if pathOD != None:
		print "Calculating congested travel times"
		for origin in pathOD:
			lengths[1][origin] = multi_dijkstra_length(origin, attrname, pathOD)
	total = time.clock() - start
	ODL = sum( [ len(OD[o]) for o in OD ] )
	print "Time to calculate", ODL, "routes:", total, "(", total/ODL, ")"
	return volume, lengths

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Not sure if this runs yet. 
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def od_dict(G, od_loc, pivot = True):
	print 'computing od dictionary'
	start = time.clock()
	# compute 'base' of origin-destination pairs -- we look up information onto the base
	taz = {n : int(G.node[n]['taz']) for n in G.node if G.node[n]['layer'] == 'taz'}

	o = [p[0] for p in itertools.product(taz.keys(), taz.keys())]
	d = [p[1] for p in itertools.product(taz.keys(), taz.keys())]
	
	df = pd.DataFrame({'o' : o, 'd' : d})
	
	# lookup the taz corresponding to the origin and the destination

	df['o_taz'] = df['o'].map(taz.get) 
	df['d_taz'] = df['d'].map(taz.get)
	
	# add flow by taz
	od = pd.read_table(od_loc, sep = " ")
	od.rename(columns = {'o' : 'o_taz', 'd' : 'd_taz'}, inplace = True)
	df = df.merge(od, left_on = ['o_taz', 'd_taz'], right_on = ['o_taz', 'd_taz'])
	
	# compute normalizer
	taz_norms = df.groupby(['o_taz','d_taz']).size()
	taz_norms = pd.DataFrame(taz_norms)
	taz_norms.rename(columns = {0 : 'taz_norm'}, inplace = True)
	
	# merge normalizer into df and compute normed flows
	df = df.merge(taz_norms, left_on = ['o_taz', 'd_taz'], right_index = True)
	df['flow_norm'] = df['flow'] / df['taz_norm']

	# Pivot -- makes for an easier dict comprehension
	
	if pivot: 
		od_matrix = df.pivot(index = 'o', columns = 'd', values = 'flow_norm')
		od_matrix[np.isnan(od_matrix)] = 0
		od = {i : {col : od_matrix[col][i] for col in od_matrix.columns if od_matrix[col][i] > 0.00001} for i in od_matrix.index}
	else:
		od = df[['o', 'd', 'flow_norm']]

	print 'OD dict computed in ' + str(round((time.clock() - start) / 60.0, 1)) + ' m'

	return od

def od_dict_igraph(g, od_loc, pivot = True):
	'''
	Figure out how much of this function generalizes to work for networkx keys as well,
	would be cool to use the same one for both igraph keys and for Zeyad's networkx keys. 
	''' 
	print 'computing od dictionary'
	start = time.clock()
	# compute 'base' of origin-destination pairs -- we look up information onto the base
	taz_vs = g.vs.select(lambda v : v['layer'] == 'taz')
	taz_indices = [v.index for v in taz_vs]
	o = [p[0] for p in itertools.product(taz_indices, taz_indices)] 
	d = [p[1] for p in itertools.product(taz_indices, taz_indices)]
	df = pd.DataFrame({'o' : o, 'd' : d})
	
	# lookup the taz corresponding to the origin and the destination
	taz_lookup = {v.index : int(v['taz']) for v in taz_vs}
	df['o_taz'] = df['o'].map(taz_lookup.get) 
	df['d_taz'] = df['d'].map(taz_lookup.get)
	
	# add flow by taz
	od = pd.read_table(od_loc, sep = " ")
	od.rename(columns = {'o' : 'o_taz', 'd' : 'd_taz'}, inplace = True)
	df = df.merge(od, left_on = ['o_taz', 'd_taz'], right_on = ['o_taz', 'd_taz'])
	
	# compute normalizer
	taz_norms = df.groupby(['o_taz','d_taz']).size()
	taz_norms = pd.DataFrame(taz_norms)
	taz_norms.rename(columns = {0 : 'taz_norm'}, inplace = True)
	
	# merge normalizer into df and compute normed flows
	df = df.merge(taz_norms, left_on = ['o_taz', 'd_taz'], right_index = True)
	df['flow_norm'] = df['flow'] / df['taz_norm']

	# Pivot -- makes for an easier dict comprehension
	if pivot: 
		od_matrix = df.pivot(index = 'o', columns = 'd', values = 'flow_norm')
		od_matrix[np.isnan(od_matrix)] = 0
		od = {i : {col : od_matrix[col][i] for col in od_matrix.columns if od_matrix[col][i] > 0.00001} for i in od_matrix.index}
	else:
		od = df[['o', 'd', 'flow_norm']]
	print 'OD dict computed in ' + str(round((time.clock() - start) / 60.0,1)) + ' m'

	return od

def ITA_igraph(g, od, base_cost = 'free_flow_time_m', P = [0.4, 0.3, 0.2, 0.1], a = 0.15, b = 4., scale = .25, attrname = 'congested_time_m'):
	
	def BPR(base, flow, capacity, a, b):
		return base * (1 + a * (1.0 * flow / capacity) ** b)
	
	es = g.es
	
	es['flow'] = 0
	es[attrname] = list(es[base_cost])
	
	edge_dict = collections.defaultdict(int)
	for p in P:
		start = time.clock()
		print 'assigning for p = ' + str(p)
		for o in od:
			ds = od[o]
			if len(ds) > 0:
				targets = ds.keys()
				paths = g.get_shortest_paths(o, 
											 to=targets, 
											 weights=attrname, 
											 mode='OUT', 
											 output="epath")
				for i in range(len(targets)):
					flow = od[o][targets[i]]
					for e in paths[i]:
						edge_dict[e] += p * scale * flow

		print 'assignment for p = ' + str(p) + ' completed in ' + str(round((time.clock() - start) / 60.0, 1)) + 'm'
		for key in edge_dict:
			es[key]['flow'] = edge_dict[key]
			es[key][attrname] = BPR(base = es[key][base_cost], 
									flow = es[key]['flow'], 
									capacity = float(es[key]['capacity']),
									a = a,
									b = b)