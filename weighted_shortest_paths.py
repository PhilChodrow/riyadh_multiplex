
import pandas as pd
import numpy as np
import cProfile
import time
import networkx as nx
from metro import multiplex as mx

m = mx.read_multi(nodes_file_name = '3_throughput/mx_flow_nodes.txt', 
                      edges_file_name = '3_throughput/mx_flow_edges.txt')

for layer in ['metro', 'metro--streets']:
    m.remove_layer(layer)

m.read_od(layer = 'taz', # keys are in this layer
          key = 'taz', # this is the key attribute
          od_file = '1_data/taz_od/0_1.txt', # here's where the file lives
          sep = ' ') # this is what separates entries

for base_cost in ['uniform_time_m', 'free_flow_time_m', 'congested_time_m_100', 'dist_km']:
	start = time.clock()
	df = m.run_ita(n_nodes = None,
	                   summary = True,
	                   base_cost = base_cost,
	                   attrname = 'time_NA',
	                   flow_name = 'flow_NA',
	                   scale = .25,
	                   P = [1]) # do shortest paths

	df.to_csv('3_throughput/shortest_' + base_cost + '.csv')

	print 'Shortest paths for ' + base_cost + ' computed in ' + str(round((time.clock() - start) / 60.0, 1)) + 'm'
