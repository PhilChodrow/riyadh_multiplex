import utility
import pandas as pd
import numpy as np
import itertools
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import scipy.ndimage as ndimage
import networkx as nx
from compute_congestion import *

def main():
    multi = utility.read_multi(nodes_file_name = '2. multiplex/multiplex_nodes.txt', 
                               edges_file_name = '2. multiplex/multiplex_edges.txt')
    G = multi.layers_as_subgraph(['taz', 'streets', 'metro'])
    g = utility.nx_2_igraph(G)
    od = od_dict(g, od_data_dir = '1. data/taz_od/', od_file_name = '0_1.txt')

    weights = ['dist_km', 'free_flow_time_m', 'congested_time_m']
    fig = plt.figure(figsize = (30,10), dpi = 500, facecolor='black')
    i = 1
    for weight in weights: 
        weighted_betweenness(g, od, weight = weight, scale = .25, attrname = 'od_bc_' + weight)
        d = {g.vs[i]['id'] : g.vs[i]['od_bc_' + weight] for i in range(len( g.vs))}
        nx.set_node_attributes(multi.G, 'od_bc_' + weight, d)

        ax = fig.add_subplot(1,3,i)
        utility.spatial_plot(G, 'od_bc_' + weight, ax, title = 'Betweenness: '  + weight)
        i += 1

    plt.savefig('4. figs/betweenness/sample2.png')

def weighted_betweenness(g, od, weight = 'free_flow_time_m',scale = .25, attrname = 'weighted_betweenness'):
    vs = g.vs
    es = g.es
    
    # initialize graph attributes for collecting later
    vs[attrname] = 0
    
    # collects flows
    node_dict = collections.defaultdict(int)

    # main assignment loop
    start = time.clock()
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
                    
    print 'betweenness calculated in in ' + str(round((time.clock() - start) / 60.0,1)) + 'm'
    for key in node_dict:
        vs[key][attrname] = node_dict[key]

if __name__ == "__main__":
    main()