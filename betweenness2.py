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

multi = utility.read_multi(nodes_file_name = '2. multiplex/multiplex_nodes.txt', 
                           edges_file_name = '2. multiplex/multiplex_edges.txt')

multi.remove_layer('taz')
multi.remove_layer('metro')

multi.igraph_betweenness_centrality(layers = multi.get_layers(),
                                   weight = 'congested_time_m',
                                   attrname = 'congested_betweenness')
multi.igraph_betweenness_centrality(layers = multi.get_layers(),
                                   weight = 'free_flow_time_m',
                                   attrname = 'free_flow_betweenness')
multi.igraph_betweenness_centrality(layers = multi.get_layers(),
                                   weight = 'dist_km',
                                   attrname = 'dist_betweenness')

G = multi.layers_as_subgraph(['streets'])

fig = plt.figure(figsize = (30,10), dpi = 500, facecolor='black')
ax1 = fig.add_subplot(131)
utility.spatial_plot(G, 'dist_betweenness', ax1, title = 'dist_betweenness')
ax2 = fig.add_subplot(132)
utility.spatial_plot(G, 'free_flow_betweenness', ax2, title = 'free_flow_betweenness')
ax3 = fig.add_subplot(133)
utility.spatial_plot(G, 'congested_betweenness', ax3, title = 'congested_betweenness')

plt.savefig('4. figs/betweenness/sample.png')