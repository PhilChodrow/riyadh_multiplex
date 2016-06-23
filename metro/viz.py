import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.cm as cm
from metro import utility
import networkx as nx
from shapely.geometry import MultiPolygon, Point, shape
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib import colors
from metro import analysis

def bubble_plot(G, size, color, size_factor = 1, **kwargs):
    '''
    Summary:
      Plot a networkx.DiGraph() in which the nodes have both size and some scalar feature reflected by color
    
    Args:
        G (networkx.DiGraph()): the graph to map 
        size (str): the node attribute of G to be mapped to size 
        color (str): the node attribute of G to be mapped to color
        size_factor (int, optional): size scaling factor for manual tuning 
        **kwargs: additional args to nx.draw

  Returns:
      None
    '''
    G.size = [G.node[n][size]*size_factor for n in G.node]
    G.color = [G.node[n][color] for n in G.node]
    nx.draw_networkx(G, get_coords(G),
        edge_color = 'white', 
        edge_size = 0.001,
        width = 0.1,
        node_color = G.color,
        node_size = G.size,
        linewidth = 0,
        with_labels = False,
        arrows = False,
        **kwargs)

def get_coords(G):
    """
    Summary:
      Extract the coordinates of the nodes of G as lon-lat pairs
    
    Args:
        G (networkx.DiGraph()): the network from which to retrieve coordinates. Must contain 'lon' and 'lat' attributes.  
    
    Returns:
        dict: keyed by node of G, values are tuples of the form (lon, lat)  
    """
    return {n : (G.node[n]['lon'], G.node[n]['lat']) for n in G}

def get_edge_scalar(G, attr):
    """
    Summary: 
      Retrieve a scalar attribute from the edges of G
    
    Args:
        G (networkx.DiGraph()): the network from which retrieve the attributes 
        attr (str): the attribute to extract
    
    Returns:
        array: an np.array of edge attributes. 
    """
    return np.array([G.edge[e[0]][e[1]][attr] for e in G.edges_iter()])

def flow_plot(multi, flow_attr, ax, cmap = 'viridis', background = True, scale = .0005, **kwargs):
    """
    Summary:
      Convenience function for plotting flows on the street and metro networks. 
    
    Args:
        multi (multiplex.multiplex): a multiplex object containing a metro layer and a streets layer.  
        flow_attr (str): the name of the edge attribute containing edge flows
        ax (ax): the matplotlib.axis on which to plot  
    
    Returns:
        None: 
    """
    from collections import Counter

    G = multi.layers_as_subgraph(['streets'])
    
    # Set flow
    one_way = nx.get_edge_attributes(G, flow_attr)
    other_way = {(e[1], e[0]) : one_way[e] for e in one_way}

    one_way = Counter(one_way)
    other_way = Counter(other_way)

    total_flow = one_way + other_way

    # Set capacity

    one_way = nx.get_edge_attributes(G, 'capacity')
    other_way = {(e[1], e[0]) : one_way[e] for e in one_way}

    one_way = Counter(one_way)
    other_way = Counter(other_way)

    total_capacity = one_way + other_way

    H = G.to_undirected()
    nx.set_edge_attributes(H, flow_attr, total_flow)
    nx.set_edge_attributes(H, 'capacity', total_capacity)

    if background:
      nx.draw_networkx_edges(H, 
                             get_coords(H),
                             edge_color = 'grey',
                             width = 1,
                             arrows = False,
                             alpha = .2,
                             ax = ax)

    nx.draw_networkx_edges(H, 
                           get_coords(H),
                           edge_color = get_edge_scalar(H, flow_attr)/get_edge_scalar(H, 'capacity'),
                           width = get_edge_scalar(H, flow_attr) * scale,
                           arrows = False,
                           edge_cmap = plt.get_cmap(cmap),
                           ax = ax,
                           **kwargs)

    if multi.check_layer('metro'):
      G = multi.layers_as_subgraph(['metro'])
      one_way = nx.get_edge_attributes(G, flow_attr)
      other_way = {(e[1], e[0]) : one_way[e] for e in one_way}

      one_way = Counter(one_way)
      other_way = Counter(other_way)

      total = one_way + other_way
      H = G.to_undirected()
      nx.set_edge_attributes(H, flow_attr, total)
      nx.draw_networkx_edges(H, get_coords(H),
              edge_color = 'white', 
              width = get_edge_scalar(H, flow_attr) * scale,
              node_color = 'white',
              node_size = 0,
              alpha = .4,
              with_labels = False,
              arrows = False,
              ax = ax)

def weighted_hist(ax, measure, weights, label, standardized = False, n = 100, **kwargs):
    """
    Summary:
      Plot a weighted histogram of one measure, weighted by another
    
    Args:
        ax (matplotlib.axis): the axis on which to plot 
        measure (np.array): the measure to plot
        weights (np.array): the weights with respect to which to plot the measure, must be same shape as measure 
        label (str): the label of the plot 
        standardized (bool, optional): if True, standardize the histogram
        n (int, optional): the number of bins with which to plot the histogram 
        **kwargs: additional keyword arguments passed to ax.plot()
    
    Returns:
        None 
    """
    if standardized:
        mu, sigma = analysis.weighted_avg_and_std(measure, weights)
        hist_data = 1.0 * (measure - mu) / sigma
    else:
        hist_data = measure
    hist = np.histogram(hist_data, weights = weights, normed = True, bins = n)
    x = hist[1][:n] 
    y = hist[0]
    ax.plot(x,y, label = label, **kwargs)