import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.cm as cm
from metro import utility
import networkx as nx
# from shapely.geometry import MultiPolygon, Point, shape
# from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib import colors
from metro import analysis

def spatial_plot(G, attr, ax, title = 'plot!', layer = 'taz'):
	import scipy.ndimage as ndimage

	cols = ['layer', 'lon', 'lat', attr]
	df = utility.nodes_2_df(G, cols)

	df = df[np.isnan(df[attr]) == False]

	n = 2000
	grid_x, grid_y = np.mgrid[df.lon.min():df.lon.max():n * 1j, 
					  df.lat.min():df.lat.max():n * 1j]
	zj = np.zeros(grid_x.shape)
	
	lonmax = df.lon.max()
	lonmin = df.lon.min()
	latmax = df.lat.max()
	latmin = df.lat.min()

	df = df[df['layer'] == layer]


	for i in df.index:
		x = int((df.loc[i]['lon'] - lonmin) / (lonmax - lonmin)*n) - 1
		y = int((df.loc[i]['lat'] - latmin) / (latmax - latmin)*n) - 1 
		zj[x][y] += df.loc[i][attr]
		
	zi = ndimage.gaussian_filter(zj, sigma=12.0, order=0)

	ax.contourf(grid_x, grid_y, zi, 100, linewidths=0.1, cmap=plt.get_cmap('afmhot'), alpha = 1, vmax = 1./1. * zi.max())
	
	G.position = {n : (G.node[n]['lon'], G.node[n]['lat']) for n in G}
	nx.draw(G, G.position,
			edge_color = 'white', 
			edge_size = 0.01,
			node_color = 'white',
			node_size = 0,
			alpha = .15,
			with_labels = False,
			arrows = False)
	plt.title(title)


def bubble_plot(G, size, color, size_factor = 1, **kwargs):
	G.position = {n : (G.node[n]['lon'], G.node[n]['lat']) for n in G}
	G.size = [G.node[n][size]*size_factor for n in G.node]
	G.color = [G.node[n][color] for n in G.node]
	n = nx.draw(G, G.position,
		edge_color = 'grey', 
		edge_size = 0.01,
		node_color = G.color,
		node_size = G.size,
		linewidth = 0,
		with_labels = False,
		arrows = False,
		**kwargs)

def choropleth(tracts, d, key_field, **kwargs):
    norm = colors.Normalize(vmin = 0, vmax = 1)
    cmap = plt.get_cmap(kwargs['cmap'])
    cols = cm.ScalarMappable(norm = norm, cmap = cmap)
    
    patches = []

    for tract in tracts:
        try:
        	interval = kwargs['vmax'] - kwargs['vmin']
        	colorval = max(d[tract['properties'][key_field]], kwargs['vmin'])
        	colorval = min(colorval, kwargs['vmax'])
        	colorval = (colorval - kwargs['vmin']) / interval
        	# colorval = 1-colorval
        	color = cols.to_rgba(colorval)
        	
        	# print cols(colorval), cols(1-colorval)
        except KeyError:
        	try: 
        		colorval = 1.0 * kwargs['default_val'] / kwargs['vmax']
        		# print colorval
        		color = cols.to_rgba(colorval)
        	except KeyError:
	        	try: 
	        		color = kwargs['default_color']
	        	except KeyError:
	        		color = (1,1,1,0)
        poly = MultiPolygon([shape(tract['geometry'])])
        for idx, p in enumerate(poly):
        	patches.append(PolygonPatch(p, fc = color, ec='#555555', lw=.0, alpha=1, zorder=0))

    xlim = kwargs['xlim']
    ylim = kwargs['ylim']

    # fig = plt.figure(figsize = (10,10), dpi = 300)
    # ax = fig.add_subplot(111)

    ax = kwargs['ax']

    ax.add_collection(PatchCollection(patches, match_original=True))
    ax.set(xlim = xlim, ylim = ylim, xticks = [], yticks = [])
    sns.despine(top = True, left = True, right = True, bottom = True)
    
    if kwargs['colorbar']:
	    m = cm.ScalarMappable(cmap=cols)
	    m.set_array([kwargs['vmin'], kwargs['vmax']])
	    plt.colorbar(m,	ax = ax)


def get_coords(G):
    return {n : (G.node[n]['lon'], G.node[n]['lat']) for n in G}

def get_edge_scalar(G, attr):
    return np.array([G.edge[e[0]][e[1]][attr] for e in G.edges_iter()])

def flow_plot(multi, flow_attr, ax):
    G = multi.layers_as_subgraph(['streets'])
    nx.draw_networkx_edges(G, 
                           get_coords(G),
                           edge_color = 'grey',
                           width = 1,
                           arrows = False,
                           alpha = .2,
                           ax = ax)

    nx.draw_networkx_edges(G, 
                           get_coords(G),
                           edge_color = get_edge_scalar(G, flow_attr)/get_edge_scalar(G, 'capacity'),
                           width = get_edge_scalar(G, flow_attr) * .0003,
                           arrows = False,
                           edge_cmap = plt.get_cmap('plasma'),
                           edge_vmin = 0, 
                           edge_vmax = 1.5,
                           ax = ax)

    if multi.check_layer('metro'):
      G = multi.layers_as_subgraph(['metro'])
      nx.draw_networkx_edges(G, get_coords(G),
              edge_color = 'white', 
              width = get_edge_scalar(G, flow_attr) * .0003,
              node_color = 'white',
              node_size = 0,
              alpha = .4,
              with_labels = False,
              arrows = False,
              ax = ax)

def weighted_hist(ax, measure, weights, label, standardized = False, n = 100, **kwargs):
    
    if standardized:
        mu, sigma = analysis.weighted_avg_and_std(measure, weights)
        hist_data = 1.0 * (measure - mu) / sigma
    else:
        hist_data = measure
    hist = np.histogram(hist_data, weights = weights, normed = True, bins = n)
    x = hist[1][:n] 
    y = hist[0]
    ax.plot(x,y, label = label, **kwargs)