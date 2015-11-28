import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate
import numpy as np
import matplotlib.cm as cm
from metro import utility
import networkx as nx

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
