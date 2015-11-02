import multiplex as mx
import networkx as nx
import utility
import os
import numpy as np
import matplotlib.pyplot as plt
import pylab
import matplotlib.cm as cm

# Should be cleaned up, the visualization function is very messy and duplicative of some functionality in multiplex. Should maybe farm out these visualization functions (idw) into the utility module. git

def main():

	directory = '4. figs/intermodality/local'
	multi = prep(directory)
	analysis(multi, beta = .3, weight = 'weight')
	plot(multi, '4. figs/intermodality/local/intermodality.png', r'Riyadh TAZ Intermodality: $\beta = .3$')

def prep(directory):

	multi = utility.multiplex_from_txt(nodes_file_name = '2. multiplex/multiplex_nodes.txt',
	                                   edges_file_name = '2. multiplex/multiplex_edges.txt',
	                                   sep = '\t',
	                                   nid = 'id',
	                                   eidfrom = 'source',
	                                   eidto = 'target')


	# multi = utility.multiplex_from_txt(nodes_file_name = '3. throughput/local_intermodality/multiplex_nodes.txt',
	# 								   edges_file_name = '3. throughput/local_intermodality/multiplex_edges.txt',
	# 								   sep = '\t',
	# 								   nid = 'id',
	# 								   eidfrom = 'source',
	# 								   eidto = 'target')

	if not os.path.exists(directory):
		os.makedirs(directory)

	nx.set_edge_attributes(multi.G, 'weight', nx.get_edge_attributes(multi.G, 'cost_time_m'))
	multi.add_epsilon(weight = 'weight', epsilon = .000001)

	return multi


def analysis(multi, beta, weight = 'weight'):
	multi.scale_edge_attribute(layer = 'metro', attribute = 'weight', beta = beta)
	multi.local_intermodality(layer = 'taz', thru_layer = 'metro', weight = 'weight')

	if not os.path.exists('3. throughput/local_intermodality'):
		os.makedirs('3. throughput/local_intermodality')

	multi.to_txt('3. throughput/local_intermodality', 'multiplex')


def plot(multi, file_name, title):
	def distance_matrix(x0, y0, x1, y1):
		obs = np.vstack((x0, y0)).T
		interp = np.vstack((x1, y1)).T

		# Make a distance matrix between pairwise observations
		# Note: from <http://stackoverflow.com/questions/1871536>
		# (Yay for ufuncs!)
		d0 = np.subtract.outer(obs[:,0], interp[:,0])
		d1 = np.subtract.outer(obs[:,1], interp[:,1])

		return np.hypot(d0, d1)

	def simple_idw(x, y, z, xi, yi, threshhold):
		dist = distance_matrix(x,y, xi,yi)

		# In IDW, weights are 1 / distance
		weights = 1.0 / dist

		# Make weights sum to one
		weights /= weights.sum(axis=0)

		# Multiply the weights for each interpolated point by all observed Z-values
		zi = np.dot(weights.T, z)
		gap = zi[dist.min(axis = 0) > threshhold].max()
		zi[dist.min(axis = 0) > threshhold] = 0
		zi = zi - gap
		zi[zi < 0] = 0
		return zi

	def plot_idw(x,y,z,grid):
	#	plt.figure(figsize = (15,15), dpi = 500)
		plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()), cmap=cm.Blues)
		plt.hold(True)
		# plt.colorbar()

	
	plt.figure(figsize = (15,15), dpi = 500)
	taz = multi.layers_as_subgraph('taz')
	metro = multi.layers_as_subgraph('metro')
	streets = multi.layers_as_subgraph('streets')

	x = np.array([taz.node[n]['pos'][1] for n in taz.node])
	y = np.array([- taz.node[n]['pos'][0] for n in taz.node])
	z = np.array([float(taz.node[n]['intermodality']) for n in taz.node])
	print x.min(), x.max()
	print y.min(), y.max()

	mx, my = 100, 100
	xi = np.linspace(x.min(), x.max(), mx)
	yi = np.linspace(y.min(), y.max(), my)

	xi, yi = np.meshgrid(xi, yi)
	xi, yi = xi.flatten(), yi.flatten()

	# Calculate IDW
	grid1 = simple_idw(x,y,z,xi,yi, threshhold = .2)
	grid1 = grid1.reshape((my, mx))

	plot_idw(x,y,z,grid1)
	
	taz.position = {n : (taz.node[n]['pos'][1], - taz.node[n]['pos'][0]) for n in taz}
	taz.intermodality = [float(taz.node[n]['intermodality']) for n in taz]
	metro.position = {n : (metro.node[n]['pos'][1], - metro.node[n]['pos'][0]) for n in metro}
	streets.position = {n : (streets.node[n]['pos'][1], - streets.node[n]['pos'][0]) for n in streets}

	nx.draw(taz,taz.position,
		node_size = 0,
		node_color = taz.intermodality,
		linewidths = 0,
		alpha = .2,
		with_labels=False,
		cmap=cm.Blues,
		arrows = False)

	nx.draw(streets,streets.position,
		edge_color = 'grey',
		edge_size = .01,
		node_size = 0,
		node_color = '#003399',
		linewidths = 0,
		alpha = .2,
		with_labels=False,
		arrows = False)

	nx.draw(metro, 
		metro.position,
		edge_color = '#5A0000',
		edge_size = 60,
		node_size = 0,
		arrows = False,
		with_labels = False)

	plt.title(title)

	plt.savefig(file_name)

if __name__ == "__main__":
	main()



