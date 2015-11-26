import multiplex as mx
import networkx as nx
import utility
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab

def main():
	directory = '4. figs/intermodality/global'
	multi = prep(directory)
	analysis(multi, directory)


def prep(directory):

	multi = utility.multiplex_from_txt(nodes_file_name = '2. multiplex/multiplex_nodes.txt',
	                                   edges_file_name = '2. multiplex/multiplex_edges.txt',
	                                   sep = '\t',
	                                   nid = 'id',
	                                   eidfrom = 'source',
	                                   eidto = 'target')

	if not os.path.exists(directory):
		os.makedirs(directory)

	nx.set_edge_attributes(multi.G, 'weight', nx.get_edge_attributes(multi.G, 'cost_time_m'))
	
	multi.add_epsilon(weight = 'weight', epsilon = .000001)

	return multi

def analysis(multi, directory):
	n_nodes = 20
	betas = [.01, 0.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
	results = {}

	for beta in betas:
		multi.scale_edge_attribute(layer = 'metro', attribute = 'weight', beta = beta)

		speed_ratio =  multi.mean_edge_attr_per(layers = ['streets'], attr = 'weight', weight_attr = 'dist_km') / multi.mean_edge_attr_per(layers = ['metro'], attr = 'weight', weight_attr = 'dist_km') 

		source = multi.random_nodes_in(layers = ['streets'], n_nodes = n_nodes)
		target = multi.random_nodes_in(layers = ['streets'], n_nodes = n_nodes)

		street_only = 0
		intermodal = 0

		for n in source: 
			for m in target:
				try:
					layers = multi.layers_of(multi.shortest_path(source = n, target = m, weight = 'weight'))
				except nx.exception.NetworkXNoPath:
					pass

				if len(layers) == 1:
					street_only += 1
				else:
					intermodal += 1
		
		intermodality = intermodal*1.0 / (intermodal + street_only)

		multi.scale_edge_attribute(layer = 'metro', attribute = 'weight', beta = 1/beta)

		results[beta] = (speed_ratio, intermodality)

	speed_ratios = np.array([results[beta][0] for beta in betas])
	intermodality = np.array([results[beta][1] for beta in betas])

	fig = plt.figure(figsize = (10,10), dpi = 300)
	fig.suptitle('Riyadh Metro')

	ax = fig.add_subplot(111)
	ax.set_title('Intermodality and Relative Metro Speed')
	ax.set_xlabel('Relative speed of metro to free-flow streets')
	ax.set_ylabel('Percent of shortest paths through metro')
	ax.plot(speed_ratios, intermodality)
	ax.scatter(speed_ratios, intermodality)
	pylab.ylim(0,1)

	plt.savefig(directory + '/intermodality_profile')


if __name__ == "__main__":
    main()