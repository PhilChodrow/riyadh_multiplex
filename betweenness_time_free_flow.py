
import multiplex as mx
import networkx as nx
import utility as utility
import os

def main():
	directory = '4. figs/betweenness/cost_time_free_flow'
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
	# multi.add_epsilon(weight = 'weight', epsilon = .000001)

	return multi

def analysis(multi, directory):

	betas = [0.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

	for beta in betas:
		multi.scale_edge_attribute(layer = 'metro', attribute = 'weight', beta = beta)
		multi.igraph_betweenness_centrality(layers = multi.get_layers(), weight = 'weight', attrname = str(beta) + 'bc')
		streets = multi.layers_as_subgraph('streets')
		max_bc = max([streets.node[n][str(beta) + 'bc'] for n in streets.node])
		max_bc = '%.1e' % max_bc
		multi.betweenness_plot_interpolated(layer1 = 'streets', 
		                                    layer2 = 'metro', 
		                                    title = 'Riyadh Streets: Betweenness Weighted By cost_time_m\n' +  r'$\beta =' + str(beta) + '$' + '\n max betweenness = ' + max_bc, 
		                                    measure = str(beta) + 'bc', 
		                                    file_name = directory + '/beta' + str(beta) + '.png',
		                                    vmin = None,
		                                    vmax = None)
		multi.scale_edge_attribute(layer = 'metro', attribute = 'weight', beta = 1/beta)

if __name__ == "__main__":
    main()