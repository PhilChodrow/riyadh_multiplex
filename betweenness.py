
import multiplex as mx
import networkx as nx
import utility
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import sys

def main():

	if len(sys.argv) == 1:
		print 'Sorry, please give an edge weight (e.g. cost_time_m or dist_km)' 		
	else:
		measure = str(sys.argv[1])
		print 'Using ' + measure + ' as edge weight.'
		directory = '4. figs/betweenness/' + measure
		utility.check_directory(directory)

		multi = utility.read_multi()
		multi.remove_layer('taz')
		nx.set_edge_attributes(multi.G, 'weight', nx.get_edge_attributes(multi.G, measure))
		multi.add_epsilon(weight = 'weight', epsilon = .000001)

		analysis(multi, directory)
		
def plot(multi, beta, measure, directory):

	fig = plt.figure(figsize = (10,12), dpi = 500)
	gs = gridspec.GridSpec(2,1, height_ratios = [5,1])
	a = fig.add_subplot(gs[0])
	sns.set_style('white')

	title = 'Riyadh Streets: Betweenness Weighted By ' + measure + '\n' +  r'$\beta =' + str(beta) + '$'
	multi.spatial_plot_interpolated(layer1 = 'streets', 
	                                    layer2 = 'metro', 
	                                    title = title,
	                                    measure = measure, 
	                                    file_name = directory + '/beta' + str(beta) + '.png',
	                                    vmin = None,
	                                    vmax = None)

	x = np.array([multi.G.node[n][measure] for n in multi.G.node if multi.G.node[n]['layer'] == 'streets'])
	y = np.array([multi.G.node[n][measure] for n in multi.G.node if multi.G.node[n]['layer'] == 'metro'])

	a = fig.add_subplot(gs[1])
	sns.kdeplot(x, label = 'street', bw = 5000, color = 'grey')
	sns.kdeplot(y, label = 'metro', bw = 100000, color = '#5A0000')
	plt.xlim(0, max(x.max(), y.max()) / 2.0)	
	# a.set(xscale="log", yscale = "log")
	sns.despine(left = True)

	plt.title('Betweenness distribution')
	plt.savefig(directory + '/beta' + str(beta) + '.png')

def analysis(multi, directory):
	betas = [0.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
	for beta in betas:
		measure = str(beta) + 'bc'
		multi.scale_edge_attribute(layer = 'metro', attribute = 'weight', beta = beta)
		multi.igraph_betweenness_centrality(layers = multi.get_layers(), weight = 'weight', attrname = measure)
		plot(multi, beta, measure, directory)
		multi.scale_edge_attribute(layer = 'metro', attribute = 'weight', beta = 1/beta)

if __name__ == "__main__":
    main()