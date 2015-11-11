
import sys
import utility
import multiplex as mx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy import stats 
import pandas as pd
import math

def main():
	if len(sys.argv) != 3:
		print 'Sorry, please give two edge weights to compare (e.g. cost_time_m and dist_km)'
	else:
		measure1 = sys.argv[1]
		measure2 = sys.argv[2] 	
		directory = '4. figs/shortest_paths' 
		utility.check_directory(directory)
		multi = prep(measure1, measure2)
		d = calc_distribution(multi, measure1, measure2, layer = 'taz')
		plot1(d, measure1, measure2)
		plot2(d, measure1, measure2)

def prep(measure1, measure2):
	multi = utility.read_multi()
	
	multi.add_epsilon(weight = measure1, epsilon = .000001)
	multi.add_epsilon(weight = measure2, epsilon = .000001)
	return multi

def calc_distribution(multi, measure1, measure2, layer):
	print 'converting multiplex to igraph'
	g = utility.nx_2_igraph(multi.as_graph())
	nodes = np.array([n['id'] for n in g.vs.select(lambda v: v['layer'] == layer)])

	target = g.vs.select(lambda v: v['id'] in nodes)
	source = target
	print 'computing shortest paths -- this could take a while'
	lengths1 = g.shortest_paths_dijkstra(weights = measure1, source = source, target = target) 
	lengths2 = g.shortest_paths_dijkstra(weights = measure2, source = source, target = target)
	lengths = {(source[i]['id'], target[j]['id']): (lengths1[i][j], lengths2[i][j]) for i in range(len(source)) for j in range(len(target))}
	m1 = np.array([tup[0] for tup in lengths.values()])
	m2 = np.array([tup[1] for tup in lengths.values()])
	d = {measure1 : m1,
		 measure2 : m2}
	d = pd.DataFrame(d)
	return d

def plot1(d, measure1, measure2):
	fig1 = plt.figure(figsize = (10,12), dpi = 500)

	with sns.axes_style("white"):
		h = sns.JointGrid(x = measure1, y = measure2, data = d, xlim=(0, 40), ylim=(0, 40))
		h = h.plot_joint(plt.hexbin, cmap = 'Purples', gridsize = 300)
		h = h.plot_marginals(sns.distplot, kde = True, color = ".5")
		h.set_axis_labels(measure1, measure2)
		sns.plt.savefig(directory + '/' + measure1 + '__' + measure2 + '_joint_density.png')

def plot2(d, measure1, measure2):
	fig2 = plt.figure()
	f = sns.distplot(d[measure1], kde = True)
	f.set(xlim = (0, None))
	sns.plt.savefig(directory + '/' + measure1 + '.png')

	fig3 = plt.figure()
	f = sns.distplot(d[measure2], kde = True)	
	f.set(xlim = (0, None))
	sns.plt.savefig(directory + '/' + measure2 + '.png')

if __name__ == "__main__":
    main()