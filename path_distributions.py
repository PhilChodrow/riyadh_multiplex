import utility
import sys
import multiplex as mx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy import stats 

if len(sys.argv) != 3:
	print 'Sorry, please give two edge weights to compare (e.g. cost_time_m and dist_km)'
else: 	
	directory = '4. figs/shortest_paths' 
	utility.check_directory(directory)
	multi = utility.read_multi()

	measure1 = sys.argv[1]
	measure2 = sys.argv[2]
	multi.add_epsilon(weight = measure1, epsilon = .000001)
	multi.add_epsilon(weight = measure2, epsilon = .000001)
	g = utility.nx_2_igraph(multi.as_graph())

nodes = np.array([n['id'] for n in g.vs.select(lambda vertex: vertex['layer'] == 'taz')])

nodes = np.random.choice(nodes, size = 10, replace = False)

target = g.vs.select(lambda vertex: vertex['id'] in nodes)
source = target


lengths1 = g.shortest_paths_dijkstra(weights = measure1, source = source, target = target) 
lengths2 = g.shortest_paths_dijkstra(weights = measure2, source = source, target = target)

lengths = {(source[i]['id'], target[j]['id']): (lengths1[i][j], lengths2[i][j]) for i in range(len(source)) for j in range(len(target))}

m1 = np.array([tup[0] for tup in lengths.values()])
m2 = np.array([tup[1] for tup in lengths.values()])

d = {measure1 : m1,
	measure2 : m2}

fig1 = plt.figure()

with sns.axes_style("white"):
	h = sns.jointplot(x = measure1, y = measure2, data = d, kind = 'hex')
# h = h.plot_joint(sns.kdeplot, cmap = 'Purples_d')	
# plt.plot([0,100],[0,100 * math.sqrt(2)], 'k-', lw = 1)
# plt.plot([0,100],[0,100], 'k-', lw = 1)
# h = h.plot_marginals(sns.distplot, kde = False, fit = stats.lognorm, color = ".5")
	# h.fig.suptitle(measure2 + ' vs ' + measure1 + ' on the Riyadh Street Network')
	# sns.plt.title(measure2 + ' vs ' + measure1 + ' on the Riyadh Street Network')
	h.set_axis_labels(measure1, measure2)
	sns.plt.savefig(directory + '/' + measure1 + '__' + measure2 + '_joint_density.png')

fig2 = plt.figure()
f = sns.distplot(d[measure1], kde = False, fit = stats.lognorm)
f.set_axis_labels(measure1, 'density')
sns.plt.savefig(directory + '/' + measure1 + '.png')

fig3 = plt.figure()
f = sns.distplot(d[measure2], kde = False, fit = stats.lognorm)
f.set_axis_labels(measure2, 'density')
sns.plt.savefig(directory + '/' + measure2 + '.png')