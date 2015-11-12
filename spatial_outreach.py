
import sys
import utility
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def main():
	if len(sys.argv) != 3:
		print 'Sorry, please give an edge weight to measure distance (e.g. cost_time_m or dist_km) and a max cost.' 
	else:
		# params
		weight = sys.argv[1]
		cost = sys.argv[2]
		directory = '4. figs/outreach/' + weight
		layer = 'taz'
		betas = [.2, 10]

		# prep
		utility.check_directory(directory)
		multi = utility.read_multi()
		nx.set_edge_attributes(multi.G, 'weight', nx.get_edge_attributes(multi.G, weight))
		compute_outreach(multi = multi, 
		                 layer = 'taz',
						 weight = 'cost_time_m',
						 cost = 30,
						 betas = betas)
		
		# First plot
		df = get_df(multi = multi, 
		            layer = layer, 
					betas = betas, 
					cost = cost)
		pairs_plot(df, directory)
		
		# Second plot
		vmin, vmax = get_vlims(df, mult = .95)
		map_plots(multi = multi, 
		          betas = betas, 
				  vmin = vmin, 
				  vmax = vmax, 
				  directory = directory,
				  cost = cost)

# primary computation
def compute_outreach(multi = None, layer = 'taz', weight = 'cost_time_m', cost = 30, betas = [1.0]):
	for beta in betas:
		m = str(beta) + weight
		multi.scale_edge_attribute(layer = 'metro', attribute = 'weight', beta = beta)
		multi.spatial_outreach(layer = layer, 
						   weight = 'weight', 
						   cost = cost, 
						   attrname = str(beta) + '_outreach_' + str(cost))
		multi.scale_edge_attribute(layer = 'metro', attribute = 'weight', beta = 1/beta)

# 1st plot
def get_df(multi = None, layer = 'taz', betas = [1.0], cost = None):

	df = utility.nodes_2_df(multi.as_graph())
	df = df[df['layer'] == layer]
	cols = [str(beta) + '_outreach_' + str(cost) for beta in betas]
	cols.append('proximity_to_metro')
	df = df[cols]

	# filter out small values -- might not be the best idea, but it makes the plots much more readable. 
	quantiles = df.quantile(0.01)
	for col in cols: 
		df = df[df[col] > quantiles[col]]

	return df

def pairs_plot(df, directory):
	g = sns.PairGrid(df)
	g = g.map_upper(plt.scatter, alpha = .02)
	g = g.map_lower(sns.kdeplot, cmap = 'Blues', shade = True, n_levels=20, reverse = False)
	g = g.map_diag(sns.kdeplot, lw = 3, shade = True)
	
	plt.savefig(directory + '/pairs.png')

def get_vlims(df = None, mult = 0.95):
	del df['proximity_to_metro']
	vmin = min([df[col].quantile(0.01) for col in df])*mult
	vmax = max([df[col].quantile(0.01) for col in df])*1/mult
	return vmin, vmax

# 2nd plot

def map_plot(multi, measure, vmin, vmax, cost):
	taz = multi.layers_as_subgraph('taz')
	metro = multi.layers_as_subgraph('metro')
	streets = multi.layers_as_subgraph('streets')
   
	taz.weight= [taz.node[n][measure] for n in taz.node]
	
	taz.position = {n : (taz.node[n]['pos'][1], taz.node[n]['pos'][0]) for n in taz}
	metro.position = {n : (metro.node[n]['pos'][1], metro.node[n]['pos'][0]) for n in metro}
	streets.position = {n : (streets.node[n]['pos'][1], streets.node[n]['pos'][0]) for n in streets}

	gini = utility.gini_coeff(np.array(taz.weight))

	nx.draw(taz,
		   taz.position,
		   node_color = taz.weight,
		   cmap = 'jet',
		   node_size = 30,
		   alpha = .2,
		   linewidths = 0,
		   # vmin = vmin,
		   # vmax = vmax,
		   edge_color = None,)

	nx.draw(streets, streets.position,
		   edge_color = 'grey', 
		   edge_size = 0.01,
		   node_color = 'black',
		   node_size = 0,
		   alpha = .2,
		   with_labels = False,
		   arrows = False)

	nx.draw(metro, 
		metro.position,
		edge_color = '#5A0000',
		edge_size = 60,
		node_size = 0,
		arrows = False,
		with_labels = False)
	plt.title(str(measure) + '\nGini: ' + str(round(gini,2)))

def map_plots(multi, betas, vmin, vmax, directory, cost = 30): 
	n = len(betas)
	fig = plt.figure(figsize = (12*n,8), dpi = 500)

	for i in range(n):
		ax = fig.add_subplot(1,n,i + 1)
		map_plot(multi, str(betas[i]) + '_outreach_' + str(cost), vmin, vmax, cost = cost)
	plt.savefig(directory + '/map.png')

if __name__ == "__main__":
	main()
