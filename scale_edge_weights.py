from metro import multiplex as mx
import numpy as np
import sys


def main():
	c = float(sys.argv[1])

	multi = mx.read_multi(nodes_file_name = '2_multiplex/multiplex_unscaled_nodes.txt', 
						  edges_file_name = '2_multiplex/multiplex_unscaled_edges.txt')

	weights = ['uniform_time_m', 'free_flow_time_m']

	print 'Means prior to rescaling'
	for weight in weights:
		x = np.array([multi.G.edge[e[0]][e[1]][weight] for e in multi.G.edges_iter()
				  if multi.G.edge[e[0]][e[1]]['layer'] == 'streets'])
		print weight + ': ' + str(round(x.mean(),2))
		
	print '\nMeans after rescaling'
	for weight in weights: 
		multi.scale_edge_attribute(layer = 'streets', attribute = weight, beta = c)
		x = np.array([multi.G.edge[e[0]][e[1]][weight] for e in multi.G.edges_iter()
				  if multi.G.edge[e[0]][e[1]]['layer'] == 'streets'])
		print weight + ': ' + str(round(x.mean(),2))

	multi.to_txt('2_multiplex/', 'mx')

if __name__ == '__main__':
	main()