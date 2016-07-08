from metro import multiplex as mx
from metro import io            # for reading multiplex
from metro import utility       # for manipulating multiplex
from metro import analysis      # analytical functions
from metro import viz           # for bubble_plot()
from metro import assignment    # for reading and manipulating OD data
from metro import ita
import networkx as nx           # assigning attributes to multiplex
import pandas as pd
from copy import deepcopy
import assign_flows

def od_total(od):
		return np.sum(np.sum(od[o].values()) for o in od)

def simulate(multi, beta, n):
		con_map = {int(multi.G.node[n]['con_name']) : n for n in multi.G.node if multi.G.node[n]['layer'] == 'taz'}
		
		df = pd.read_csv('3_throughput/route_info_' + str(beta) + '.csv')
		df['congestion_impact'] = df['gradient'] * df['flow']
		df = df.sort_values('congestion_impact', ascending = False)

		sub = df.head(n)
		ratio = 1 - sub.flow.sum() / df.flow.sum() # % of flow we are targeting here

		od = deepcopy(multi.od)
		
		# Targeted removal 
		for o, d in zip(sub.o_con, sub.d_con):
				multi.od[con_map[o]][con_map[d]] = 0
		


		df = multi.run_ita(n_nodes = None, 
						   summary = True, # change this to get route tables 
						   attrname = 'congested_time_m_TEST',
						   flow_name = 'flow_TEST',
						   P = [.2, .2, .2, .2, .1, .1],
						   scale = .25)
		
		df.to_csv('3_throughput/targeted_' + str(beta) + '.csv')
		
		
		# Reset the OD
		multi.od = od
		
		# uniform removal
		for o in multi.od:
				for d in multi.od[o]:
						multi.od[o][d] *= ratio
		
		df = multi.run_ita(n_nodes = None, 
						   summary = True, # change this to get route tables 
						   attrname = 'congested_time_m_RAND',
						   flow_name = 'flow_RAND',
						   P = [.2, .2, .2, .2, .1, .1],
						   scale = .25)
		
		df.to_csv('3_throughput/uniform_' + str(beta) + '.csv')
		
		# Reset the OD again, so we can clean up. 
		multi.od = od


def main():

	no_metro_beta = 1000
	betas = pd.read_csv('plot_betas.csv').beta

	m = mx.read_multi(nodes_file_name = '3_throughput/mx_flow_nodes.txt', 
						  edges_file_name = '3_throughput/mx_flow_edges.txt')

	m.read_od(layer = 'taz', # keys are in this layer
				  key = 'taz', # this is the key attribute
				  od_file = '1_data/taz_od/0_1.txt', # here's where the file lives
				  sep = ' ') # this is what separates entries

	mean_free_flow_time = m.mean_edge_attr_per(layers = ['streets'],
							   attr = 'free_flow_time_m',
							   weight_attr = 'flow_' + str(no_metro_beta))
	
	mean_congested_time = m.mean_edge_attr_per(layers = ['streets'],
											   attr = 'congested_time_m_' + str(no_metro_beta),
											   weight_attr = 'flow_' + str(no_metro_beta))

	dist = m.mean_edge_attr_per(layers = ['streets'],
							   attr = 'dist_km',
							   weight_attr = 'flow_' + str(no_metro_beta))

	v_f = dist * 1.0 / mean_free_flow_time 
	v_c = dist * 1.0 / mean_congested_time 

	print 'Mean free flow speed = ' + str(round(v_f,2)) + 'km per minute'
	print 'Mean congested speed = ' + str(round(v_c,2)) + 'km per minute'
	# Scale metro so that it runs at v_c (this is beta = 1)

	mean_metro_time = m.mean_edge_attr_per(layers = ['metro'],
										   attr = 'free_flow_time_m')

	mean_metro_dist = m.mean_edge_attr_per(layers = ['metro'],
										   attr = 'dist_km')

	v_m = mean_metro_dist / mean_metro_time 
	
	print 'Mean metro speed = ' + str(round(v_m,2)) + 'km per minute'

	ratio = v_m / v_c  
	
	print 'Scaling metro speed by factor of ' + str(round(1 / ratio,2))
	m.scale_edge_attribute(layer = 'metro',
						   attribute = 'free_flow_time_m',
						   beta = ratio)

	print 'Metro now runs at mean speed of street layer'

	# run
	
	for beta in betas:
			m.scale_edge_attribute(layer = 'metro',
									   attribute = 'free_flow_time_m',
									   beta = beta)
			
			simulate(m, beta = beta, n = 50000)
			
			m.scale_edge_attribute(layer = 'metro',
									   attribute = 'free_flow_time_m',
									   beta = 1.0/beta)

if __name__ == '__main__':
	main()
