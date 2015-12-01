from metro import io
from metro import utility
from metro import assignment
import sys
import networkx as nx


def main():
	if len(sys.argv) <= 1:
		print 'Please choose one of \n homebrew \n igraph \n both'
		return None

	P = [.4, .3, .2, .1]

	attrname = 'congested_time_m'
	second = ""

	if sys.argv[1] == 'both':
		second= '_2'

	multi = io.read_multi(nodes_file_name = '2_multiplex/multiplex_no_traffic_nodes.txt', 
						  edges_file_name = '2_multiplex/multiplex_no_traffic_edges.txt')	

	if sys.argv[1] == 'homebrew' or sys.argv[1] == 'both':
		print 'computing ITA using homebrew method'
		G = multi.as_graph()
		ITA1(multi, 
			 od_loc = '1_data/taz_od/0_1.txt', 
			 attrname = 'congested_time_m',
			 volumeScale = .25,
			 P = P,
			 exclude_layers = ['metro'],
			 base_cost = 'free_flow_time_m')
	
	# igraph method
	if sys.argv[1] == 'igraph' or sys.argv[1] == 'both':
		print 'computing ITA using igraph method, no metro layer'
		ITA2(multi, 
			 layers = ['streets', 'taz'], 
			 od_loc = '1_data/taz_od/0_1.txt',
			 P = P,
			 scale = .25,
			 attrname = 'congested_time_m' + second,
			 flow_name = 'flow')

		print 'computing ITA using igraph method with metro layer'
		ITA2(multi, 
			 layers = ['streets', 'taz', 'metro'], 
			 od_loc = '1_data/taz_od/0_1.txt',
			 P = P,
			 scale = .25,
			 attrname = 'congested_time_metro_m' + second,
			 flow_name = 'flow_metro')

	if sys.argv[1] == 'both':
		ratios = {e : (multi.G.edge[e[0]][e[1]]['congested_time_m_1'] 
			   / multi.G.edge[e[0]][e[1]]['congested_time_m_2']) for e in multi.G.edges_iter()}

		if max(ratios.values()) - min(ratios.values()) < .00001:
			print 'methods agree to within 0.001%'
		else:
			print 'methods do not agree to within 0.001%'

	io.multiplex_to_txt(multi, '2_multiplex/', 'multiplex_unscaled')

def ITA1(multi, od_loc = '1_data/taz_od/0_1.txt', attrname = 'congested_time_m',  **kwargs):
	G = multi.as_graph()
	od = assignment.od_dict(G, '1_data/taz_od/0_1.txt')
	assignment.geo_betweenness_ITA(multi, 
								   OD = od,
								   attrname = attrname,
								   **kwargs)
	
def ITA2(multi, layers = ['streets', 'taz'], od_loc = '1_data/taz_od/0_1.txt', attrname = 'congested_time_m', flow_name = 'flow', **kwargs):
	
	# get the layers we'll be working with as an nx graph
	sublayers = multi.layers_as_subgraph(layers)

	nx.set_edge_attributes(multi.G, attrname, nx.get_edge_attributes(multi.G, 'free_flow_time_m')) 
	
	# ----------------
	# Alternative to above line: sets the 'base' time only for the specified sublayers. Layers of the multiplex not specified will have this attribute equal to None.
	# nx.set_edge_attributes(multi.G, attrname, nx.get_edge_attributes(sublayers, 'free_flow_time_m'))
	
	nx.set_edge_attributes(multi.G, flow_name, 0)

	# Execute the assignment algorithm
	g = utility.nx_2_igraph(sublayers)
	od = assignment.od_dict_igraph(g, od_loc)
	assignment.ITA_igraph(g = g, od = od, attrname = attrname, **kwargs)
	d = {(g.vs[g.es[i].source]['id'], g.vs[g.es[i].target]['id']) : g.es[i][attrname] for i in range(len(g.es))}
	f = {(g.vs[g.es[i].source]['id'], g.vs[g.es[i].target]['id']) : g.es[i]['flow'] for i in range(len(g.es))}
	nx.set_edge_attributes(multi.G, attrname, d)
	nx.set_edge_attributes(multi.G, flow_name, f)

if __name__ == '__main__':
	main()