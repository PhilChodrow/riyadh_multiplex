import utility
import pandas as pd
import numpy as np
import itertools
import time
import collections
import networkx as nx

def main():
	start = time.clock()
	multi = utility.read_multi()
	G = multi.layers_as_subgraph(['taz', 'streets'])

	print 'converting to igraph format'
	g = utility.nx_2_igraph(G)
	
	print 'computing od_dict'
	od = od_dict_igraph(g, od_data_dir = '1. data/taz_od/', od_file_name = '0_1.txt', nx_keys = True)
	
	sub_keys = od.keys()[:10]
	sub_od = {k : od[k] for k in sub_keys}
	
	print 'beginning assignment'
	ITA(g, sub_od)
	
	# grab the new congested times, write them to multi, and save to .txt
	d = {(g.vs[g.∏es[i].source]['id'], g.vs[g.es[i].target]['id']) : g.es[i]['congested_time_m'] for i in range(len(g.es))}
	nx.set_edge_attributes(multi.G, 'congested_time_m', d)
	multi.to_txt('2. multiplex', 'test')
	print 'script completed in ' + str((time.clock() - start) / 60.0) + ' m'


def read_od(data_dir, file_name):
    print 'reading OD data'
    data_dir = data_dir
    file_name = file_name
    od = pd.read_table(data_dir + file_name, sep = " ")
    return od

def od_dict_igraph(g, od_data_dir, od_file_name, nx_keys = False):
    '''
    Figure out how much of this function generalizes to work for networkx keys as well,
    would be cool to use the same one for both igraph keys and for Zeyad's networkx keys. 
    ''' 
    start = time.clock()
    # compute 'base' of origin-destination pairs -- we lookup information onto the base
    taz_vs = g.vs.select(lambda v : v['layer'] == 'taz')
    taz_indices = [v.index for v in taz_vs]
    o = [p[0] for p in itertools.product(taz_indices, taz_indices)] 
    d = [p[1] for p in itertools.product(taz_indices, taz_indices)]
    df = pd.DataFrame({'o' : o, 'd' : d})
    
    # lookup the taz corresponding to the origin and the destination
    taz_lookup = {v.index : int(v['taz']) for v in taz_vs}
    df['o_taz'] = df['o'].map(taz_lookup.get) 
    df['d_taz'] = df['d'].map(taz_lookup.get)
    
    # add flow by taz
    od = read_od(data_dir = od_data_dir, file_name = od_file_name)
    od.rename(columns = {'o' : 'o_taz', 'd' : 'd_taz'}, inplace = True)
    df = df.merge(od, left_on = ['o_taz', 'd_taz'], right_on = ['o_taz', 'd_taz'])
    
    # compute normalizer
    taz_norms = df.groupby(['o_taz','d_taz']).size()
    taz_norms = pd.DataFrame(taz_norms)
    taz_norms.rename(columns = {0 : 'taz_norm'}, inplace = True)
    
    # merge normalizer into df and compute normed flows
    df = df.merge(taz_norms, left_on = ['o_taz', 'd_taz'], right_index = True)
    df['flow_norm'] = df['flow'] / df['taz_norm']

    if nx_keys:
    	key_lookup = {v.index : v['id'] for v in taz_vs}
    	df['o'] = df['o'].map(key_lookup.get)
    	df['d'] = df['d'].map(key_lookup.get)

    print df.head()

    Pivot -- makes for an easier dict comprehension
    od_matrix = df.pivot(index = 'o', columns = 'd', values = 'flow_norm')
    od_matrix[np.isnan(od_matrix)] = 0
    
    od = {i : {col : od_matrix[col][i] for col in od_matrix.columns if od_matrix[col][i] > 0.00001} for i in od_matrix.index}
    
    print 'OD dict computed in ' + str((time.clock() - start) / 60.0) + ' m'

    return od_matrix

def ITA(g, od, base_cost = 'cost_time_m', P = (0.4, 0.3, 0.2, 0.1), a = 0.15, b = 4., scale = .25, attrname = 'congested_time_m'):
    vs = g.vs
    es = g.es
    
    vs['flow'] = 0
    es['flow'] = 0
    es[attrname] = list(es[base_cost])
    
    edge_dict = collections.defaultdict(int)
    for p in P:
    	start = time.clock()
        print 'assigning for p = ' + str(p)
        for o in od:
            ds = od[o]
            if len(ds) > 0:
                targets = ds.keys()
                paths = g.get_shortest_paths(o, to=targets, weights=attrname, mode='OUT', output="vpath")
                # print paths
                for path in paths:
                    l = len(path)
                    if l > 0:
                        d = path[-1:][0]
                        flow = od[o][d]
                        for i in range(l):
                            vs[path[i]]['flow'] += p * scale * flow
                            if i < l - 1:
                                edge_dict[g.get_eid(path[i], path[i+1])] += p * scale * flow
        print 'assignment for p = ' + str(p) + ' completed in ' + str((time.clock() - start) / 60.0) + 'm'
        for key in edge_dict:
            es[key]['flow'] = edge_dict[key]
            es[key][attrname] = es[key][base_cost] * (1 + a*( es[key]['flow'] / float(es[key]['capacity']))**b)

# -------------------------------


def main2():
	od = read_od()
	multi = utility.read_multi()
	od_dict = make_od_dict(multi = multi, od_df = od, layer = 'taz')
	print len(od_dict)
	print len(od_dict['taz_5233'])

	nx.set_edge_attributes(multi.G, 'free_flow_time', nx.get_edge_attributes(multi.G, 'cost_time_m'))

	# x = multi.geo_betweenness_ITA(volumeScale = .25, OD = od_dict)

def make_od_dict_nx(multi, od_df, layer = 'taz'):

	G = multi.layers_as_subgraph(layers = [layer])

	taz_nodes = [n for n in G.node]
	taz = [int(G.node[n]['taz']) for n in G.node]
	taz_df = pd.DataFrame({'node' : taz_nodes, 'taz' : taz})

	o = [p[0] for p in itertools.product(taz_nodes, taz_nodes)] 
	d = [p[1] for p in itertools.product(taz_nodes, taz_nodes)]
	df = pd.DataFrame({'o' : o, 'd' : d})

	df = df.merge(taz_df, left_on = 'o', right_on = 'node')
	df.rename(columns={'taz': 'o_taz'}, inplace=True)
	del df['node']

	df = df.merge(taz_df, left_on = 'd', right_on = 'node')
	df.rename(columns={'taz': 'd_taz'}, inplace=True)
	del df['node']

	df = df.merge(od_df, left_on = ['o_taz', 'd_taz'], right_on = ['o','d'], how = 'inner')
	df.rename(columns={'d_x': 'd', 'o_x' : 'o'}, inplace=True)
	del df['o_y']
	del df['d_y']
	df.head()

	taz_nums = df.groupby(['o_taz','d_taz']).size()
	taz_nums = pd.DataFrame(taz_nums)
	taz_nums = taz_nums.rename(columns = {0 : 'num'} )
	df = df.merge(right = taz_nums, left_on = ['o_taz', 'd_taz'], right_index = True, how = 'inner')
	df['flow_norm'] = df['flow'] / df['num']

	
	df = df.pivot(index = 'o', columns = 'd', values = 'flow_norm')
	print df.head()
	od_dict = {i : {col : df[col][i] for col in df.columns} for i in df.index}

	return od_dict

if __name__ == '__main__':
	main()