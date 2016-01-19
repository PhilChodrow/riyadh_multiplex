import pandas as pd
import numpy as np
import time


def gradient_component(base, flow, capacity, a, b):
    return BPR(base, flow, capacity, a, b) - base + base * a * b * (1.0 * flow / capacity) ** b

def compute_gradient(base_attr, flow_attr, capacity_attr, a, b, es):
    es['gradient'] = [gradient_component(e[base_attr], e[flow_attr], e[capacity_attr], a, b) for e in es]

def summary_constructor(es):
    def get_path_summary(path):
        congested_time_m = sum([es[e]['congested_time_m'] for e in path])
        free_flow_time_m = sum([es[e]['free_flow_time_m'] for e in path])
        dist_km = sum([es[e]['dist_km'] for e in path])
        weighted_capacity = (1.0 * sum([es[e]['dist_km']*es[e]['capacity'] for e in path]))
        weighted_flow = (1.0 * sum([es[e]['dist_km']*es[e]['flow'] for e in path]))
        gamma = [weighted_flow / weighted_capacity if weighted_capacity > 0 else np.nan][0]
        gradient = (1.0 * sum([es[e]['gradient'] for e in path]))
        return congested_time_m, free_flow_time_m, dist_km, gamma, gradient
    return get_path_summary



def update_paths_list(paths_list, o, targets, od, paths,p):
    paths_update = [{'o' : o, 
             'd' : targets[i], 
             'p' : p,
             'flow' : od[o][targets[i]], 
             'path' : paths[i]} for i in range(len(targets))]
    paths_list += paths_update            
    
def make_details_df(paths_list, es):
    df = pd.DataFrame(paths_list)
    get_path_summary = summary_constructor(es)
    df['congested_time_m'], df['free_flow_time_m'], df['dist_km'], df['gamma'], df['gradient'] = zip(*df['path'].map(get_path_summary))
    del df['path']
    return df[['o', 'd', 'p', 'flow', 'dist_km', 'free_flow_time_m', 'congested_time_m', 'gamma', 'gradient']]

def agg_df(df):
    cols = ['flow', 
            'dist_km', 
            'free_flow_time_m', 
            'congested_time_m', 
            'gamma', 
            'gradient']
    for col in cols:
        df[col] = df[col] * df.p
    f = {col : sum for col in cols}
    
    agged = df.groupby(["o", "d"]).agg(f)
    
    agged['o'] = agged.index.get_level_values('o')
    agged['d'] = agged.index.get_level_values('d')    
    
    return agged

# ------------------------------------------------------------------------------------------------------------------------------
def BPR(base, flow, capacity, a, b):
    return base * (1 + a * (1.0 * flow / capacity) ** b)

def update_edges(es, flow_dict, base_cost, a, b):
            '''
            Modifies es in place
            '''
            for key in flow_dict:
                es[key]['flow'] = flow_dict[key]
                es[key]['congested_time_m'] = BPR(base = es[key][base_cost], 
                                        flow = es[key]['flow'], 
                                        capacity = float(es[key]['capacity']),
                                        a = a,
                                        b = b)

def update_flow_dict(od, o, targets, p, scale, paths, flow_dict):
    for i in range(len(targets)):
        flow = od[o][targets[i]]
        for e in paths[i]:
            flow_dict[e] += p * scale * flow


def ITA_iteration(g, od, p, scale, details, flow_dict, es, paths_list, base_cost, a, b):            
    start = time.clock()
    for o in od:
        ds = od[o]
        if len(ds) > 0:
            targets = ds.keys()
            paths = g.get_shortest_paths(o, 
                                         to=targets, 
                                         weights='congested_time_m', 
                                         mode='OUT', 
                                         output="epath")
            update_flow_dict(od, o, targets, p, scale, paths, flow_dict)

            if details:
                update_paths_list(paths_list, o, targets, od, paths, p)

    print 'assignment for p = ' + str(p) + ' completed in ' + str(round((time.clock() - start) / 60.0, 1)) + 'm'
    update_edges(es, flow_dict, base_cost, a, b)

def ITA(g, od, base_cost = 'free_flow_time_m', P = [0.4, 0.3, 0.2, 0.1], a = 0.15, b = 4., scale = .25, details = False):
    
    from collections import defaultdict
    import time 
    
    start = time.clock()
    
    paths_list = []
    
    es = g.es
    
    es['flow'] = 0
    es['congested_time_m'] = list(es[base_cost])

    flow_dict = defaultdict(int)

    for p in P: 
        ITA_iteration(g, od, p, scale,  details, flow_dict, es, paths_list, base_cost, a, b)
        
    compute_gradient('free_flow_time_m', 'flow', 'capacity', a, b, es)
    
    # Compute details
    if details:
        df = make_details_df(paths_list, es)
        df = agg_df(df)
        con_map = { v.index : v['con_name'] for v in g.vs}
        df['o_con'] = df.o.map(con_map.get)
        df['d_con'] = df.d.map(con_map.get)
        nx_map = { v.index : v['name'] for v in g.vs}
        df['o_nx'] = df.o.map(nx_map.get)
        df['d_nx'] = df.d.map(nx_map.get)
        return df