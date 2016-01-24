import pandas as pd
import numpy as np
import time
from collections import defaultdict
import os 

def gradient_component(base, flow, capacity, a, b):
    """
    Summary:
        Compute a component of the gradient on an edge of the network
    
    Args:
        base (float): the base cost of the edge, usually free_flow_time_m
        flow (float): the flow through the edge
        capacity (float): the capacity of the edge, usually capacity
        a (float): BPR parameter
        b (float): BPR parameter
    
    Returns:
        float: the gradient component 
    """
    return BPR(base, flow, capacity, a, b) - base + base * a * b * (1.0 * flow / capacity) ** b

def compute_gradient(base_attr, flow_attr, capacity_attr, a, b, es):
    """
    Summary: 
        Compute the gradient over an entire edge sequence
    
    Args:
        base (float): the base cost of the edge, usually free_flow_time_m
        flow (float): the flow through the edge
        capacity (float): the capacity of the edge, usually capacity
        a (float): BPR parameter
        b (float): BPR parameter
        es (igraph edge sequence): the sequence of edges over which to compute 
    
    Returns:
        list: a list of gradient components (floats) 
    """
    es['gradient'] = [gradient_component(e[base_attr], 
                                         e[flow_attr], 
                                         e[capacity_attr], a, b) for e in es]

def summary_constructor(es, base_cost):
    """
    Summary:
        Construct a function to return routewise summaries over the edge sequence es. 
    
    Args:
        es (igraph edge sequence): the edge sequence containing attributes to summarise 
        base_cost (str): the attribute containing the base cost per edge, usually free_flow_time_m 
    
    Returns:
        function: computes path summaries when supplied with a path consisting of elements of es. 
    """
    def get_path_summary(path):
        edges = eval(path)
        congested_time_m = sum([es[e]['congested_time_m'] for e in edges])
        uniform_time_m = sum([es[e]['uniform_time_m'] for e in edges])
        free_flow_time_m = sum([es[e]['free_flow_time_m'] for e in edges])
        dist_km = sum([es[e]['dist_km'] for e in edges])
        base = sum([es[e][base_cost] for e in edges])
        weighted_capacity = (1.0 * sum([es[e]['dist_km']*es[e]['capacity'] 
                             for e in edges]))
        weighted_flow = (1.0 * sum([es[e]['dist_km']*es[e]['flow'] 
                         for e in edges]))
        gamma = [weighted_flow / weighted_capacity if weighted_capacity > 0 else np.nan][0]
        gradient = (1.0 * sum([es[e]['gradient'] for e in edges]))
        return (congested_time_m, uniform_time_m, free_flow_time_m, dist_km, base, gamma, gradient)
    return get_path_summary

def make_details_df(df, es, base_cost):
    """
    Summary:
        Apply a path summary function over the rows of a dataframe containing paths. 
    
    Args:
        df (pd.DataFrame): contains a column called 'path' containing elements of es 
        es (igraph edge sequence): the edge sequence containing edge attributes 
        base_cost (str): the attribute containing the base cost, usually free_flow_time_m 
    
    Returns:
        pd.DataFrame: a dataframe with summary measures. The path attribute is deleted, since it takes large amounts of memory.  
    """
    get_path_summary = summary_constructor(es, base_cost)
    df['congested_time_m'], df['uniform_time_m'], df['free_flow_time_m'], df['dist_km'], df['base_cost'], df['gamma'], df['gradient'] = zip(*df['path'].map(get_path_summary))
    del df['path']
    return df[['o', 'd', 'p', 'flow', 'dist_km', 'uniform_time_m', 'free_flow_time_m', 'congested_time_m', 'base_cost', 'gamma', 'gradient']]

def agg_df(df):
    """
    Summary:
        Aggregate a dataframe by origin and destination, weighting by flows between them. 
    
    Args:
        df (pandas.DataFrame): the df to aggregate 
    
    Returns:
        pandas.DataFrame: the aggregated df, grouped by 'o' and 'd'.  
    """
    cols = ['flow', 
            'dist_km', 
            'uniform_time_m',
            'free_flow_time_m', 
            'congested_time_m', 
            'base_cost',
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
    """
    Summary:
        Compute the congestion on a segment using the standard BPR function
    
    Args:
        base (float): the base cost of the edge, usually free_flow_time_m
        flow (float): the flow through the edge
        capacity (float): the capacity through the edge, usually capacity
        a (float): BPR parameter
        b (float): BPR parameter
    
    Returns:
        float: the congested travel time through the edge 
    """
    return base * (1 + a * (1.0 * flow / capacity) ** b)

    
def ITA(g, od, base_cost = 'free_flow_time_m', P = [0.4, 0.3, 0.2, 0.1], a = 0.15, b = 4., scale = .25, details = False):
    """
    Summary: 
        Run Iterated Traffic Assignment on a network. 
    
    Args:
        g (igraph.Graph()): the network on which to run ITA 
        od (dict): the OD dictionary, keyed according to vertices of g
        base_cost (str, optional): attribute containing base cost per edge, usually 'free_flow_time_m' 
        P (list, optional): the iterations in which to conduct assignment. Must add to 1. 
        a (float, optional): BPR parameter
        b (float, optional): BPR parameter
        scale (float, optional): the proportion of flow to assign
        details (bool, optional): whether to supply a summary data frame with routewise metrics as a return value. VERY computationally expensive. This function should run in roughly 12-15 minutes if details = False, but closer to 2.5 hours if details = True. 

    Returns:
        df: only if details = True, returns a dataframe summarising route information 
    """
    import time 
    

    columns = ['o', 'd', 'p', 'flow', 'path']
    flow_dict = defaultdict(int)

    es = g.es
    
    es['flow'] = 0
    es['congested_time_m'] = list(es[base_cost])
    
    j = 0
    for p in P: 
        start = time.clock()
        paths_list = pd.DataFrame(columns = columns)
        for o in od:
            ds = od[o]
            if len(ds) > 0:
                targets = ds.keys()
                paths = g.get_shortest_paths(o, 
                                             to=targets, 
                                             weights='congested_time_m', 
                                             mode='OUT', 
                                             output="epath")

                # Update flow dict
                for i in range(len(targets)):
                        flow = od[o][targets[i]]
                        for e in paths[i]:
                            flow_dict[e] += p * scale * flow


                # Update paths list
                if details:
                    update_piece = [{'o' : o, 
                    'd' : targets[i], 
                    'p' : p,
                    'flow' : scale * od[o][targets[i]], 
                    'path' : str(paths[i])} for i in range(len(targets))]
                    update_piece = pd.DataFrame(update_piece)
                    paths_list = paths_list.append(update_piece)
                

        # Assign the flows to the graph
        for key in flow_dict:
            es[key]['flow'] = flow_dict[key]
            es[key]['congested_time_m'] = BPR(base = es[key][base_cost], 
                                    flow = es[key]['flow'], 
                                    capacity = float(es[key]['capacity']),
                                    a = a,
                                    b = b)
        if details:
            paths_list.to_csv('3_throughput/paths_list_' + str(j) + '.csv')
            j += 1
            del paths_list
        time_taken = str(round((time.clock() - start) / 60.0, 1)) + 'm'
        print 'assignment for p = ' + str(p) + ' completed in ' + time_taken
        

    compute_gradient('free_flow_time_m', 'flow', 'capacity', a, b, es)
    
    # Compute details
    if details: 
        df = pd.DataFrame(columns = columns)
        for k in range(len(P)):
            piece = pd.read_csv('3_throughput/paths_list_' + str(k) + '.csv')
            df_piece = make_details_df(piece, es, base_cost)
            df = df.append(piece)
            os.remove('3_throughput/paths_list_' + str(k) + '.csv')
        df = agg_df(df)
        con_map = { v.index : v['con_name'] for v in g.vs}
        df['o_con'] = df.o.map(con_map.get)
        df['d_con'] = df.d.map(con_map.get)
        nx_map = { v.index : v['name'] for v in g.vs}
        df['o_nx'] = df.o.map(nx_map.get)
        df['d_nx'] = df.d.map(nx_map.get)
        del df['o']
        del df['d']
        return df