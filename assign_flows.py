from metro import multiplex as mx
from metro import utility
from metro import ita

import pandas as pd
import numpy as np
import cProfile
import time
import networkx as nx


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def main():

    # Read in the multiplex
    m = mx.read_multi()

    # Scale metro so that it runs at the mean street speed. 
    dists = nx.get_edge_attributes(m.G, 'dist_km')
    times = nx.get_edge_attributes(m.G, 'free_flow_time_m')

    speed_km_m = {key : dists[key]/times[key] for key in dists}

    nx.set_edge_attributes(m.G, 'speed_km_m', speed_km_m)

    street_speed = m.mean_edge_attr_per(layers = ['streets'], 
                                        attr = 'speed_km_m', 
                                        weight_attr = 'dist_km')

    metro_speed = m.mean_edge_attr_per(layers = ['metro'], 
                                       attr = 'speed_km_m', 
                                       weight_attr = 'dist_km') 

    scale = street_speed / metro_speed

    m.scale_edge_attribute(layer = 'metro', 
                           attribute = 'free_flow_time_m', 
                           beta = 1/scale)

    # Read OD keyed to m
    m.read_od(layer = 'taz', 
              key = 'taz', 
              od_file = '1_data/taz_od/0_1.txt', 
              sep = " ")

    # main loop: for each beta, scale the metro layer of m, run ITA, and save
    # the results. Note that using summary = True prints out route-wise 
    # summaries for each level of beta, and is VERY computationally expensive. 
    betas = [100, 10, 1, .9, .8, .7, .6, .5, .4, .3, .2, .1, .01]

    for beta in betas:

      start = time.clock()
      m.scale_edge_attribute(layer = 'metro',
                                 attribute = 'free_flow_time_m',
                                 beta = beta)

      df = m.run_ita(n_nodes = None, 
                    summary = True, # change this to get route tables 
                    attrname = 'congested_time_m_' + str(beta),
                    flow_name = 'flow_' +str(beta),
                    P = [.2, .2, .2, .2, .1, .1],
                    scale = .25)

      if df is not None:
        df.to_csv('3_throughput/route_info_' + str(beta) + '.csv')

      m.scale_edge_attribute(layer = 'metro',
                             attribute = 'free_flow_time_m',
                             beta = 1.0/beta)

      time_taken = str(round((time.clock() - start) / 60.0, 1)) + 'm'
      print 'assignment for beta = ' + str(beta) + ' completed in ' + time_taken

    m.to_txt('3_throughput/', 'mx_flow')

if __name__ = '__main__':
    main()
