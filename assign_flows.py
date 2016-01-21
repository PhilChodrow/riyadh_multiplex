from metro import multiplex as mx
from metro import assignment
from metro import utility
from metro import ita

import pandas as pd
import numpy as np
import cProfile
import time
import networkx as nx

multi = mx.read_multi()

# Scale metro so that it runs at the mean street speed. 
dists = nx.get_edge_attributes(multi.G, 'dist_km')
times = nx.get_edge_attributes(multi.G, 'free_flow_time_m')

speed_km_m = {key : dists[key]/times[key] for key in dists}

nx.set_edge_attributes(multi.G, 'speed_km_m', speed_km_m)

street_speed = multi.mean_edge_attr_per(layers = ['streets'], attr = 'speed_km_m', weight_attr = 'dist_km')
metro_speed = multi.mean_edge_attr_per(layers = ['metro'], attr = 'speed_km_m', weight_attr = 'dist_km') 

scale = street_speed / metro_speed

multi.scale_edge_attribute(layer = 'metro', attribute = 'free_flow_time_m', beta = 1/scale)


# Read OD
multi.read_od(layer = 'taz', key = 'taz', od_file = '1_data/taz_od/0_1.txt', sep = " ")

betas = [100, 10, 1, .9, .8, .7, .6, .5, .4, .3, .2, .1, .01]

for beta in betas:

  if beta in [100, 1, .5, .1, .01]:
    summary = True
  else:
    summary = False

  start = time.clock()
  multi.scale_edge_attribute(layer = 'metro',
                             attribute = 'free_flow_time_m',
                             beta = beta)

  df = multi.run_ita(n_nodes = None, 
                summary = summary, # change this to get route tables 
                attrname = 'congested_time_m_' + str(beta),
                flow_name = 'flow_' +str(beta),
                P = [.2, .2, .2, .2, .1, .1],
                scale = .25)

  if df is not None:
    df.to_csv('3_throughput/route_info_' + str(beta) + '.csv')

  multi.scale_edge_attribute(layer = 'metro',
                         attribute = 'free_flow_time_m',
                         beta = 1.0/beta)

  print 'assignment for beta = ' + str(beta) + ' completed in ' + str(round((time.clock() - start) / 60.0, 1)) + 'm'

multi.to_txt('3_throughput/', 'mx_flow')
