from metro import multiplex as mx
from metro import utility
from metro import ita

import pandas as pd
import numpy as np
import cProfile
import time
import networkx as nx
import pandas as pd

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def main():

    # Read in the multiplex
    m = mx.read_multi()

    # Read OD keyed to m
    m.read_od(layer = 'taz', 
              key = 'taz', 
              od_file = '1_data/taz_od/0_1.txt', 
              sep = " ")

    # compute ITA with no metro
    no_metro_beta = 1000
    ita_iteration(m, beta = no_metro_beta)

    # compute the mean free flow speed v_f and the mean congested speed v_c
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

    with open("true_beta.txt", "w") as text_file:
      text_file.write('The true beta is approximately ' + str(1.0 / ratio))

    

    # main loop: for each beta, scale the metro layer of m, run ITA, and save
    # the results. Note that using summary = True prints out route-wise 
    # summaries for each level of beta, and is VERY computationally expensive. 

    betas = pd.read_csv('betas.csv').beta

    for beta in betas:
      ita_iteration(m, beta)

      # start = time.clock()
      # m.scale_edge_attribute(layer = 'metro',
      #                            attribute = 'free_flow_time_m',
      #                            beta = beta)

      # df = m.run_ita(n_nodes = None, 
      #               summary = True, # change this to get route tables 
      #               attrname = 'congested_time_m_' + str(beta),
      #               flow_name = 'flow_' +str(beta),
      #               P = [.2, .2, .2, .2, .1, .1],
      #               scale = .25)

      # if df is not None:
      #   df.to_csv('3_throughput/route_info_' + str(beta) + '.csv')

      # m.scale_edge_attribute(layer = 'metro',
      #                        attribute = 'free_flow_time_m',
      #                        beta = 1.0/beta)

      # time_taken = str(round((time.clock() - start) / 60.0, 1)) + 'm'
      # print 'assignment for beta = ' + str(beta) + ' completed in ' + time_taken

    m.to_txt('3_throughput/', 'mx_flow')

def ita_iteration(m, beta):
  start = time.clock()
  m.scale_edge_attribute(layer = 'metro',
                         attribute = 'free_flow_time_m',
                         beta = beta)

  df = m.run_ita(n_nodes = None, 
                summary = True, 
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

if __name__ == '__main__':
    main()
