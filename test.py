
from multiplex import *
import networkx as nx
import utility

print 'Reading metro network.'
metro = utility.read_metro(directory = 'data/metro', file_prefix = 'metro') # networkx graph

print 'Reading street network'
streets = utility.read_streets(directory = 'data/street', file_prefix = 'street') # networkx graph


layer_dict = {'metro' : metro, 'streets' : streets} # need a dict to add to multiplex.

multi = multiplex() # initialize empty multiplex
multi.add_layers(layer_dict) # add metro and street layers to multiplex

E = len(multi.network.edges())

print 'Before spatial join, the layers of the multiplex are ' + str(multi.get_layers()) + '. It has ' + str(E) + ' edges.'


# multi.spatial_join('metro', 'streets', time = 0., distance = 0., both = True)

multi.spatial_join_2(layer1 = 'metro', layer2 = 'streets', transfer_speed = .1, base_cost = 2, both = True)

F = len(multi.network.edges())

print 'After spatial join, the layers of the multiplex are ' + str(multi.get_layers()) + '. It has ' + str(F) + ' edges, of which ' + str(F - E) + ' are transfers. Now we\'ll save the network in a .txt file.' 

multi.to_txt('data/multiplex', 'multi')

