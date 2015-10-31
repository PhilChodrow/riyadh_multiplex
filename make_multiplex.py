import multiplex as mx
import utility as utility

def main():
	metro_dir = '1. data/metro'
	metro_prefix = 'metro'

	street_dir = '1. data/street'
	street_prefix = 'street'

	print 'Reading metro network.'
	metro = utility.read_metro(directory = metro_dir, file_prefix = metro_prefix) # networkx graph

	print 'Reading street network'
	streets = utility.read_streets(directory = street_dir, file_prefix = street_prefix) # networkx graph

	layer_dict = {'metro' : metro, 'streets' : streets} # need a dict to add to multiplex.

	multi = mx.multiplex() # initialize empty multiplex
	multi.add_layers(layer_dict) # add metro and street layers to multiplex
	multi.spatial_join(layer1 = 'metro', layer2 = 'streets', transfer_speed = .1, base_cost = 0, both = True)

	multi.to_txt('2. multiplex', 'multiplex')

if __name__ == "__main__":
    main()