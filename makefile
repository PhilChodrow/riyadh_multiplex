all: 2_multiplex/multiplex_nodes.txt methods/methods.pdf

clean: 
	rm -rf 4.\ figs/betweenness
	rm 2.\ multiplex/multiplex_edges.txt 2.\ multiplex/multiplex_nodes.txt
	rm \methods/methods.pdf 

2_multiplex/multiplex_unscaled_nodes.txt: 1_data/street/street_nodes.txt 1_data/street/street_edges.txt 1_data/metro/metro_nodes.txt 1_data/metro/metro_edges.txt 1_data/taz/taz_nodes.txt
	python make_multiplex.py

2_multiplex/multiplex_nodes.txt: 2_multiplex/multiplex_no_traffic_nodes.txt 2_multiplex/multiplex_no_traffic_edges.txt 1_data/taz_od/0_1.txt
	python assign_traffic.py igraph 
	rm 2_multiplex/multiplex_unscaled_nodes.txt 2_multiplex/multiplex_unscaled_edges.txt 2_multiplex/multiplex_no_traffic_nodes.txt 2_multiplex/multiplex_no_traffic_edges.txt

2_multiplex/multiplex_no_traffic_nodes.txt: 2_multiplex/multiplex_unscaled_nodes.txt 2_multiplex/multiplex_unscaled_edges.txt 
	python scale_edge_weights.py 1.51
	
