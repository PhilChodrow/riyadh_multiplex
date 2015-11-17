all: 2.\ multiplex/multiplex_nodes.txt  4.\ figs/betweenness/dist_km/beta1.png 4.\ figs/betweenness/cost_time_m/beta1.png 
	# 4.\ figs/intermodality/global/intermodality_profile.png  4.\ figs/intermodality/global/intermodality_profile.png methods/methods.pdf

clean: 
	rm -rf 4.\ figs/betweenness
	rm 2.\ multiplex/multiplex_edges.txt 2.\ multiplex/multiplex_nodes.txt
	rm \methods/methods.pdf 
 	
2.\ multiplex/multiplex_no_traffic_nodes.txt: 1.\ data/street/street_nodes.txt 1.\ data/street/street_edges.txt 1.\ data/metro/metro_nodes.txt 1.\ data/metro/metro_edges.txt
	python make_multiplex.py

2.\ multiplex/multiplex_nodes.txt: 2.\ multiplex/multiplex_no_traffic_nodes.txt 2.\ multiplex/multiplex_no_traffic_edges.txt
	python compute_congestion.py

methods/methods.pdf: methods/methods.tex
	cd methods; ls; pdflatex methods.tex; rm methods.aux methods.log; cd ..

4.\ figs/betweenness/dist_km/beta1.png: 2.\ multiplex/multiplex_nodes.txt 2.\ multiplex/multiplex_edges.txt
	python betweenness.py dist_km 

4.\ figs/betweenness/cost_time_m/beta1.png: 2.\ multiplex/multiplex_nodes.txt 2.\ multiplex/multiplex_edges.txt
	python betweenness.py cost_time_m

4.\ figs/intermodality/global/intermodality_profile.png: 2.\ multiplex/multiplex_nodes.txt 2.\ multiplex/multiplex_edges.txt
	python intermodality_analysis.py

4.\ figs/shortest_paths/dist_km.png: 2.\ multiplex/multiplex_nodes.txt 2.\ multiplex/multiplex_edges.txt 
	python path_distributions.py dist_km cost_time_m


# 4.\ figs/intermodality/global/intermodality_profile.png: 2.\ multiplex/multiplex_nodes.txt 2.\ multiplex/multiplex_edges.txt
# 	python intermodality_local.py







