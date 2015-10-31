
all: 2.\ multiplex/multiplex_nodes.txt  4.\ figs/betweenness/dist_km/beta1.png methods/methods.pdf

clean: 
	rm -rf 4.\ figs/betweenness
	rm 2.\ multiplex/multiplex_edges.txt 2.\ multiplex/multiplex_nodes.txt
	rm \methods/methods.pdf 
 	
2.\ multiplex/multiplex_nodes.txt: 1.\ data/street/street_nodes.txt 1.\ data/street/street_edges.txt 1.\ data/metro/metro_nodes.txt 1.\ data/metro/metro_edges.txt
	python make_multiplex.py

methods/methods.pdf: methods/methods.tex
	cd methods; ls; pdflatex methods.tex; rm methods.aux methods.log; cd ..

4.\ figs/betweenness/dist_km/beta1.png: 2.\ multiplex/multiplex_nodes.txt 2.\ multiplex/multiplex_edges.txt
	python betweenness_dist.py

