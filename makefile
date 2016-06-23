# filesets 

_unscaled = 2_multiplex/multiplex_unscaled_nodes.txt 2_multiplex/multiplex_unscaled_edges.txt 

_raw = 1_data/street/street_nodes.txt 1_data/street/street_edges.txt 1_data/metro/metro_nodes.txt 1_data/metro/metro_edges.txt 1_data/taz/taz_nodes.txt

_mx = 2_multiplex/mx_nodes.txt 2_multiplex/mx_edges.txt

_mx_flows = 3_throughput/mx_flow_nodes.txt 3_throughput/mx_flow_edges.txt

_shortest = 3_throughput/shortest_uniform_time_m.csv 3_throughput/shortest_congested_time_m_100.csv 3_throughput/shortest_free_flow_time_m.csv 3_throughput/shortest_dist_km.csv

# high-level interface --------------------------------------------------------
all: $(_mx_flows) $(_shortest)

clean:
	@rm -f $(_mx)
	@rm -f $(_unscaled)
	@echo 'All clean!'

# CLI for intermediate data prep stages ---------------------------------------
mx: $(mx)
unscaled: $(_unscaled)
mx_flows: $(_mx_flows)
shortest: $(_shortest)

# dependency logic ------------------------------------------------------------

$(_unscaled): $(_raw) make_multiplex.py
	@echo constructing multiplex
	@python make_multiplex.py

$(_mx): $(_unscaled) scale_edge_weights.py
	@echo scaling multiplex edges 
	@python scale_edge_weights.py 1.51

$(_mx_flows): $(_mx) assign_flows.py
	@echo assigning flows -- this could take a while
	@python assign_flows.py

$(_shortest): $(_mx_flows) weighted_shortest_paths.py
	@echo computing weighted shortest weighted_shortest_paths
	@python weighted_shortest_paths.py