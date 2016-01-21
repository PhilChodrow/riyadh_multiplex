# Multiplex Metro

## Purpose
This repo contains code for the Riyadh Multiplex Metro project's analytical pipeline. Data prep and analysis are written in discrete Python scripts, which are then structured into a pipeline via GNU Make. The terminal command `make all` in this directory will execute the analysis to the full extent of its latest implementation. The analysis currently includes the following scripts and source files: 

## Data
The `data` directory includes the following data sets

1. `street`, comprising `street_nodes.txt` and `street_edges.txt` 
2. `metro`, comprising `metro_nodes.txt` and `metro_edges.txt` 
3. `taz`, comprising `taz_connectors.txt`
4. `taz_od`, comprising 8 different files of origin-destination flow data. The one currently used in the pipeline is `0_1.txt`.
5. `google_times.txt`, a data frame of google travel time estimates used for validation.

## Modules

The following modules are all contained in the package `metro`. To import one of the modules, use syntax as in the following: 
```
    from metro import multiplex as mx
```

1. `multiplex.py` : a Python class definition that implements a relatively thin wrapper around the networkx.DiGraph class for handling multilayer networks.
2. `utility.py` : a collection of functions for interacting with multiplex objects, including modifying their attributes and extracting information for further analysis. 
3. `assignment.py` : a collection of functions for running assignment algorithms on a multiplex object. Includes two primary ITA implementations: a homebrew implementation by Zeyad and an igraph implementation by Phil. These implementions have identical results. 
4. `analysis.py` : a collection of functions for analytical computations involving multiplex objects.
5. `viz.py` : a collection of functions for visualizations of multiplex objects. 
6. `ita.py` : a collection of functions for performing ITA-like calculations, including shortest paths, with a multiplex object. 

## Scripts

1. `make_multiplex.py` : a Python script for constructing a `multiplex` object using data in the `1_data` directory. This script saves the resulting multiplex as nodes and edges in the `2_multiplex` directory. 
2. `scale_edge_weights.py` : a Python script for scaling the edge weights of the multiplex by a fixed constaint. Typically applied to travel time weights like `uniform_time_m`, `free_flow_time_m`, and `congested_time_m`. Default used in the makefile is 1.51. 
3. `assign_flows.py` : A Python script that performs repeated ITA for varying levels of metro speed. 
4. `weighted_shortest_paths.py`: A Python script for computing weighted shortest paths on the single layer network under different edge weights. Primarily for a single plot; Phil would like to fold this into another script. 

## Other
1. `makefile` : a makefile automating the data preparation and analysis pipeline
