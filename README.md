# Multiplex Metro

## Purpose
This repo contains code for the Riyadh Multiplex Metro project's analytical pipeline. Data prep and analysis are written in discrete Python scripts, which are then structured into a pipeline via GNU Make. The terminal command `make all` in this directory will execute the analysis to the full extent of its latest implementation. The analysis currently includes the following scripts and source files: 

## Data
The `data` directory includes the following data sets

1. `street`, comprising `street_nodes.txt` and `street_edges.txt` 
2. `metro`, comprising `metro_nodes.txt` and `metro_edges.txt` 
3. `taz`, comprising `taz_connectors.txt`
4. `taz_od`, comprising 8 different files of origin-destination flow data. The one currently used in the pipeline is `0_1.txt`.
5. google_times.txt, a data frame of google travel time estimates used for validation.

## Usage
To run the full data-preparation pipeline in its most comprehensive state, run the following terminal commands:
```
    rm 2_multiplex/*                     # deletes existing multiplex data
    python make_multiplex.py             # assembles multiplex
    python scale_edge_weights.py 1.51    # or another constant
    python assign_traffic.py both        # alternatives: homebrew, igraph
```
This full process takes roughly one hour on PC's machine. As an alternative, running the terminal command `make all` is currently equivalent to the following sequence: 
```
    python make_multiplex.py             # assembles multiplex
    python scale_edge_weights.py 1.51    # or another constant
    python assign_traffic.py igraph      # alternatives: homebrew, both
```
This is currently the fastest way to run the full pipeline, and takes about 20 minutes on PC's machine. 
## Modules

The following modules are all contained in the package `metro`. To import one of the modules, use syntax as in the following: 
```
    from metro import multiplex 
```

1. `multiplex.py` : a Python class definition that implements a relatively thin wrapper around the networkx.DiGraph class for handling multilayer networks.
2. `io.py` : a collection of input-output functions for quickly reading and writing multiplex objects. 
3. `utility.py` : a collection of functions for interacting with multiplex objects, including modifying their attributes and extracting information for further analysis. 
4. `assignment.py` : a collection of functions for running assignment algorithms on a multiplex object. Includes two primary ITA implementations: a homebrew implementation by Zeyad and an igraph implementation by Phil. These implementions have identical results. 
5. `analysis.py` : a collection of functions for analytical computations involving multiplex objects.
6. `viz.py` : a collection of functions for visualizations of multiplex objects. 

## Scripts

1. `make_multiplex.py` : a Python script for constructing a `multiplex` object using data in the `1_data` directory. This script saves the resulting multiplex as nodes and edges in the `2_multiplex` directory. 
2. `assign_traffic.py` : a Python script for ITA. Requires an argument at the command line, one of `homebrew`, `igraph`, or `both`. If `both`, will run both homebrew and igraph implementations and compare results. These are identical to within .001%, i.e. roundoff error. 
3. `scale_edge_weights.py` : a Python script for scaling the edge weights of the multiplex by a fixed constaint. Typically applied to travel time weights like `uniform_time_m`, `free_flow_time_m`, and `congested_time_m`. Default used in the makefile is 1.51. 

## Notebooks
1. `scaling_coefficient.ipynb` : a jupyter notebook with exploratory analysis to determine the scaling coefficient used in the pipeline.
2. `scaling_coefficient.html` : an .html render of the above for easy sharing. 
3. `spatial_outreach.ipynb` : an analysis of spatial outreach in the multiplex.  
4. `shortest_paths.ipynb` : an analysis of the behavior of shortest paths in the multiplex under uniform and OD demand, including with metro scaling.  
5. `reducing_congestion.ipynb` : an analysis of the 'urban planner's dream' for reducing congestion by strategically routing ODs through public transportation. 
 
## Other
1. `makefile` : a makefile automating the data preparation and analysis pipeline
2. `methods/methods.tex` : A .tex document outlining the analytical approach of the project.  