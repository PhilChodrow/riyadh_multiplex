# Multiplex Metro

## Purpose
This repo contains code for the Riyadh Multiplex Metro project's analytical pipeline. Data prep and analysis are written in discrete Python scripts, which are then structured into a pipeline via GNU Make. The terminal command `make all` in this directory will execute the analysis to the full extent of its latest implementation. The analysis currently includes the following scripts and source files: 

## Scripts
1. `methods/methods.tex`: A .tex document outlining the analytical approach of the project. 
2. `utility.py`: A Python module containing a variety of useful helper functions used in other scripts. 
3. `multiplex.py`: A Python class definition that implements a relatively thin wrapper around the networkx.DiGraph class for handling multilayer networks. 
4. `make_multiplex.py`: a Python script for constructing a `multiplex` objecting using data in the `1. data` directory. The script saves the nodes and edges of the resulting object in the `2. multiplex` directory.
5. `betweenness.py`: a Python script that reads in a multiplex object from the `2. multiplex` directory and prints out a series of betweenness plots weighted by the user-specified edge measure. Comparable to Figure 6 in Strano et al.'s [recent paper](http://arxiv.org/abs/1508.07265)
7. `intermodality_analysis.py`: a Python script that reads in a multiplex object from the `2. multiplex` directory and creates a plot of intermodality -- the ratio of shortest paths that pass through the metro -- as a function of beta. 
8. `intermodality_local.py`: a Python script that reads in a multiplex object from the `2. multiplex` directory and creates a plot of the spatial distribution of intermodality for a fixed value of beta. 

## Data
The `data` directory includes the following data sets

1. `street`, comprising `street_nodes.txt` and `street_edges.txt`. 
2. `metro`, comprising `metro_nodes.txt` and `metro_edges.txt`. 
3. `taz`, comprising `taz_connectors.txt`.  

## Todo
1. Update README.md! 
2. Incorporate travel-time validation into pipeline
3. 



