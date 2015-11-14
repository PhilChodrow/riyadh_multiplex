import networkx as nx
import matplotlib.pyplot as plt
from numpy import std
from sys import stdout
import multiplex

################################################################################
################################## Constants ###################################
################################################################################

roadOffset = (-0.0009, -0.0025) 
walkSpeed = 5./60.                #The average walking time in km/min (corresponds to 5 km/h)
drivingScale = 1.73               #The scaling factor for driving times on roads (relative to the speed limit)
metroWait = 0.                    #The average waiting time (min) at metro stops


################################################################################
######################## Creates the transport network #########################
################################################################################

#Imports the metro stations (nodes) and lines (edges)
def readMetro(filePath):
    with open(filePath+"Metro Nodes.txt", 'r') as f:
        rows = f.read().splitlines()
        size = len(rows)
        stations = [0]*(size)
        for i in range(0,size):
            row = rows[i].split()
            stations[i] = [ row[0].strip(), float(row[1]), float(row[2]) ]
    with open(filePath+"Metro Edges.txt", 'r') as f:
        rows = f.read().splitlines()
        size = len(rows)
        lines = [0]*(size)
        for i in range(0,size):
            row = rows[i].split()
            lines[i] = [ row[0].strip(), row[1].strip() , float(row[2].strip()), float(row[3].strip()) ]
    return stations, lines

#Imports the list of parking areas near specific metro station
def readParking(filePath):
    with open(filePath + "Parking.txt", 'r') as f:
        rows = f.read().splitlines()
        size = len(rows)
        parking = [0]*size
        for i in range(0,size):
            row = rows[i].split()
            parking[i] = [ row[0].strip(), row[1].strip() ]
        return parking

#Imports the intersections (nodes) and roads (edges)
def readRoad(filePath):
    with open(filePath+"Road Nodes.txt", 'r') as f:
        rows = f.read().splitlines()
        size = len(rows)
        inter = [0]*(size-1)
        for i in range(1,size):
            row = rows[i].split()
            inter[i-1] = [int(row[0]), float(row[1])+roadOffset[0], float(row[2])+roadOffset[1] ]
    with open(filePath+"Road Edges 2.txt", 'r') as f:
        rows = f.read().splitlines()
        size = len(rows)
        roads = [0]*(size-1)
        for i in range(1,size):
            row = rows[i].split()
            roads [i-1] = [ row[0], int(row[1]), int(row[2]), float(row[3]), float(row[4]), float(row[5]) ]
    return inter, roads

#Imports the data for TAZ connectors
def readCon(filePath):
    with open(filePath, 'r') as f:
        data = f.read().splitlines()
        connectors = [0]*len(data)
        for i in range(len(data)):
            d = data[i].split()
            connectors[i] = [ int(d[0]), ( float(d[2]), float(d[3]) ), int(d[4]), int(d[5]) ]
    return connectors


#Reads the tract OD data and returns a dictionary of all OD flows
def readOD(directory, purpose, time):
    if purpose < 0 or purpose > 3 or time < 0 or time > 3:
        print 'No file exists: choose parameters between 0 and 3'
        return 
    ODpath = directory + str(purpose) + '_' + str(time) + '.txt'
    with open(ODpath, 'r') as f:
        rows = f.read().splitlines()
        size = len(rows)-1
        ODdata = {}
        for i in range(size):
            row = rows[i+1].split()
            r =  [ int(row[0]), int(row[1]), float(row[2])]
            if r[0] in ODdata: ODdata[r[0]][r[1]] = r[2]
            else: ODdata[r[0]] = { r[1]: r[2] } 
    return ODdata

def connectorGraph(filePath):
    graph = nx.DiGraph()
    con = readCon(filePath)
    connectors = []
    intersections = {}
    for c in con:
        connectors.append(c[0]) 
        graph.add_node( c[0], pos = c[1], inter = "D_"+str(c[2]), taz = c[3], ntype = 'C', layer = "C")
        intersections["C_"+str(c[0])] = "D_"+str(c[2])
    return graph, connectors, intersections

def connectorOD(multi, filePath, con):
    OD  = readOD(filePath+"OD/", 0,1)
    con2 = [ [ 0, [] ] for i in range(1493) ]
    for c in multi.G.nodes():
        if multi.G.node[c]['layer'] == "C":
            taz = multi.G.node[c]['taz']
            con2[taz][0] += 1
            con2[taz][1].append(c)
    allCon = [ ]
    for c in con2: allCon += c[1]
    OD2 = { o: { d: 0. for d in allCon } for o in allCon}
    for origin in OD:
        for dest in OD[origin]:
            if con2[origin][0]*con2[dest][0] > 0: 
                flow = OD[origin][dest]/(con2[origin][0]*con2[dest][0])
                for i in con2[origin][1]:
                    for j in con2[dest][1]: 
                        OD2[i][j] += flow
    OD3 = {}
    for origin in OD2:
        destinations = {}
        for destination in OD2[origin]:
            if OD2[origin][destination] > 0: destinations[destination] = OD2[origin][destination]
        if len(destinations) > 0: OD3[origin] = destinations
    return OD3

    
# Sets up the multiplex road and metro network with different layers indicated by node types (ntypes) and edge types (etypes)
#     Roads are designated by the ntype "I" (intersections) and etype "s" (streets)
#     The metro is designated by the ntype "S" (stations) and etype "r" (rail)
#     TAZ connectors are indicated by the ntype 'C'
def createNetwork(filePath):
    nodes, edges = readRoad(filePath)
    stations, lines = readMetro(filePath)
    D, M = nx.DiGraph(), nx.DiGraph()
    C, con, conJoin = connectorGraph(filePath + "TAZ Connectors.txt")
    for n in nodes: D.add_node(n[0], pos = (n[1], n[2]), ntype = 'I')
    for e in edges: D.add_edge(e[1], e[2], gid = e[0], dist_km = e[3], cost_time_m = drivingScale*e[4], free_flow_time = drivingScale*e[4], capacity = e[5])
    for s in stations: M.add_node(s[0], pos = (s[1], s[2]), ntype = 'S')
    for r in lines: M.add_edges_from( [ (r[0], r[1]), (r[1], r[0]) ], dist_km = r[2]/1000., cost_time_m = r[3]/60.)
    removeZeros(D)
    multi = multiplex.multiplex()
    multi.add_layers2( { 'C': C, 'D':D, 'M':M } )
    multi.spatial_join2( 'M', 'D', 0., 0. )
    multi.manual_join( 'C', 'D', conJoin, 0., 0. )
    return multi, con

# Sets up the road network
def createDrivingNetwork(filePath):
    nodes, edges = readRoad(filePath)
    D = nx.DiGraph()
    C, con, conJoin = connectorGraph(filePath + "TAZ Connectors.txt")
    for n in nodes: D.add_node(n[0], pos = (n[1], n[2]), ntype = 'I', layer = "D")
    for e in edges: D.add_edge(e[1], e[2], gid = e[0], dist_km = e[3], cost_time_m = drivingScale*e[4], free_flow_time = drivingScale*e[4], capacity = e[5])
    removeZeros(D)
    multi = multiplex.multiplex()
    multi.add_layers2( { 'C': C, 'D':D } )
    multi.manual_join( 'C', 'D', conJoin, 0., 0. )
    return multi, con

#Removes any edges with zero capacity from the initial road network
def removeZeros(graph):
    for s, t in graph.edges():
        if graph.edge[s][t]['capacity'] == 0:
            graph.remove_edge(s,t)


def writeVolume(filePath, volume, multi, delim = "\t"):
    with open(filePath, "w+") as f:
        f.write( "Source" + delim + "Target" + delim + "Volume" + delim + "cost_time_m" )
        for e1, e2 in volume:
            f.write( str(e1) + delim + str(e2) + delim + str(volume(e1,e2)) + delim + multi.G.edge[e1][e2]['cost_time_m'] )

def readGoogle(filePath):
    with open(filePath, 'r') as f:
        data = f.read().splitlines()
        routes = {}
        unfounds = []
        for d in data[1:]: 
            line = d.split('\t')
            if line[3] != '-1':
                origin = "D_"+line[1]
                destination = "D_"+line[2]
                try: lineData = { "Distance": int(line[3]), "Free Flow": int(line[4]), "Expected": int(line[5]), "Additional Estimate": int(line[6]), "Best Case": int(line[7]), "Worst Case": int(line[8]) }
                except: 
                    unfounds.append( int(line[0]) )
                    continue
                if origin in routes: routes[origin][destination] = lineData
                else: routes[origin] = { destination: lineData }
    return routes

def google_comparison(pathLengths, googleResults):
    results, google = [ [] for i in range(len(pathLengths)) ], [ [], [], [] ]
    for origin in googleResults:
        for destination in googleResults[origin]:
            google[0].append(googleResults[origin][destination]['Free Flow']/60.)
            for i in range(len(pathLengths)): results[i].append(pathLengths[i][origin][destination])
            google[1].append(googleResults[origin][destination]['Expected']/60.)
    ratios = [ results[0][i]/google[0][i] for i in range(len(google[0])) ]
    print "Free flow travel time ratio (results/google) \n\tAverage:", sum(ratios)/len(ratios), "(SD: " + str(std(ratios)) + ")"
    ratios = [ results[-1][i]/google[1][i] for i in range(len(google[1])) ]
    print "Congested travel time ratio (results/google) \n\tAverage:", sum(ratios)/len(ratios), "(SD: " + str(std(ratios)) + ")"
    return results, google

def google_comparison_plots(results, google):
    print "Plotting a comparison of free flow travel times with Google Maps estimates"
    plt.figure(figsize=(10.,10.))
    plt.scatter(results[0], google[0], s = 1, alpha = 0.025)
    plt.plot( [0,100], [0,100] )
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.title("Travel Times for Morning Flows (Free Flow) ")
    plt.xlabel("Multiplex Class")
    plt.ylabel("Google Maps")
    plt.show()
    print "Plotting a comparison of congested travel times with Google Maps estimates"
    plt.figure(figsize=(10.,10.))
    plt.scatter(results[1], google[1], s = 1, alpha = 0.025)
    plt.plot( [0,100], [0,100] )
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.title("Travel Times for Morning Flows (Congestion) ")
    plt.xlabel("Multiplex Class")
    plt.ylabel("Google Maps")
    plt.show()
    avg1, avg2 = sum(results[1])/len(results[1]), sum(google[1])/len(google[1])
    print "Plotting a comparison of congested travel times (rescaled by " + str(avg2/avg1)[:5] + ") with Google Maps estimates"
    plt.figure(figsize=(10.,10.))
    results2 = [ (avg2/avg1)*m for m in results[1] ]
    plt.scatter(results2, google[1], s = 1, alpha = 0.025)
    plt.plot( [0,100], [0,100] )
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.title("Travel Times for Morning Flows (Rescaled Congestion) ")
    plt.xlabel("Multiplex Class")
    plt.ylabel("Google Maps")
    plt.show()

def multiplex_comparison(pathLengths, pathLengths2):
    withMetro, noMetro = [], []
    for origin in pathLengths[-1]:
        for destination in pathLengths[-1][origin]:
            withMetro.append( pathLengths[-1][origin][destination] )
            noMetro.append( pathLengths2[-1][origin][destination] )
    plt.figure(figsize=(10.,10.))
    plt.scatter(noMetro, withMetro, s = 1, alpha = 0.025)
    plt.plot( [0.,max(noMetro)], [0.,max(noMetro)] )
    plt.xlim(0,120)
    plt.ylim(0,120)
    plt.title("Change in Congested Travel Times")
    plt.xlabel("Travel Time (No Metro)")
    plt.ylabel("Travel Time (With Metro)")
    plt.show()


def plot_volume(multi, volume):
    pos = {}
    for n in multi.G.node: pos[n] = multi.G.node[n]['pos']
    for e1,e2 in volume:
        if multi.G.edge[e1][e2]['layer'] == 'D':
            color = 'red' if volume[(e1,e2)] > 2500 else 'orange' if volume[(e1,e2)] > 1750 else 'yellow' if volume[(e1,e2)] > 1000 else 'lightgreen' if volume[(e1,e2)] > 500 else 'green'
            nx.draw_networkx_edges(multi.G, pos, edgelist = [(e1,e2)], edge_color = color, arrows = False)
    for e1, e2 in multi.G.edges():
        if multi.G.edge[e1][e2]['layer'] == 'M':
            nx.draw_networkx_edges(multi.G, pos, edgelist = [(e1,e2)], edge_color = 'blue', arrows = False, width = 3, alpha = 0.15) 
    plt.xlim(46.45, 46.95)
    plt.ylim(24.54, 24.85)
    plt.show()


def plot_voc(multi, volume):
    pos = {}
    for n in multi.G.node: pos[n] = multi.G.node[n]['pos']
    for e1,e2 in volume:
        if multi.G.edge[e1][e2]['layer'] == 'D':
            color = 'red' if volume[(e1,e2)]/multi.G.edge[e1][e2]['capacity'] > 1. else 'orange' if volume[(e1,e2)]/multi.G.edge[e1][e2]['capacity'] > 0.75 else 'yellow' if volume[(e1,e2)]/multi.G.edge[e1][e2]['capacity'] > 0.5 else 'lightgreen' if volume[(e1,e2)]/multi.G.edge[e1][e2]['capacity'] > 0.25 else 'green'
            nx.draw_networkx_edges(multi.G, pos, edgelist = [(e1,e2)], edge_color = color, arrows = False)
    for e1, e2 in multi.G.edges():
        if multi.G.edge[e1][e2]['layer'] == 'M':
            nx.draw_networkx_edges(multi.G, pos, edgelist = [(e1,e2)], edge_color = 'blue', arrows = False, width = 3, alpha = 0.15) 
    plt.xlim(46.45, 46.95)
    plt.ylim(24.54, 24.85)
    plt.show()


directory = "/Users/Zeyad/Desktop/Public Transportation Network/Network Data/"
googleDirectory = '/Users/Zeyad/Downloads/Routes 2.txt'

multi, con = createNetwork(directory)
driving, con2 = createDrivingNetwork(directory)
conOD = connectorOD(multi, directory, con)
    
routes = readGoogle(googleDirectory)
googleOD = { origin: { destination: True for destination in routes[origin] } for origin in routes } 
volume, pathLengths = multi.geo_betweenness_ITA(0.25, conOD, googleOD)
volume2, pathLengths2 = driving.geo_betweenness_ITA(0.25, conOD, googleOD)

#Use the following functions to validate the results against Google Maps estimates
"""
results, google = google_comparison(pathLengths2, routes)
comparison_plots(results, google)
"""

#Use the following functions to plot the edge volume and volume/capacity for the two networks and compare their travel times under congestion
"""
plot_volume(driving, volume2)
plot_voc(driving, volume2)
plot_volume(multi, volume)
plot_voc(driving, volume)
multiplex_comparison(pathLengths, pathLengths2)
"""