import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import shapely


############################################################
# load data
# get Singapore island road network
# large network, aspect ratio: (north - south) / (east - west) * 1.84
north, south, east, west = 1.42000830738447, 1.2452363695255813, 103.99630998097328, 103.67516457816659
aspect_ratio = 1.84
# G_sgp = ox.graph_from_bbox(
#     north, south, east, west,
#     network_type='drive',
#     retain_all=False,
#     truncate_by_edge=False,
#     simplify=True,
# )
# ox.save_graphml(G_sgp, filepath="./data/G_sgp.graphml")
G_sgp = ox.load_graphml("./data/G_sgp.graphml")
# simple network, aspect ratio: (north - south) / (east - west) * 1.71
# north, south, east, west = 1.2980476913475176, 1.2700868360667301, 103.84660194295621, 103.79894232251958
# aspect_ratio = 1.71
# G_sgp = ox.graph_from_bbox(
#     north, south, east, west,
#     network_type='drive',
#     retain_all=False,
#     truncate_by_edge=False,
#     simplify=True,
# )

# coordination system
nodes = pd.DataFrame([], columns=['NodeName', 'x', 'y'])
nodes['NodeName'] = list(G_sgp.nodes)
nodes['x'] = [G_sgp.nodes[i]['x'] for i in G_sgp.nodes]  # longitude
nodes['y'] = [G_sgp.nodes[i]['y'] for i in G_sgp.nodes]  # latitute
while True:
    c = 0
    for i in nodes.index:
        if len([i for i in G_sgp.neighbors(nodes.loc[i, 'NodeName'])]) == 0:
            G_sgp.remove_node(nodes.loc[i, 'NodeName'])
            nodes = nodes.drop(i)
            c += 1
    if c == 0:
        break

pos = dict(zip(nodes['NodeName'].to_numpy(), nodes[['x', 'y']].to_numpy()))

edges_lst = list(G_sgp.edges.data())
for i in range(len(edges_lst)):
    # fill maxspeed for which does not have
    if 'geometry' not in list(edges_lst[i][2].keys()):
        node_1_coor = nodes.loc[nodes['NodeName'] == edges_lst[i][0], ['x', 'y']].to_numpy().flatten().tolist()
        node_2_coor = nodes.loc[nodes['NodeName'] == edges_lst[i][1], ['x', 'y']].to_numpy().flatten().tolist()
        edges_lst[i][2]['geometry'] = shapely.geometry.LineString([node_1_coor, node_2_coor])
    else:
        None
    # fill maxspeed for which does not have
    if 'maxspeed' not in list(edges_lst[i][2].keys()):
        edges_lst[i][2]['maxspeed'] = int(edges_lst[i - 1][2]['maxspeed'])
    else:
        if isinstance(edges_lst[i][2]['maxspeed'], str):
            edges_lst[i][2]['maxspeed'] = int(edges_lst[i][2]['maxspeed'])
        elif isinstance(edges_lst[i][2]['maxspeed'], list):
            edges_lst[i][2]['maxspeed'] = max(map(int, edges_lst[i][2]['maxspeed']))

edges = pd.DataFrame([], columns=['EdgeName', 'from', 'to', 'distance'])
edges['EdgeName'] = range(len(edges_lst))
edges['osmid'] = [str(edges_lst[i][2]['osmid']) for i in range(len(edges_lst))]
edges['from'] = [edges_lst[i][0] for i in range(len(edges_lst))]
edges['to'] = [edges_lst[i][1] for i in range(len(edges_lst))]
edges['distance'] = [edges_lst[i][2]['length']/1000 for i in range(len(edges_lst))]  # unit: km
edges['maxspeed'] = [edges_lst[i][2]['maxspeed'] for i in range(len(edges_lst))]  # unit: km/hr
edges['travel_time_minimum'] = edges['distance'] / edges['maxspeed']  # unit: hour

G_sgp.add_weighted_edges_from(edges.iloc[:, 1:4].to_numpy())

############################################################
# Define rider
############################################################


