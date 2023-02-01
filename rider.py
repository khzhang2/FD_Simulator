import numpy as np
import networkx as nx
import pandas as pd


nodes = pd.read_csv('./data/node.csv')
edges = pd.read_csv('./data/edge.csv')

edges['distance'] = ''
for i in edges.index:
    from_node = nodes.loc[nodes['NodeName']==edges.loc[i, 'from']]
    to_node = nodes.loc[nodes['NodeName']==edges.loc[i, 'to']]
    edges.loc[i, 'distance'] = np.sqrt(
        (from_node['x'].values[0] - to_node['x'].values[0])**2 + (from_node['y'].values[0] - to_node['y'].values[0])**2
    )


G = nx.Graph()
G.add_nodes_from(nodes['NodeName'])
G.add_weighted_edges_from(edges.iloc[:, 1:].to_numpy())



