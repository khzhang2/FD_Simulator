import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox


############################################################
# load data
# get HK island road network
G_osm = ox.graph_from_bbox(
    22.296438, 22.272757, 114.188941, 114.124862,
    network_type='drive',
    retain_all=False,
    truncate_by_edge=False,
    simplify=True,
)

nodes = pd.DataFrame([], columns=['NodeName', 'x', 'y'])
nodes['NodeName'] = list(G_osm.nodes)
nodes['x'] = [G_osm.nodes[i]['x'] for i in G_osm.nodes]
nodes['x'] = (nodes['x'] - nodes['x'].min()) * 111  # from longitude to km
nodes['y'] = [G_osm.nodes[i]['y'] for i in G_osm.nodes]
nodes['y'] = (nodes['y'] - nodes['y'].min()) * 111  # from latitude to km

edges_lst = list(G_osm.edges.data())
l = len(edges_lst)

edges = pd.DataFrame([], columns=['EdgeName', 'from', 'to', 'distance'])
edges['EdgeName'] = range(l)
edges['from'] = [edges_lst[i][0] for i in range(l)]
edges['to'] = [edges_lst[i][1] for i in range(l)]
edges['distance'] = [edges_lst[i][2]['length']/1000 for i in range(l)]

G = nx.Graph()
G.add_nodes_from(nodes['NodeName'])
G.add_weighted_edges_from(edges.iloc[:, 1:].to_numpy())

############################################################

## Difine rider
# some useful utility functions
def get_edge(G, from_node, to_node):
    edge = edges.loc[(edges['from']==from_node)&(edges['to']==to_node)]
    return edge['EdgeName'].iloc[0]

def norm_vec(a):
    return (np.array(a) / np.linalg.norm(np.array(a))).flatten()

def get_node(ID):
    node_index = nodes.loc[nodes['NodeName']==ID].index.values[0]
    return node_index

def get_node_xy(ID):
    node_position = nodes.loc[nodes['NodeName']==ID, ['x', 'y']].to_numpy().flatten()
    return node_position
    
def get_adj_node_position(position):
    # return the position of the closest adjacent node
    adj_node_index = (((nodes.iloc[:, 1:] - position )**2).sum(axis=1)).idxmin()
    adj_node = nodes.loc[adj_node_index, 'NodeName']
    adj_node_position = nodes.loc[nodes['NodeName']==adj_node, ['x', 'y']].to_numpy().flatten()
    return adj_node, adj_node_position

def get_closest_node(from_position, target_nodes):
    # by distance
    distance = 1e10
    closest_node = None
    for i in target_nodes:
        this_distance = np.linalg.norm(from_position - get_node_xy(i))
        if this_distance < distance:
            distance = this_distance
            closest_node = i
    return closest_node

def get_closest_node_dijkstra(from_node, target_nodes):
    # by distance traveled, dijkstra distance
    # from_node: ID or position
    if type(from_node)==int:
        None
    elif type(from_node)==np.ndarray:
        current_node = int(
            nodes.loc[(nodes['x']==from_node[0])&(nodes['y']==from_node[1]), 'NodeName'].values
        )
        from_node = current_node

    distance = 1e10
    closest_node = None
    for i in target_nodes:
        if nx.dijkstra_path_length(G, from_node, i) < distance:
            distance = nx.dijkstra_path_length(G, from_node, i)
            closest_node = i
    return closest_node

class rider:
    def __init__(self, config, dec_var, merchant_node_set, rd_st=np.random.RandomState(42)):
        # config is a dictionary
        self.ID = config['ID']                                  # scalar
        self.position = config['initial_position']              # 2-D array
        self.maxspeed = config['maxspeed']                      # scalar
        self.stop_time = 0                                      # scalar, float
        self.state = 'idle'                                     # string 'idle' or 'working' or 'stop'
        self.speed = self.maxspeed / 2
        self.if_matched = False
        self.customer_nodes = []
        self.newly_finished_destination = None
        self.merchant_node = None
        self.path = None
        self.destination = None
        self.total_time = 0
        self.total_time_rec = []
        self.if_matchable = False
        self.merchant_node_set = merchant_node_set
        self.rd_st = rd_st
        
        adj_node, adj_node_position = get_adj_node_position(self.position)
        self.position = adj_node_position
        self.closest_merchant_node = get_closest_node(self.position, self.merchant_node_set)
        if np.linalg.norm(adj_node_position - self.position) == 0:
            self.direction = norm_vec(self.rd_st.rand(2))
        else:
            self.direction = norm_vec(adj_node_position - self.position)
        
        self.next_node = adj_node
        self.nextnext_node = self.rd_st.choice([i for i in G.neighbors(self.next_node)])
        
        self.dec_var = dec_var  # decision variables

    def update_customer_order(self):
        new_customer_nodes = []
        customer_2_merchant_distance_set = []
        for customer in self.customer_nodes:
            customer_2_merchant_distance_set.append(np.linalg.norm(get_node_xy(customer) - get_node_xy(self.merchant_node)))
        
        d_min = min(customer_2_merchant_distance_set)
        d_min_ind = customer_2_merchant_distance_set.index(d_min)

        # time complexity O(n^2)
        for i in range(len(self.customer_nodes)):
            for customer in self.customer_nodes:
                if np.linalg.norm(get_node_xy(self.merchant_node) - get_node_xy(customer)) > d_min:
                    d_min = np.linalg.norm(get_node_xy(self.merchant_node) - get_node_xy(customer))
                    new_customer_nodes.append(customer)
        new_customer_nodes.append(self.customer_nodes[d_min_ind])
        self.customer_nodes = new_customer_nodes

    def update_customer_nodes(self):
        self.update_customer_order()
        # rider merchant_node has been updated in function "move_rider"
        self.destination = self.merchant_node
        adj_node, adj_node_position = get_adj_node_position(self.position)
        if np.linalg.norm(adj_node_position - self.position) == 0:
            self.direction = norm_vec(self.rd_st.rand(2))
        else:
            self.direction = norm_vec(adj_node_position - self.position)
        
        self.next_node = adj_node
        
        # first go to the closest node, then follow the path
        self.path = nx.dijkstra_path(
            G, self.next_node, self.destination
        )
        # next node is the last node, then the next next node is random
        self.nextnext_node = self.rd_st.choice([i for i in G.neighbors(self.next_node)]) if self.next_node==self.path[-1] else self.path[1]

        # when matched, set it to be unmatchable
        self.if_matchable = False
    
    def check_distance_2_closest_merchant(self):
        self.closest_merchant_node = get_closest_node(self.position, self.merchant_node_set)
        
        closest_merchant_position = get_node_xy(self.closest_merchant_node)
        if np.linalg.norm(self.position - closest_merchant_position) <= self.dec_var['r']:
            self.if_matchable = True
        else:
            self.if_matchable = False

    def move(self, t_resolution, dec_var):
        self.newly_finished_destination = None
        self.dec_var = dec_var
        # move one step foward
        if self.state == 'idle':
            travel_distance_mag = self.speed * self.rd_st.rand() * t_resolution
            if self.if_matched:
                self.state = 'working'
                self.speed = self.maxspeed
                # customer_nodes has been updated in function "move_rider"
                self.update_customer_nodes()
                
        elif self.state == 'working':
            travel_distance_mag = self.speed * t_resolution
            self.total_time += t_resolution
        elif self.state == 'stop':
            self.total_time += t_resolution
            self.stop(t_resolution)
            return
        
        next_node_position = get_node_xy(self.next_node)
        distance_to_next_node = np.linalg.norm(self.position - next_node_position)
        
        if travel_distance_mag < distance_to_next_node:
            self.position = self.position + travel_distance_mag * self.direction
        elif travel_distance_mag >= distance_to_next_node:
            # travel distance greater than the distance to the next node
            self.position = next_node_position
            
            if self.state=='working' and self.next_node == self.destination:  # this is for working riders
                # arrive the destination
                # give up the abundant distance, and stop
                self.stop(t_resolution)
                return
                
            exceed_distance = travel_distance_mag - distance_to_next_node
            
            nextnext_node_position = get_node_xy(self.nextnext_node)
            self.direction = norm_vec(nextnext_node_position - next_node_position)
            
            self.position = self.position + exceed_distance * self.direction
            self.next_node = self.nextnext_node
            
            if self.state=='idle':
                self.nextnext_node = self.rd_st.choice([i for i in G.neighbors(self.next_node)])
            else:
                if self.next_node == self.destination:
                    # next node is the last node, then the next next node is random
                    self.nextnext_node = self.rd_st.choice([i for i in G.neighbors(self.next_node)])
                else:
                    self.nextnext_node = self.path[self.path.index(self.next_node) + 1]
        self.check_distance_2_closest_merchant()

    def stop(self, t_resolution):
        # when pickup, delivered, or else
        self.state = 'stop'
        if self.stop_time > 0.1:
            # restart
            self.state = 'working'
            self.stop_time = 0
            self.newly_finished_destination = self.destination  # consider the current destination as finished
            self.update_next_desination(t_resolution)
        else:
            self.stop_time = self.stop_time + t_resolution
    
    def complete(self):
        # complete all orders in current bundle
        print('rider %i completed! time:%.2f'%(self.ID, self.total_time))
        self.next_node = self.rd_st.choice([i for i in G.neighbors(self.next_node)])
        next_node_position = get_node_xy(self.next_node)
        self.direction = norm_vec(next_node_position - self.position)
        self.nextnext_node = self.rd_st.choice([i for i in G.neighbors(self.next_node)])
        
        self.state = 'idle'
        self.speed = self.maxspeed / 2
        self.if_matched = False
        self.customer_nodes = []
        self.path = None
        self.destination = None
        self.merchant_node = None
        self.total_time_rec.append(self.total_time)
        self.total_time = 0
        return
    
    def update_next_desination(self, t_resolution):
        ##############################################################
        # To use shortest customer, uncomment the following line
        ##############################################################
        # update the shortest node as the next destination
        # next_destination = get_closest_node_dijkstra(self.position, self.customer_nodes)

        next_destination = self.customer_nodes[0] if len(self.customer_nodes) > 0 else None
        if next_destination==None:
            # complete
            self.complete()
            return
        next_dest_i = self.customer_nodes.index(next_destination)
        # udpate customer_nodes
        new_customer_nodes = []
        new_customer_nodes.extend(self.customer_nodes[:next_dest_i])
        new_customer_nodes.extend(self.customer_nodes[next_dest_i+1:])
        self.customer_nodes = new_customer_nodes


        self.destination = next_destination
        self.path = nx.dijkstra_path(G, self.next_node, self.destination)  # next_node is current location
        if len(self.path)==1:
            self.stop(t_resolution)
            return
        self.next_node = self.path[1]
        # if only 2 nodes, then nextnext node is random
        self.nextnext_node = self.path[2] if len(self.path)>2 else self.rd_st.choice([i for i in G.neighbors(self.next_node)])  
        next_node_position = get_node_xy(self.next_node)
        self.direction = norm_vec(next_node_position - self.position)

def move_rider(rider_set, t_resolution, matched_rider_IDs, matched_batches, matched_merchants, dec_var):
    for i in range(len(rider_set)):
        rider_i = rider_set[i]
        if rider_i.ID in matched_rider_IDs:
            # if this rider is mathced, update if_matched
            rider_i.if_matched = True
            # find corresponding index of rider in matched_rider_IDs
            batch_index = matched_rider_IDs.index(rider_i.ID)
            
            # get the corresponding matched batch and update to customer_nodes
            rider_i.customer_nodes = list(matched_batches[batch_index, :])
            # get the corresponding matched merchant and update to merchant_node
            rider_i.merchant_node = matched_merchants[batch_index]
            # move rider with updated customer_nodes
            rider_i.move(t_resolution, dec_var)
        else:
            rider_i.move(t_resolution, dec_var)
    
    return rider_set