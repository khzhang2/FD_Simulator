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


class rider:
    def __init__(self, config):
        # config is a dictionary
        self.ID = config['ID']                                  # scalar
        self.position = config['initial_position']              # 2-D array
        self.maxspeed = config['maxspeed']                      # scalar
        self.stop_time = 0                                      # scalar, float
        self.state = 'idle'                                     # string 'idle' or 'working' or 'stop'
        self.speed = self.maxspeed / 10
        self.knowledge = None
        self.if_matched = False
        self.customer_nodes = []
        self.matched_orders = []
        self.merchant_node = None
        self.path = None
        self.destination = None
        self.total_time = 0
        self.total_time_rec = []
        
        adj_node, adj_node_position = get_adj_node_position(self.position)
        if np.linalg.norm(adj_node_position - self.position) == 0:
            self.direction = norm_vec(np.random.rand(2))
        else:
            self.direction = norm_vec(adj_node_position - self.position)
        
        self.next_node = adj_node
        self.nextnext_node = np.random.choice([i for i in G.neighbors(self.next_node)])

    def update_knowledge(self, knowledge):
        self.knowledge = knowledge
        self.merchant_node = self.knowledge['merchant_node']     # scalar
        self.customer_nodes = self.knowledge['customer_nodes']   # 1-D array
        self.destination = self.merchant_node
        
        adj_node, adj_node_position = get_adj_node_position(self.position)
        if np.linalg.norm(adj_node_position - self.position) == 0:
            self.direction = norm_vec(np.random.rand(2))
        else:
            self.direction = norm_vec(adj_node_position - self.position)
        
        self.next_node = adj_node
        # first go to the closest node, then follow the path
        self.path = nx.dijkstra_path(
            self.knowledge['map'], self.next_node, self.destination
        )                                                       # list
        # next node is the last node, then the next next node is random
        self.nextnext_node = np.random.choice([i for i in G.neighbors(self.next_node)]) if self.next_node==self.path[-1] else self.path[1]

    def move(self, t_resolution, knowledge=None):
        # move one step foward
        if self.state == 'idle':
            travel_distance_mag = self.speed * np.random.rand() * t_resolution
            if self.if_matched:
                self.state = 'working'
                self.speed = self.maxspeed
                self.update_knowledge(knowledge)
                
        elif self.state == 'working':
            travel_distance_mag = self.speed * t_resolution
            self.total_time += t_resolution
        elif self.state == 'stop':
            self.total_time += t_resolution
            self.stop(t_resolution)
            return
        
        next_node_position = nodes.loc[nodes['NodeName']==self.next_node, ['x', 'y']].to_numpy().flatten()
        distance_to_next_node = np.linalg.norm(self.position - next_node_position)
        
        if travel_distance_mag < distance_to_next_node:
            self.position = self.position + travel_distance_mag * self.direction
        elif travel_distance_mag >= distance_to_next_node:
            # travel distance greater than the distance to the next node
            self.position = next_node_position
            
            if self.knowledge!=None and self.next_node == self.destination:
                # give up the abundant distance, and stop
                self.stop(t_resolution)
                return
                
            exceed_distance = travel_distance_mag - distance_to_next_node
            
            nextnext_node_position = nodes.loc[nodes['NodeName']==self.nextnext_node, ['x', 'y']].to_numpy().flatten()
            self.direction = norm_vec(nextnext_node_position - next_node_position)
            
            self.position = self.position + exceed_distance * self.direction
            self.next_node = self.nextnext_node
            
            if self.knowledge==None:
                self.nextnext_node = np.random.choice([i for i in G.neighbors(self.next_node)])
            else:
                if self.next_node == self.destination:
                    # next node is the last node, then the next next node is random
                    self.nextnext_node = np.random.choice([i for i in G.neighbors(self.next_node)])
                else:
                    self.nextnext_node = self.path[self.path.index(self.next_node) + 1]

    def stop(self, t_resolution):
        # when pickup, delivered, or else
        self.state = 'stop'
        if self.stop_time > 0.1:
            # restart
            self.state = 'working'
            self.stop_time = 0
            # update
            next_destination = self.get_closest_node(self.position)
            if next_destination==None:
                # complete
                self.complete()
                return
            next_dest_i = self.customer_nodes.index(next_destination)
            new_customer_nodes = []
            new_customer_nodes.extend(self.customer_nodes[:next_dest_i])
            new_customer_nodes.extend(self.customer_nodes[next_dest_i+1:])
            self.customer_nodes = new_customer_nodes
            self.matched_orders.append(next_destination)
            self.update(next_destination, t_resolution)
        else:
            self.stop_time = self.stop_time + t_resolution
    
    def complete(self):
        # complete all orders in current bundle
        print('rider %i completed! time:%.2f'%(self.ID, self.total_time))
        self.next_node = np.random.choice([i for i in G.neighbors(self.next_node)])
        next_node_position = nodes.loc[nodes['NodeName']==self.next_node, ['x', 'y']].to_numpy().flatten()
        self.direction = norm_vec(next_node_position - self.position)
        self.nextnext_node = np.random.choice([i for i in G.neighbors(self.next_node)])
        
        self.state = 'idle'
        self.speed = self.maxspeed / 10
        self.knowledge = None
        self.if_matched = False
        self.customer_nodes = []
        self.matched_orders = []
        self.merchant_node = None
        self.path = None
        self.destination = None
        self.total_time_rec.append(self.total_time)
        self.total_time = 0
        return
    
    def update(self, destination, t_resolution):
        # after arriving one destination
        self.destination = destination
        self.path = nx.dijkstra_path(self.knowledge['map'], self.next_node, self.destination)  # next_node is current location
        if len(self.path)==1:
            self.stop(t_resolution)
            return
        self.next_node = self.path[1]
        # if only 2 nodes, then nextnext node is random
        self.nextnext_node = self.path[2] if len(self.path)>2 else np.random.choice([i for i in G.neighbors(self.next_node)])  
        next_node_position = nodes.loc[nodes['NodeName']==self.next_node, ['x', 'y']].to_numpy().flatten()
        self.direction = norm_vec(next_node_position - self.position)
        
    def get_closest_node(self, from_node):
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
        for i in self.customer_nodes:
            if nx.dijkstra_path_length(self.knowledge['map'], from_node, i) < distance:
                distance = nx.dijkstra_path_length(self.knowledge['map'], from_node, i)
                closest_node = i
        return closest_node



def get_edge(G, from_node, to_node):
    edge = edges.loc[(edges['from']==from_node)&(edges['to']==to_node)]
    return edge['EdgeName'].iloc[0]

def norm_vec(a):
    return (np.array(a) / np.linalg.norm(np.array(a))).flatten()



def get_adj_node_position(position):
    # return the position of the closest adjacent node
    adj_node_index = (((nodes.iloc[:, 1:] - position )**2).sum(axis=1)).idxmin()
    adj_node = nodes.loc[adj_node_index, 'NodeName']
    adj_node_position = nodes.loc[nodes['NodeName']==adj_node, ['x', 'y']].to_numpy().flatten()
    return adj_node, adj_node_position


def move_rider(rider_set, knowledge_set, t_resolution, matched_rider_IDs, matched_batches):
    for i in range(len(rider_set)):
        rider_i = rider_set[i]
        knowledge_i = knowledge_set[i]

        if rider_i.ID in matched_rider_IDs:
            # if this rider is mathced, update if_matched
            rider_i.if_matched = True if len(knowledge_i['customer_nodes'])>0 else False
            # find corresponding index of rider in matched_rider_IDs
            batch_index = matched_rider_IDs.index(rider_i.ID)
            # get the corresponding matched batch and update to knowledge
            knowledge_i['customer_nodes'] = list(matched_batches[batch_index, :])
            # move rider with knowledge
            rider_i.move(t_resolution, knowledge_i)
        else:
            rider_i.move(t_resolution)
    
    return rider_set

