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


class rider:
    def __init__(self, config, dec_var, merchant_node_set, rd_st=np.random.RandomState(42)):
        # config is a dictionary
        self.ID = config['ID']                                  # scalar
        self.position = config['initial_position']              # 1-D array, [x, y]
        self.maxspeed = config['maxspeed']                      # scalar
        self.stop_time = 0                                      # scalar, float
        self.state = 'idle'                                     # string 'idle' or 'working' or 'stop'
        self.speed = self.maxspeed / 2
        self.if_matched = False
        self.customer_nodes = []
        self.newly_finished_destination = None
        self.merchant_node = None
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

        self.prev_node = adj_node  # possible to contain current pos
        self.next_node = self.rd_st.choice([i for i in G_sgp.neighbors(self.prev_node)])
        self.update_idle_destination()  # update destination, path_to_dest, nextnext_node
        self.link = get_link(G_sgp, self.prev_node, self.next_node)

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

    def update_att_according_to_the_next_cust(self):
        self.update_customer_order()
        # rider merchant_node has been updated in function "move_rider"
        self.destination = self.merchant_node
        self.path_to_dest = nx.dijkstra_path(G_sgp, self.next_node, self.destination)

        # next node is the last node, then the next next node is random, else, nextnext node is the second node of the path
        self.nextnext_node = self.rd_st.choice([i for i in G_sgp.neighbors(self.next_node)]) if self.next_node == self.path_to_dest[-1] else self.path_to_dest[1]

        # when matched, set the driver to be unmatchable
        self.if_matchable = False

    def check_distance_2_closest_merchant(self):
        self.closest_merchant_node = get_closest_node(self.position, self.merchant_node_set)

        closest_merchant_position = get_node_xy(self.closest_merchant_node)
        if np.linalg.norm(self.position - closest_merchant_position) * 111 <= self.dec_var['r']:  # from lon-lat system to km
            self.if_matchable = True
        else:
            self.if_matchable = False

    def move(self, travel_distance_mag, t_resolution, dec_var):
        self.newly_finished_destination = None
        self.dec_var = dec_var
        # move one time step foward
        if self.state == 'idle':
            self.check_distance_2_closest_merchant()
            if self.if_matched:
                self.state = 'working'
                self.speed = self.maxspeed
                # customer_nodes has been updated in function "move_rider"
                self.update_att_according_to_the_next_cust()
        elif self.state == 'working':
            None
        elif self.state == 'stop':
            self.stop(t_resolution)
            return

        # the current link
        link_geometry = self.link['geometry']
        # first and last node of the geometry of this link
        first_point, last_point = link_geometry.coords[0], link_geometry.coords[-1]
        first_node = nodes.loc[(nodes['x'] == first_point[0]) & (nodes['y'] == first_point[1]), 'NodeName'].iloc[0]
        last_node = nodes.loc[(nodes['x'] == last_point[0]) & (nodes['y'] == last_point[1]), 'NodeName'].iloc[0]

        # whether the direction of the link is the reverse of our traveling direction
        reverse = (self.prev_node != first_node) and (self.next_node != last_node)
        if reverse:
            link_geometry = shapely.geometry.LineString(list(link_geometry.coords)[::-1])

        # the remaining length from current location to the next node
        distance_to_next_node = link_geometry.length - link_geometry.project(shapely.geometry.Point(self.position))

        if travel_distance_mag < distance_to_next_node:
            # self.position = self.position + travel_distance_mag * self.direction
            new_position_point = link_geometry.interpolate(link_geometry.project(shapely.geometry.Point(self.position)) + travel_distance_mag)
            self.position = np.array([new_position_point.x, new_position_point.y])
        elif travel_distance_mag >= distance_to_next_node:
            # the residual distance to travel
            travel_distance_mag -= distance_to_next_node

            # travel to the next node
            next_node_position = get_node_xy(self.next_node)
            self.position = next_node_position
            self.prev_node = self.next_node
            self.next_node = self.nextnext_node

            # update nextnext_node
            if self.prev_node == self.destination:  # prev_node is the current node
                # arrive the destination
                # give up the residual distance, and stop
                if self.state == 'working':
                    self.stop(t_resolution)
                elif self.state == 'idle':
                    self.nextnext_node = self.rd_st.choice([i for i in G_sgp.neighbors(self.next_node)])
                else:
                    None
            else:
                if self.next_node != self.destination:
                    # the next_node is not the destination
                    self.nextnext_node = self.path_to_dest[self.path_to_dest.index(self.next_node) + 1]
                elif self.next_node == self.destination:
                    # the next_node is the destination, then randomly choose on for
                    # nextnext_node
                    self.nextnext_node = self.rd_st.choice([i for i in G_sgp.neighbors(self.next_node)])
                else:
                    None

                self.move(travel_distance_mag, t_resolution, dec_var)
        else:
            None

        if self.state == 'idle':
            self.update_idle_destination()
        else:
            None

        # finish moving, add total_time and update link
        self.total_time += t_resolution
        self.link = get_link(G_sgp, self.prev_node, self.next_node)

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

    def update_next_desination(self, t_resolution):
        next_destination = self.customer_nodes[0] if len(self.customer_nodes) > 0 else None
        if next_destination is None:
            # complete
            self.complete()
            return

        try:
            self.path_to_dest = nx.dijkstra_path(G_sgp, self.prev_node, next_destination)  # prev_node is current location
        except Exception:
            print('Removed customer No. %i becuase of no path from %i'%(next_destination, self.prev_node))
            # if there is no path from current position to the destination
            self.customer_nodes.remove(self.customer_nodes[0])
            # update the destination again
            self.update_next_desination(t_resolution)
            return

        next_dest_i = self.customer_nodes.index(next_destination)
        # udpate customer_nodes
        new_customer_nodes = []
        new_customer_nodes.extend(self.customer_nodes[:next_dest_i])
        new_customer_nodes.extend(self.customer_nodes[next_dest_i+1:])
        self.customer_nodes = new_customer_nodes

        self.destination = next_destination

        # update the next_node, and nextnext_node
        if len(self.path_to_dest) == 1:
            self.stop(t_resolution)
            return
        else:
            self.next_node = self.path_to_dest[1]

        # if only 2 nodes, then nextnext node is random
        self.nextnext_node = self.path_to_dest[2] if len(self.path_to_dest) > 2 else self.rd_st.choice([i for i in G_sgp.neighbors(self.next_node)])
        next_node_position = get_node_xy(self.next_node)
        self.direction = norm_vec(next_node_position - self.position)

    def complete(self):
        # complete all orders in current bundle
        print('rider %i completed! time:%.2f' % (self.ID, self.total_time))
        self.update_idle_destination()

        self.state = 'idle'
        self.speed = self.maxspeed / 2
        self.if_matched = False
        self.customer_nodes = []
        self.merchant_node = None
        self.total_time_rec.append(self.total_time)
        self.total_time = 0
        return

    def update_idle_destination(self):
        self.closest_merchant_node = get_closest_node(self.position, self.merchant_node_set)
        try:
            self.destination = self.closest_merchant_node
            # previous node is the current node
            self.path_to_dest = nx.dijkstra_path(G_sgp, self.next_node, self.destination)
            # update the next_node, and nextnext_node
            if len(self.path_to_dest) == 1:
                self.nextnext_node = self.rd_st.choice([i for i in G_sgp.neighbors(self.next_node)])
            elif len(self.path_to_dest) > 1:
                self.nextnext_node = self.path_to_dest[1]
            else:
                None
        except:
            # if there is no route
            # update the nextnext_node
            self.nextnext_node = self.rd_st.choice([i for i in G_sgp.neighbors(self.next_node)])
            self.destination = self.next_node
            self.path_to_dest = nx.dijkstra_path(G_sgp, self.next_node, self.destination)


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
            travel_distance_mag = rider_i.speed * t_resolution
            rider_i.move(travel_distance_mag, t_resolution, dec_var)
        else:
            travel_distance_mag = rider_i.speed * t_resolution
            rider_i.move(travel_distance_mag, t_resolution, dec_var)

    return rider_set


# some useful utility functions
def norm_vec(a):
    return (np.array(a) / np.linalg.norm(np.array(a))).flatten()


def get_node(ID):
    node_index = nodes.loc[nodes['NodeName'] == ID].index.values[0]
    return node_index


def get_node_xy(ID):
    node_position = nodes.loc[nodes['NodeName'] == ID, ['x', 'y']].to_numpy().flatten()
    return node_position


def get_adj_node_position(position):
    # return the position of the closest adjacent node
    adj_node_index = (((nodes.iloc[:, 1:] - position)**2).sum(axis=1)).idxmin()
    adj_node = nodes.loc[adj_node_index, 'NodeName']
    adj_node_position = nodes.loc[nodes['NodeName'] == adj_node, ['x', 'y']].to_numpy().flatten()
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
    if type(from_node) == int:
        None
    elif type(from_node) == np.ndarray:
        current_node = int(
            nodes.loc[(nodes['x'] == from_node[0]) & (nodes['y'] == from_node[1]), 'NodeName'].values
        )
        from_node = current_node

    distance = 1e10
    closest_node = None
    for i in target_nodes:
        if nx.dijkstra_path_length(G_sgp, from_node, i) < distance:
            distance = nx.dijkstra_path_length(G_sgp, from_node, i)
            closest_node = i
    return closest_node


def get_link(net, node1, node2):
    '''input: node1 and node2, output: full informaition of link'''
    link = net.get_edge_data(node1, node2)
    assert link is not None
    return link[0]
