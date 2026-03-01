## Last code update on Feb. 4, 2023
Author: K.Z.

Agent-based simulation. CPU multiprocessing for current demos.

## Demonstration animation
N_v represents the number of idle drivers (drivers), N_b represents the number of accumulated order batches, p is the customer matching probability, pp is the driver matching probability. The merchant node size represents the number of accumulated orders in this merchant.

In this demo, the max matching radius r=1 km, max delivery radius R=2 km, batch size (bundling ratio) k=3, and matching interval t=0.005 hour. The order arrival rate $\overline{q}=800$ orders/hour and the total number of drivers is $N=200$. 5 merchants are spread in the city of Singapore.

<img width="889" alt="FD simulator demo" src="https://github.com/khzhang2/khzhang2.github.io/assets/38817831/f5bdf55c-ab16-42f3-98a3-044b348a88c9">

The Singapore demo (real road network) is placed on a Chinese video stream platform Bilibili: [link](https://bilibili.com/video/BV1uM4y1474J/?spm_id_from=333.999.0.0)

## Github repository [here](https://github.com/khzhang2/FD_Simulator)

## Features
### How do we match the drivers (aka drivers) to customers?
Batch matching (see this [paper](https://doi.org/10.1016/j.trb.2019.11.005) for an introduction). The customers (of a merchant) are not eligible for matching until the platform has accumulated k orders for this merchant. These k orders are bundled into a batch and matched to one driver. The batch matching process happens every ceiling(t/t_resolution) iteration, where t means the matching interval and t_resolution is the time duration for each iteration.

### How do we compute the optimal routes of drivers?
Dijkstra algorithm. We assume fixed travel time on each link.

<!-- ### How do we consider the drivers' decisions on which customer to serve next?
Follow the order of the customer_nodes, which is pre-defined when updating the customer information on matching. The order is, (2nd closest, 3rd closest, ..., farthest, closest), to make sure the driver will go back (at least be close to) this merchant. -->

<!-- ### How do we consider drivers' decisions on where to go when they are idle?
Random walk (randomly select the next node and the next next node) and wait for being matched. -->

<!-- ### What do we consider drivers' decisions when they've completed delivering a bundle of orders?
Random walk and wait for being matched again. -->

### What is the delivery strategy of drivers?
We allow drivers to carry multiple orders at the same time, which is termed as the bundling delivery. The bundling mechanism is as follows:
- Drivers tend to bundle as many as possible until at least one order is “approaching” its ETA
- The next order that is about to be accepted should not make any existing orders be undeliverable before ETA
- If not, they should otherwise start to delivery the most urgent order
- Repeat the process over and over

Due to market friction, there must exist some drivers not assigned any orders. These drivers are in idle state and they do random walk to "search" for potential orders.

### How do we generate customer demands?
Randomly generate $n_q$ (in codes, it is called num_generated_order) customers on every iteration. $n_q$ is the number of generated customers in this iteration, determined by q_bar, R, and Delta, representing the potential hourly customer demand, maximum delivery radius, and attractiveness radius of merchants. The platform will randomly choose 1 node from all the nodes in the network as the customer node ID (assuming the probability that the new customer will be located at each node is the same for all nodes) and 1 node from the merchant node set as merchant node ID. Then, the platform will check whether the distance from this customer to her corresponding merchant is less than the maximum delivery distance, R. If yes, keep it, and remove it otherwise. Then repeat the process until the number of newly generated customers reached $n_q$.

## Driver, platform, and customer attributes
### Driver attributes
|Attributes|Type|Note|Value|
| --- | --- | --- | --- |
|ID i nt|specified on generation|unchanged|
|position                       | 1D array|x y coordinate of this driver|updated on every move (every iteration)|
|state                          | string|the state of this driver|'idle', 'working' or 'stop'|
|speed                          | int|speed in km/hr|maxspeed when working, half maxspeed when idle|
|maxspeed                       | int|specified on generation|unchanged|
|stop_time                      | float|accumulated stop time|0 when moving, increased on every iteration when stop, back to 0 when restarting|
|total_time                     | float|accumulated (total) time spent on the current batch (pickup time + total delivery time)|initially 0, increasing from matched (start working), back to 0 when complete|
|total_time_rec                 | list|record of each total_time|a list that stores every value of total_time|
|prev_node                      | int|the previous node that this driver traveled through|equals the next node when travel pass the next node|
|next_node                      | int|the next node that this driver may travel by|determined by the next node on path or randomly chosen|
|nextnext_node                  | int|the next node that this driver may travel by|determined by the next node on path or randomly chosen|
|destination                    | int|the destination of this current routing|can be merchant_node or customer_node, is None when idle|
|path_to_dest                   | list|a list of nodes by order|obtained by Dijkstra method, is None when idle|
|link                           | dict|a dictionary records the information of the current link this driver is traveling on|
|closest_merchant_node          | int|the closest merchant node ID (by distance)|closet merchant node ID, changed according to distance to each merchant|
|merchant_node                  | int|the merchant node ID that this driver is currently serving|given by the platform after being matched, is None when idle|
|merchant_node_set              | int|set of all merchant nodes|unchanged|
|customer_nodes                 | list|the unserved customers (the customer that this driver is currently heading for is not included), order:(2nd closest, 3rd closest, ..., farthest, closest)|updated on every arrival of destinations|
|newly_finished_destination     | int|the new finished destination, merchant node ID or customer node ID|updated on every completion of stop, is None when working or idle|
|if_matched                     | boolean|if the driver is matched|is True after being matched, if False after complete serving the batch|
|if_matchable                   | boolean|if the driver lies in the matching area, it is matchable|updated on every move, is True if the tider state is idle and the distance to the closest merchant is less than the maximum matching radius, False otherwise|
|dec_var                        | dict|decision variables, contains r cR k t N q_bar|speficied by user|
|rd_st                          | object|Global random state| n/a |


### Platform attributes

|Attributes|Type|Note|Value|
|---|---|---|---|
| customer_df           | pd.DataFrame | a table of unmatched customers, columns: ['node_ID', 'merchant_node', 'waiting_time', 'position_x', 'position_y'] | add unmatched customer(s) on each iteration (order generation), remove (to matched_customer_df) on each matching |
| matched_customer_df   | pd.DataFrame | a table of matched customers, columns: ['node_ID', 'merchant_node', 'waiting_time', 'position_x', 'position_y'] | add matched customers on each matching, update (reduce) customer(s) on each iteration (if there is anyone delivered) |
| num_accumulated_order | float | number of accumulated orders, continuous number | updated on every iteration (order generation) |
| r                     | float | one of the decision variables | assigned by user |
| cR                    | float | one of the decision variables | assigned by user |
| k                     | float | one of the decision variables | assigned by user |
| t                     | float | one of the decision variables | assigned by user |

### Customer attributes
#### Customers are defined as pd.DataFrame and are stored in class "platform"

|Attributes|Type|Note|Value|
|---|---|---|---|
| node_ID               | int | ID of this customer | assigned on generation |
| merchant_ID           | int | the merchant where the customer ordered meal from | assigned on generation |
| waiting_time          | float | the total waiting time (for matching and meal delivery) for this customer | initially 0, increase over time |
| position_x            | float | the x coordinate of this customer | assigned on generation |
| position_y            | float | the y coordinate of this customer | assigned on generation |

## Driver, platform behaviors
### Driver behaviors

|Behavior name                      |Description|When excecute|sup behavior(s)|sub behavior(s)|
|---                                |---|---|---|---|
|__init__                           |Initialize the driver as a idle driver, define attributes|On the generation of this driver|n/a||
|update_customer_order              |Update the order of the customers|After being matched, after updating merchant node ID and customer nodes|update_customer_nodes|n/a|
|update_customer_nodes              |Update the attributes according to the customers that this driver is going to serve (customer nodes are given in function "move_driver", which is not in this class)|After being matched|move|n/a|
|check_distance_2_closest_merchant  |Check the distance from this driver to the closest merchant. If it is less than the maximum matching radius, if_matchable is True, otherwise, if_matchable is False|after move (position change)|move|n/a|
|move                               |Move the driver to the next position, and update the corresponding attributes|every iteration (after order generation and matching)|n/a|update_customer_nodes, stop, check_distance_2_closest_merchant|
|stop                               |Stop at the current position, and update the corresponding attributes|When the stop time is below the threshold or when the driver arrives at the destination (can be merchant or customer)|move, update_next_desination|update_next_desination|
|complete                           |Complete serving the batch, and update the corresponding attributes|When arrived the last destination (the next destination is None)|update_next_desination|n/a|
|update_next_desination             |Update the next destination and corresponding attributes|After completed stop|stop|complete|
|update_idle_destination             |Update the next destination and corresponding attributes|After completed stop|__init__, complete, move| / |

### Platform behaviors

|Behavior name                      |Description|When excecute|sup behavior(s)|sub behavior(s)|
|---                                |---|---|---|---|
|update_cust_df_with_new_cust       |Update the attribute customer_df with newly generated customers|When (quickly after) order generation|acquire_order|n/a|
|acquire_order                      |Generate new orders, only those who lie in the delivery area will be kept|At the beginning of each iteration|n/a|update_cust_df_with_new_cust|
|update_matched_order               |Update attribute matched_customer_df with matched orders (by node ID, if multiple customers locate on one node, label the one with the longest waiting time as matched), and remove the matched customers from the attributes customer_df|After complete matching|match|n/a|
|match                              |Match the idle drivers to customers (and corresponding merchant, only when the closest merchant of this driver is the corresponding merchant can it be matched, and of course, its if_matchable attribute should be True), batch matching|Every matching period|n/a|update_matched_order|

