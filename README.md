# Food Delivery System Simulator
## Updated on Feb. 3, 2023
Author: K.Z.

Agent based model.

## Deomonstration animation
N_v represents number of idle riders, N_b represents number of accumulated order batches, p is the customer matching probability, pp is the rider matching probability. The merchant node size represents the number of accumulated orders in this merchant.

<img src="./res_img/demo.gif" width="750">

## Features
### How riders get their routes?
Dijkstra method
### How riders decide which customer to serve next?
Shortest Dijkstra distance
### How riders decide where to go when idle?
Random walk (randomly select the next node and the next next node)

## Rider, platform, and customer attributes
### Rider attributes

|Attributes|Type|Note|Value|
|---|---|---|---|
|ID|int|specified on generation|unchanged|
|position|1D array|x y coordinate of this rider|updated on every move (every iteration)|
|direction|1D array|the direction of this rider, unit vector|updated on initialization (random), moving (when next node changes), matched (next customer), arrival of customer (next customer), complete batch (to be random)|
|state|string|the state of this rider|'idle', 'working' or 'stop'|
|speed|int|speed in km/hr|maxspeed when working, half maxspeed when idle|
|maxspeed|int|specified on generation|unchanged|
|stop_time|float|accumulated stop time|0 when moving, increased on every iteration when stop, back to 0 when restarting|
|total_time|float|accumulated (total) time spent on the current batch (pickup time + total delivery time)|initially 0, increase from matched (start working), bakc to 0 when complete|
|total_time_rec|list|record of each total_time|a list that stores every value of total_time|
|next_node|int|the next node that this rider may travel by|determined by the next node on path or randomly chosen|
|nextnext_node|int|the next node that this rider may travel by|determined by the next node on path or randomly chosen|
|destination|int|the destination of this current routing|can be merchant_node or customer_node, is None when idle|
|path|list|a list of nodes by order|obtained by dijkstra method, is None when idle|
|closest_merchant_node|int|the closest merchant node ID (by distance)|closet merchant node ID, changed according to distance to each merchant|
|merchant_node|int|the merchant node ID that this rider is currently serving|given by platform after being matched, is None when idle|
|merchant_node_set|int|set of all merchant nodes|unchanged|
|customer_nodes|list|the unserved customers (the customer that this rider is currently heading for is not included)|updated on every arrival of destinations|
|finished_destination|list|list of finished destination, including merchant node ID and customer node IDs|updated on every arrival of destination, is empty when initalized, back to empty again when updated customer information again|
|if_matched|boolean|if the rider is matched|is True after being matched, if False after complete serving the batch|
|if_matchable|boolean|if the rider lies in the matching area, it is matchable|updated on every move, is True if the distance to the closest merchant is less than the maximum mathcing radius, False otherwise|
|dec_var|dict|decision variables, contains r cR k t N q_bar|speficied by user|

## Platform attributes

|Attributes|Type|Note|Value|
|---|---|---|---|
| customer_df | pd.DataFrame | a table of unmatched customers, columns: ['node_ID', 'merchant_node', 'state', 'waiting_time', 'position_x', 'position_y'] | add unmatched customer(s) on each iteration (order generation), remove (to matched_customer_df) on each matching |
| matched_customer_df | pd.DataFrame | a table of matched customers, columns: ['node_ID', 'merchant_node', 'state', 'waiting_time', 'position_x', 'position_y'] | add matched customers on each mathing, update (reduce) customer(s) on each iteration (if there is anyone delivered) |
| num_accumulated_order | float | number of accumulated orders, continuous number | updated on every iteration (order generation) |
| r | float | one of the decision variables | assigned by user |
| cR | float | one of the decision variables | assigned by user |
| k | float | one of the decision variables | assigned by user |
| t | float | one of the decision variables | assigned by user |

## Customer attributes
### Customers are defined and stored in class "platform"
|Attributes|Type|Note|Value|
|---|---|---|---|
| node_ID | int | ID of this customer | assigned on generation |
| merchant_ID | int | the merchant where the customer ordered meal from | assigned on generation |
| state | string | the state of this customer | "waiting" or "matching", meaning this customer is waiting for matching or has been matched respectively, updated on each matching process |
| waiting_time | float | the total waiting time (for matching and meal delivery) for this customer | initially 0, increase over time |
| position_x | float | the x coordinate of this customer | assigned on generation |
| position_y | float | the y coordinate of this customer | assigned on generation |


