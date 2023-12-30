# Food Delivery System Simulator
## Updated on Feb. 4, 2023
Author: K.Z.

Agent based model. Multiprocessing.

## Environment
`pip install -r requirements.txt`

## Deomonstration animation
N_v represents number of idle riders, N_b represents number of accumulated order batches, p is the customer matching probability, pp is the rider matching probability. The merchant node size represents the number of accumulated orders in this merchant.

In this demo, max matching radius r=1 km, max delivery radius R=2 km, batch size (bundling ratio) k=3, matching interval t=0.005 hour. The order arrival rate $\overline{q}=800$ orders/hour and the total number of riders is $N=200$. 5 merchants are spread in the city, in Kennedy Town, HKU, Central, Wan Chai, and Causeway Bay.


https://user-images.githubusercontent.com/38817831/219488679-ef1185e9-418f-4f86-9ec3-410203a7fb3e.mp4


## More details, tutorial, and description in [here](https://khzhang2.github.io/project/FD_simulator/)
