# Multi Level Clustering
Code base for the paper "Social Media as a Vector for Escort Ads: A Study on OnlyFans advertisements on Twitter" published in WebSci '23


To run:

First Partial Intersection is computed. Once PI clusters are found, the big clusters are filtered out and the small ones are retained for Joint Clustering. 

Then run,

```
python joint_clustering.py --communities_path /path/to/communities/ 
```
