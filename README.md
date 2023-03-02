# Multi Level Clustering
Code base for the paper "Social Media as a Vector for Escort Ads: A Study on OnlyFans advertisements on Twitter" published in WebSci '23


To run:

First Partial Intersection is computed. Once PI clusters are found, the big clusters are filtered out and the small ones are retained for Joint Clustering. 

`partial_intersection.py` assumes that your connection and content graphs (matrices) have been precomputed and requires a dictionary of community membership 
information for both the connection graph and the content graph. The partial intersections are then computed and saved to the current directory.

```
python partial_intersection.py /path_to_connection_comunities/ /path_to_content_communities/
```

Then run,

```
python joint_clustering.py --communities_path /path_to_small_PI_communities/ 
```
