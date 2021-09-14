import math
import numpy as np


def hai_single(xi, xj, hier):
    a_node = find_closet_ancestor(xi, xj, hier)

    s = 0 if a_node is None else hier.node_size[a_node]
    return s
    
def hai(n, h1, h2):
    h_score = 0.
    
    for i in range(n):
        for j in range(n):
            s1 = hai_single(i, j, h1)
            s2 = hai_single(i, j, h2)
            
            h_score += math.abs(s1 - s2)

    return 1 - h_score / (n ** 2)

def find_closet_ancestor(xi, xj, hier):
    traj1 = hier.traj(xi)
    traj2 = hier.traj(xj)
    
    a_node = min(set(traj1) & set(traj2))
        
    if a_node == hier.parents[xi] == hier.parents[xj]:
        a_node = None
    
    return a_node 
