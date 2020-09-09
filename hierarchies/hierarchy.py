import numpy as np
from scipy import sparse
from itertools import chain


class Hierarchy(object):

    def __init__(self, X, Z, *args, **kwargs):
        ''' X: the original data of order mxn
            Z: the hierarchy 2D list 
        '''
        self.X = X
        self.Z = Z          
        self.m = len(Z)     # number of clusters
        self.n = X.shape[0] # number of data points

        # self.D = self.distance()
        self.parents = self.get_parents()
        self.item_nodes = self.get_item_nodes()
        self.node_size = self.get_node_size()
        self.siblings = self.get_siblings()

    def get_item_nodes(self):
        m, n = self.m, self.n
        mat = np.zeros((m+n, n), dtype=np.int64)
        mat[:n, :n] = np.eye(n)

        for i in range(m):
            indices = self.get_items_at_node(i+n)
            mat[i+n, indices] = 1
        
        return mat

    def get_items_at_node(self, i):
        ''' This returns a list of indices list of the items for 
            the target node i
        '''
        m, n = self.m, self.n
        Z = self.Z
        assert i <= n + m, ('The index is out of bound'
                            ' i={}, m={}, n={}'.format(i, m, n))

        if i < n:   # This is a leaf node
            return np.array([i], dtype=np.int32)

        items_list = (self.get_items_at_node(child) for child in iter(Z[i-n]))
        indices = np.fromiter(chain.from_iterable(items_list), np.int32)
            
        return indices

    def get_parents(self):
        ''' The function returns an array where each index stands for a node
            and the value for that index is the index of its parent
        '''
        m, n = self.m, self.n
        parents = np.zeros(m+n, dtype=np.int32)
        Z = self.Z
        for i, z in enumerate(Z):
            parents[z] = i + n
        parents[-1] = m + n # there is no parent for the root

        return parents
            
    def get_siblings(self):
        m, n = self.m, self.n
        Z = self.Z
        parents = self.parents
        sibs = []
        for i in range(m+n-1):
            parent_id = parents[i]
            z = set(Z[parent_id-n]) 
            z.remove(i)
            sibs.append(list(z))
        else:
            sibs.append([])

        return sibs

    def get_node_size(self):
        m, n = self.m, self.n
        Z = self.Z
        node_size = np.zeros(m+n)
        node_size[:n] = 1
        
        for i, nodes in enumerate(Z):
            node_size[n+i] = node_size[nodes].sum()

        return node_size        

    def traj(self, i): 
        assert 0 <= i < self.n

        parents = self.parents
        base = self.m + self.n

        the_traj = []
        while i < base:
            the_traj.append(i)
            i = parents[i]

        return np.array(the_traj)

