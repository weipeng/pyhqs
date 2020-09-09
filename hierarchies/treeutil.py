import networkx as nx


''' m: number of non-leaf nodes
    n: number of items (or data points)
'''

def Z_to_tree(n, Z):
    edge_iter = [(i+n, node) for i, nodes in enumerate(Z)
                             for _, node in enumerate(nodes)]
    tree = nx.DiGraph()
    tree.add_edges_from(edge_iter)
    return tree
 
def tree_to_Z(m, n, tree):
    # Do not use Z = [[]] * m here, as this will make 
    # every row points to the same reference
    # so that the change in a row affects all the entries in Z
    Z = [[] for _ in range(m)]
    for s, e in tree.edges(data=False):
        if s - n < 0: continue
        Z[s-n].append(e)
    return Z

def Z_to_etetree(n, Z):
    # Convert a networkx directed graph to an ete tree
    m = len(Z)
    strs = ['' for i in range(m)]
    for i in range(n):
        strs.append(str(i+m))

    for i in range(len(Z)-1, -1, -1):
        tmp_strs = []
        for z in Z[i]:
            tmp_strs.append(strs[z])
        
        tree_str = ','.join(tmp_strs) 
        strs[i] = '({}){}'.format(tree_str, i)
    else:
        ete_tree = '({});'.format(strs[0])

    return ete_tree

def Z_to_etetree_v2(n, Z, names):
    ''' Convert a networkx directed graph to an ete tree '''
    assert n == len(names), print(n, len(names))

    m = len(Z)
    strs = np.array(['' for i in range(n+m)], dtype=object)
    strs[:n] = names[:] 

    for i in range(m):
        tree_str = ','.join(map(str, strs[Z[i]])) 
        strs[i+n] = '({}){}'.format(tree_str, i)
    else:
        ete_tree = '({});'.format(strs[-1])

    return ete_tree

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Z = [[0, 1], [2, 3]]
 
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Z = [[0, 1], [2, 3]]
    print(Z)
    tree = Z_to_tree(3, Z)
    nx.draw(tree)
    plt.show()
    print(tree)
    Z_to_etetree(3, Z)
