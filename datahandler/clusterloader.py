import os
import traceback
from joblib import Parallel, delayed
from networkx.readwrite import json_graph
from common.util import read_csv, load_json
from hierarchies.treeutil import Z_to_tree


def map_int(l):
    return list(map(int, l))

def load_csv_for_Z(fname, parallel=False, **kwargs):
    Z = []
    if not parallel:
        with read_csv(fname) as f:
            Z = [map_int(row) for row in f]
    else:
        n_jobs = kwargs.get('n_jobs', 4)
        verbose = kwargs.get('verbose', 1)  
        backend = kwargs.get('backend', 'multiprocessing')
        
        with read_csv(fname) as f:
            Z = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
                delayed(map_int)(row) for row in f)

    return Z

def load_json_Z(fname):
    Z = load_json(fname)
    return Z

def load_json_tree(fname):
    data = load_json(fname)
    return json_graph.tree_graph(data)

def load_cluster(path, name):
    name = name.rstrip('/')
    
    try:
        tree = load_json_tree(os.path.join(path, '{}_tree.json'.format(name)))
    except:
        tree = None

    Z = load_json_Z(os.path.join(path, '{}_Z.json'.format(name)))
    
    return tree, Z
