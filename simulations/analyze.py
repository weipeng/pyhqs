import pickle
import pandas as pd
from time import time
from common.util import create_path
from hierarchies.hierarchy import Hierarchy
from hierarchies.treeutil import tree_to_Z
from policies import GreedyPolicy
from evaluation.eval import Evaluator
from scipy.sparse import csr_matrix, load_npz
from datahandler import load_cluster, load_json_Z


def analyze_amazon(index, parallel=False, num_thd=2, backend='multiprocessing',
                   **kwargs):
    X = load_npz('./data/amazon_hqs.npz')

    Z = load_json_Z('./data/amazon_tree_{}_Z.json'.format(index))

    scores = evaluation(X, Z, parallel, num_thd, backend, **kwargs)
    df = pd.DataFrame(scores, columns=['stop_at', 'prob', 'reward', 'V'])

    path = './output/scores/amazon_{}.csv'.format(index)
    create_path(path)
    df.to_csv(path, index=False)

    return scores


def analyze_amazon_large(index, parallel=False, num_thd=2, 
                         backend='multiprocessing', **kwargs):
    df = pd.read_csv('./data/amazon_hqs_large_scale_values.csv', sep='\t', header=None)
    X = csr_matrix(df.iloc[:, 1:].values)
    del df

    Z = load_json_Z('./output/amazon-large_agg/amazon-large_agg_{}_Z.json'
                    .format(index))

    scores = evaluation(X, Z, parallel, num_thd, backend, **kwargs)
    df = pd.DataFrame(scores, columns=['stop_at', 'prob', 'reward', 'V'])

    path = './output/scores/amazon_agg_{}.csv'.format(index)
    create_path(path)
    df.to_csv(path, index=False)

    return scores

def analyze_amazon_large_rand(index, parallel=False, num_thd=2, 
                              backend='multiprocessing', **kwargs):
    df = pd.read_csv('./data/amazon_hqs_large_scale_values.csv', sep='\t', header=None)
    X = csr_matrix(df.iloc[:, 1:].values)
    del df

    Z = load_json_Z('./output/amazon-large_randagg/amazon-large_randagg_{}_Z.json'
                    .format(index))

    scores = evaluation(X, Z, parallel, num_thd, backend, **kwargs)
    df = pd.DataFrame(scores, columns=['stop_at', 'prob', 'reward', 'V'])

    path = './output/scores/amazon_randagg_{}.csv'.format(index)
    create_path(path)
    df.to_csv(path, index=False)

    return scores

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def analyze_cifar100(index, parallel=False, num_thd=2, isrand=False,
                     backend='multiprocessing', **kwargs):
     
    #train = unpickle("./data/cifar-100-python/train")
    #X = train[b'data']
    test = unpickle("./data/cifar-100-python/test")
    X = csr_matrix(test[b'data'])

    rd_str = 'rand' if isrand else ''
    
    fn = ('./output/cifar100_{}agg/cifar100_{}agg_{}_Z.json'
            .format(rd_str, rd_str, index))

    print('loading {}'.format(fn))
    Z = load_json_Z(fn)

    scores = evaluation(X, Z, parallel, num_thd, backend, **kwargs)
    df = pd.DataFrame(scores, columns=['stop_at', 'prob', 'reward', 'V'])

    path = './output/scores/cifar100_{}agg_{}.csv'.format(rd_str, index)
    create_path(path)
    df.to_csv(path, index=False)

    return scores


def evaluation(X, Z, parallel=False, num_thd=2,
               backend='multiprocessing', **kwargs):
    hier = Hierarchy(X, Z)
    if kwargs.get('policy', None) is None:
        policy = GreedyPolicy(hier)

    if parallel:
        score = Evaluator.parallel_measure(policy, num_thd, backend)
    else:
        score = Evaluator.measure(policy, **kwargs)

    return score

if __name__ == '__main__':
    import sys

    dname = 'amazon'
    index = 1
    loops = 1


    if len(sys.argv) == 1:
        dname = 'amazon'
    elif len(sys.argv) == 2:
        dname = sys.argv[1]
    elif len(sys.argv) == 3:
        dname = sys.argv[1]
        index = int(sys.argv[2])
    else:
        dname = sys.argv[1]
        index = int(sys.argv[2])
        loops = int(sys.argv[3])

    if dname.lower() == 'amazon': 
        analyze_amazon(index, parallel=False, distance_method='cosine',
                       distance_style='all')
    elif dname.lower() == 'amazon-large':
        ts = []
        for i in range(loops):
            s = time()
            analyze_amazon_large(index, parallel=True, num_thd=16,
                                 distance_method='euclidean',
                                 distance_style='one')
            e = time()
            t = e - s 
            ts.append(t)
        print(ts)
        pd.DataFrame(ts).to_csv('amazon-large-running-time.csv', sep='\t')
    elif dname.lower() == 'amazon-large-rand':
        ts = []
        for i in range(loops):
            s = time()
            analyze_amazon_large_rand(index, parallel=True, num_thd=16,
                                      distance_method='euclidean',
                                      distance_style='one')
            e = time()
            t = e - s
            ts.append(t)
        print(ts)
        pd.DataFrame(ts).to_csv('amazon-large-rand-running-time.csv', sep='\t')
    elif dname.lower() == 'cifar100': 
        ts = []
        for i in range(loops):
            s = time()
            analyze_cifar100(index, parallel=True, num_thd=16,
                             distance_method='euclidean',
                             distance_style='one')
            e = time()
            t = e - s 
            ts.append(t)
        print(ts)
        pd.DataFrame(ts).to_csv('cifar100-running-time.csv', sep='\t')
    elif dname.lower() == 'cifar100-rand': 
        ts = []
        for i in range(loops):
            s = time()
            analyze_cifar100(index, parallel=True, num_thd=16, 
                             isrand=True,
                             distance_method='euclidean',
                             distance_style='one')
            e = time()
            t = e - s 
            ts.append(t)
        print(ts)
        pd.DataFrame(ts).to_csv('cifar100-running-time-randAC.csv', sep='\t')
    else:
        raise("'{}' is not supported".format(dname))

