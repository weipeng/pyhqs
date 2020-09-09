import pandas as pd
from common.util import create_path
from hierarchies.hierarchy import Hierarchy 
from hierarchies.treeutil import tree_to_Z
from evaluation.eval import Evaluator
from scipy.sparse import csr_matrix
from policies import GreedyPolicy
from datahandler import load_cluster


def eval_clust(data_name, method, data=None, index=0,
               parallel=False, num_thd=2, backend='multiprocessing',
               **kwargs):
    ''' method: {'ap', 'kmeans', ... }
        data: if data is None, we need to process it with the TFIDF score
        parallel: the switch for single or multi- processing for the evaluation
        num_thd: number of threads (only valid when parallel is true)
        backend: only valid when parallel is true
    '''
    if data is None:
        data = load_mat_data('./data/{}/{}_tfidf.mat'.format(data_name, data_name))
        X = data['tfidf']
    else:
        X = data

    if 'kmeans' in method:
        method_tmp = 'kmeans'
    else:
        method_tmp = method

    try:
        tree, Z = load_cluster('./output/results/{}_{}'.format(data_name, method), 
                               '{}_{}_{}'.format(data_name, method_tmp, index))
    except:
        tree, Z = load_cluster('./output/results/{}_{}'.format(data_name, method),
                                '{}_{}_{}'.format(data_name, method, index))
        
    
    scores = evaluation(X, Z, parallel, num_thd, backend, **kwargs)
    df = pd.DataFrame(scores, columns=['V'])
    
        
    path = ('./output/scores/test/{}_{}/stoch_{}_{}_{}.csv'
            .format(data_name, method, data_name, method, index))
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
    for data_name in ['compound']:
        data = pd.read_csv('./data/{}/{}.txt'.format(data_name, data_name),
                           sep=' +|\t|,', engine='python', header=None)
        X = csr_matrix(data.values[:, :2])
        for method in ('kmeans3',): #, 'kmeans4', 'kmeans5', 'agg', 'randagg'):
            for i in range(15):
                eval_clust(data_name, method, data=X, index=i, parallel=False,
                           distance_method='euclidean', distance_style='one',   
                           backend='threading')
                print('Finish round {}'.format(i))
           
