import numpy as np
import logging
import logging.config
from scipy import sparse
from common.stats import softmax, linear
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from .basepolicy import BasePolicy
from scipy.stats import multivariate_normal


logging.config.fileConfig('logging.conf')
lg = logging.getLogger('simpleLogger')

def euclidean_sim(x, y, sigma=None, epsilon=.0001):
    return 1 / (np.power(x-y, 2).mean() + epsilon)

def cosine_sim(x, y):
    return 1 - spatial.distance.cosine(x, y)

class GreedyPolicy(BasePolicy):

    def __init__(self, hier, *args, **kwargs):
        super(GreedyPolicy, self).__init__(hier)

        # for cifar100
        # self.temperature = 0.0001

        # for Amazon-large
        self.temperature = 0.01

    def similarity(self, x, ys, method='cosine', style='one'):
        if method == 'cosine':
            sim = cosine_sim
        elif method == 'euclidean':
            sim = euclidean_sim
        elif callable(method):
            sim = method
        else:
            raise Exception("The currently supported values of for method "
                            "are 'cosine' or 'euclidean'")

        if style == 'all':
            ds = [sim(x, y) for y in ys]
            #print(ds, np.mean(ds))
            return np.mean(ds)
        elif style == 'one':
            y = np.asarray(ys.mean(axis=0)).squeeze()
            return sim(x, y)
        else:
            raise Exception("The currently supported values of for style "
                            "are 'all' or 'one'")

    def eta(self, sims, l=1):
        ws = softmax(sims, self.temperature*l)
        #print(ws, sims)
        return ws

    def reward_stay(self, node_n, n):
        #return 1 - (node_n / n)
        return 1 - (np.power(np.e, node_n/n) - 1) / (np.e - 1)

    def get_path(self, ind):
        parents = self.hier.parents
        path = []
        parent = ind
        while parent < self.hier.m + self.hier.n:
            path.append(parent)
            parent = parents[parent].item()
        return path[::-1]

    def policy(self, target, **kwargs):
        sim_method = kwargs.get('distance_method', 'cosine')
        sim_style = kwargs.get('distance_style', 'all')

        rwd = self.reward_stay
        sim = self.similarity

        hier = self.hier
        get_items_at_node = hier.get_items_at_node

        m, n = hier.m, hier.n
        sibs = hier.siblings
        node_size = hier.node_size
        X = hier.X.tocsr()

        path = self.get_path(target)
        leaf_level = len(path) - 1

        chds = []
        path_p = 1.             # the probability of the current path
        x = X[target].toarray().squeeze()
        stop_at = -1
        r_stop, p_stop = 0., 0.
        V_hat = V_next_hat = 0.
        #lg.info('\n {}'.format(target))
        for i, node in enumerate(path[1:], 1):
            #lg.info('\n')
            #lg.info('level {}, node: {}'.format(i, node))
            rc = node           # rc is the child on the right path
            wcs = sibs[rc]      # wc is the child on the wrong path

            #print(node, n)
            chds = [node-n]
            rc_indices = list(set(get_items_at_node(rc)) - set([target]))
            #rc_indices = get_items_at_node(rc)

            if rc_indices:
                rc_sim = sim(x, X[rc_indices].toarray(), sim_method, sim_style)

            else:
                rc_sim = 1.

            sims = [rc_sim]
            c_sizes = [len(rc_indices)+1]
            if not wcs:
                probs = np.array([1.])
            else:
                for wc in iter(wcs):
                    #print(wc, n)
                    chds.append(wc-n)
                    wc_indices = hier.get_items_at_node(wc)
                    wc_sim = sim(x, X[wc_indices].toarray(),
                                 sim_method, sim_style)

                    sims.append(wc_sim)
                    c_sizes.append(len(wc_indices))

                probs = self.eta(sims, 1)

            node_rwds = list(map(lambda x: rwd(x, n), c_sizes))

            #lg.info('children are {}'.format(chds))
            #lg.info('similarities are {}'.format(sims))
            #lg.info('probs are {}'.format(probs))
            #lg.info('beliefs for the right states are {}'.format(probs * path_p))
            #lg.info('rewards are {}'.format(node_rwds))
            s = 0.
            for j in range(len(probs)):
                #lg.info('{} {} {}'.format(j, probs[j], node_rwds[j]))
                tmp_s = self.value(probs[j], node_rwds[j])
                s += tmp_s
                #lg.info('value of staying at child {},    p: {}   value: {}'
                #        .format(chds[j], probs[j], tmp_s))

            V_next_hat = sum(p*self.value(path_p*p, r)
                             for p, r in zip(probs, node_rwds))
            #lg.info('Vt: {}  Vt+1: {}'.format(V_hat, V_next_hat))
            if V_hat > V_next_hat or i == leaf_level:
                stop_at = i
                #r_stop = node_rwds[0]
                #path_p *= probs[0]
                p_stop = path_p
                #lg.info('target {} stops at level {}'.format(target, stop_at))
                break

            r_stop = node_rwds[0]
            path_p *= probs[0]
            V_hat = self.value(path_p, node_rwds[0])

        return stop_at, p_stop, r_stop

    def eval_hier_4_target(self, target, **kwargs):
        stop_at, p, r = self.policy(target, **kwargs)
        #lg.info("eval results: {} {} {}".format(stop_at, p, r))
        return stop_at, p, r, self.value(p, r)

    def value(self, prob, reward):
        ''' a short cut function for oracle value
        '''
        return prob * (reward + 1) - 1
