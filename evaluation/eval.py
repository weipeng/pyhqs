import numpy as np
from time import time
from joblib import Parallel, delayed


class Evaluator(object):

    @staticmethod
    def measure(policy, **kwargs):
        eval_hier_4_target = policy.eval_hier_4_target
        n = policy.hier.n

        R = []

        s = time()
        for i in range(n):
            res = eval_hier_4_target(i, **kwargs)
            R.append(res)
            #if i % 100 == 0:
            #    print('Quality = ', np.mean(R))
            #    e = time()
            #    print('Elapsed time for {} data points: {}'.format(i+1, e-s))
        return R

    @staticmethod
    def parallel_measure(policy, n_jobs=2, backend='multiprocessing'):
        ''' Interestingly, when using the default backend setting
            "multiprocessing",the processs may freeze in Mac even though
            this is fine in Linux. However, switching the backend to
            "threading" can solve the problem.
        '''
        eval_hier_4_target = policy.eval_hier_4_target
        n = policy.hier.n
        R = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
                     delayed(eval_hier_4_target)(i) for i in range(n))

        return R
