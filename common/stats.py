import numpy as np
from scipy.special import softmax as sp_softmax


def softmax(x, temperature=1):
    ''' temperature = 1 leads the softmax to the most commonly seen version
    '''
    x = np.array(x, dtype=np.float64) / temperature
    x = sp_softmax(x)
    return x 


def linear(x, epsilon=.0001):
    x = np.array(x, dtype=np.float64) + epsilon
    x /= x.sum()
    return x 
