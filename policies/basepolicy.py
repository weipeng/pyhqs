

class BasePolicy(object):

    def __init__(self, hier):
        ''' m: the number of nodes 
            
            This initializes the policy with three arrays representing 
            the r, p and q respectively.
            r - the probability of staying in the current node
            p - the probability of selecting the right child
            q - the probability of choosing a wrong path
        '''
        self.hier = hier 
        self.r = self.make_r()
        self.p = self.make_p()
        self.q = self.make_q()

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        self._r = value 

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._p = value

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        self._q = value

    def make_r(self):
        pass 

    def make_p(self):
        pass

    def make_q(self):
        pass

