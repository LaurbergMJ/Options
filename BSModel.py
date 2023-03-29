from optparse import Option
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

class BSOption:
    def __init__(self, S, K, T, r, sigma, q=0):
        self.S = S
        self.K = K
        self.T = T 
        self.r = r
        self.sigma = sigma
        self.q = q 
    
    @staticmethod
    def N(x):
        return norm.cdf(x)
    
    @property
    def params(self):
        return {'S': self.S,
                'K': self.K,
                'T': self.T,
                'r': self.r,
                'q': self.q,
                'sigma': self.sigma}
    
    def d1(self):
        return (np.log(self.S/self.K) + (self.r - self.q + self.sigma**2/2)*self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma*np.sqrt(self.T)
    
    def _call_value(self):
        return self.S*np.exp(-self.q*self.T) * self.N(self.d1()) - self.K*np.exp(-self.r*self.T)*self.N(self.d2())
    
    def _put_value(self):
        return self.K*np.exp(-self.r*self.T) * self.N(-self.d2()) - self.S*np.exp(-self.q*self.T)*self.N(-self.d1())
    
    def price(self, type_ = 'C'):
        if type_ == 'C':
            return self._call_value()
        elif type_ == 'P':
            return self._put_value()
        elif type_ == 'B':
            return {'call': self._call_value(), 'put': self._put_value()}
        else:
            raise ValueError('Wrong choice of type')
    
    def _delta_call(self):
        return self.N(self.d1())
    
    def _delta_put(self):
        return -self.N(-self.d1())
    
    def _gamma(self):
        return self.N(self.d1())/(self.S*self.sigma*np.sqrt(self.T))
    
    def _vega(self):
        return self.S*np.sqrt(self.T)*self.N(self.d1())
    
    def _theta_call(self):
        p1 = -self.S*self.N(self.d1())*self.sigma / (2*np.sqrt(self.T))
        p2 = self.r*self.K*np.exp(-self.r*self.T)*self.N(self.d2())
        return p1 - p2

    def _theta_put(self):
        p1 = - self.S*self.N(self.d1())*self.sigma / (2*np.sqrt(self.T))
        p2 = self.r*self.K*np.exp(-self.r*self.T)*self.N(-self.d2())
        return p1 + p2

    def rho_call(self):
        return self.K*self.T*np.exp(-self.r*self.T)*self.N(self.d2())

    def rho_put(self):
        return -self.K*self.T*np.exp(-self.r*self.T)*self.N(-self.d2()) 
     





class Option:
    def __init__(self, type_, K, price, side):
        self.type = type_ 
        self.K = K 
        self.price = price 
        self.side = side

    def __repr__(self):
        side = 'long' if self.side == 1 else 'short'
        return f'Option(type={self.type}, K={self.K}, price={self.price}, side={self.side})'

class OptionStrat:
    def __init__(self, name, S0, params=None):
        self.name = name
        self.S0 = S0
        if params:
            self.STs=np.arange(params.get('start',0),
                               params.get('stop',S0*2),
                               params.get('by',1))
        else:
            self.STs = np.arange(0, S0*2,1)
        self.payoffs = np.zeros_like(self.STs)
        self.instruments = []

    def long_call(self, K, C, Q=1):
        payoffs = np.array([max(s-K,0) - C for s in self.STs])*Q
        self.payoffs = self.payoffs + payoffs
        self._add_to_self('call', K, C, 1, Q)

    def short_call(self, K, C, Q=1):
        payoffs = np.array([max(s-K,0) * -1 + C for s in self.STs])*Q
        self.payoffs = self.payoffs + payoffs 
        self._add_to_self('call', K, C, -1, Q)

    def long_put(self, K, P, Q=1):
        payoffs = np.array([max(K-s, 0) - P for s in self.STs])*Q 
        self.payoffs = self.payoffs + payoffs
        self._add_to_self('put', K, P, 1, Q)

    def short_put(self, K, P, Q=1): 
        payoffs = np.array([max(K-s,0)*-1 + P for s in self.STs])*Q 
        self.payoffs = self.payoffs + payoffs
        self._add_to_self('put', K, P, -1, Q)

    def _add_to_self(self, type_, K, price, side, Q):
        o = Option(type_, K, price, side)
        for _ in range(Q):
            self.instruments.append(o) 

    def plot(self, **params):
        plt.plot(self.STs, self.payoffs, **params)
        plt.title(f'Payoff Diagram for {self.name}')
        plt.fill_between(self.STs, self.payoffs,
            where=(self.payoffs > 0), facecolor='g', alpha=0.4)
        plt.fill_between(self.STs, self.payoffs, 
            where=(self.payoffs < 0), facecolor='r', alpha=0.4)
        plt.xlabel(f'$S_T$')
        plt.ylabel('Profit in $')
        plt.show()

    def describe(self):
        max_profit = self.payoffs.max()
        max_loss = self.payoffs.min() 
        print(f'Max Profit: ${round(max_profit,3)}')
        print(f'Max loss: ${round(max_loss,3)}')
        c = 0 
        for o in self.instruments:
            if o.type == 'call' and o.side==1: 
                c += o.price 
            elif o.type == 'call' and o.side == -1:
                c -= o.price 
            elif o.type == 'put' and o.side == 1:
                c += o.price 
            elif o.type == 'put' and o.side == -1: 
                c -+ o.price 
        print(f"Cost of entering position ${c}")
        