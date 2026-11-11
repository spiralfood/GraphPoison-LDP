import numpy as np

class HMM(object):
    def __init__(self, N=2, M=2, pi=None, A=None, B=None):
        
        self.N = N    
        self.M = M    
        self.pi = pi  
        self.A = A    
        self.B = B    
    
    def cal_prob(self, X):
        
        alpha = np.ones((len(X), 2))
        
        
        for t in range(0, len(X)-1):
            alpha[t+1, 0] = alpha[t, 0] * self.B[0, X[t]] * self.A[0, 0] + alpha[t, 1] * self.B[1, X[t]] * self.A[0, 1]
            alpha[t+1, 1] = alpha[t, 0] * self.B[0, X[t]] * self.A[1, 0] + alpha[t, 1] * self.B[1, X[t]] * self.A[1, 1]
        
        
        beta = np.zeros(2)
        beta[0] = self.B[0, X[-1]] * self.pi[0]
        beta[1] = self.B[1, X[-1]] * self.pi[1] 

        
        P_O = alpha[-1, 0] * beta[0] + alpha[-1, 1] * beta[1]

        return (alpha[-1, 1] * beta[1]) / P_O
