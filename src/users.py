import numpy as np
import networkx as nx 
import random


class User:
    def __init__(self, data=None):
        self.data = data         
        self.perb_degree = None  
        self.final_hmg_graphs = {}  
        
    def Random_Response(self, eps, t, delta):
        

        user_num = len(self.data[t].nodes())

        G = np.zeros((user_num, user_num), dtype=np.bool_)
        
        p = np.exp(eps) / (1 + np.exp(eps))

        all_list = np.arange(user_num)
        mask = (self.perb_degree[t] > delta)
        perb_list = all_list[mask]

        for i, usei in enumerate(perb_list):  
            for usej in perb_list[(i+1):]:
                rand_num = np.random.random()
                if self.data[t].has_edge(usei, usej):
                    if rand_num < p:
                        G[usei, usej] = G[usej, usei] = True
                else:
                    if rand_num > p:
                        G[usei, usej] = G[usej, usei] = True

        return G  

    def cal_trans(self, eps, t, delta):
        

        if t == 0:
             return np.array([[1 - 1e-4, 1e-4], [0.9, 0.1]])

        user_num = len(self.data[t].nodes())

        all_list = np.arange(user_num)
        mask = (self.perb_degree[t] > delta)
        perb_list = all_list[mask]
        report_num = len(perb_list)

        matrix1 = nx.to_numpy_array(self.data[t], dtype=bool)
        matrix2 = nx.to_numpy_array(self.data[t-1], dtype=bool)
        
        transcount = np.zeros((report_num, 2))
        for i, usei in enumerate(perb_list):
            for usej in perb_list:
                if (matrix1[usei, usej] == False) and (matrix2[usei, usej] == False):
                    transcount[i, 0] += 1
                elif (matrix1[usei, usej] == True) and (matrix2[usei, usej] == True):
                    transcount[i, 1] += 1               

        scale = 1 / eps
        noise0 = np.random.laplace(scale=scale, size=report_num)
        transcount[:, 0] = transcount[:, 0] + noise0

        noise1 = np.random.laplace(scale=scale, size=report_num)
        transcount[:, 1] = transcount[:, 1] + noise1

        d_tol = np.sum(self.perb_degree[t][mask])

        num0 = np.sum(transcount[:, 0])
        num1 = np.sum(transcount[:, 1])

        num0 = np.clip(num0, 0.1 * (report_num**2 - d_tol), (1 - 1e-8) * (report_num**2 - d_tol))
        num1 = np.clip(num1, 0.1 * (d_tol), 0.9 * d_tol)
        
        a_0_0 = num0 / (report_num**2 - d_tol)
        a_1_1 = num1 / d_tol
        
        return np.array([[a_0_0, 1 - a_0_0], [1 - a_1_1, a_1_1]])
