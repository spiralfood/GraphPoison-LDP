import utils
import users
import model
import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
import copy
import os
import time
import random


SEED = 2025
random.seed(SEED)
np.random.seed(SEED)


dataset_label = "I"    

data_switcher = {
    "A": "email/email.txt",
    "B": "SMS/SMS.txt",
    "C": "contacts/contacts.txt",
    "D": "LyonSchool/LyonSchool.txt",
    "E": "collegeMsg/CollegeMsg.txt",
    "F": "facebook/facebook.txt",
    "G": "yahoo/yahoo.txt",
    "H": "reply/reply.txt",
    "I": "as-733"  
}


func_switcher = {
    "A": utils.read_email,
    "B": utils.read_SMS,
    "C": utils.read_contact,
    "D": utils.read_email,
    "E": utils.read_email,
    "F": utils.read_facebook,
    "G": utils.read_yahoo,
    "H": utils.read_reply,
    "I": utils.read_as733  
}
filename = data_switcher[dataset_label]
select_func = func_switcher[dataset_label] 
datapath = "./dataset/" 

if dataset_label == "I":
    data, node_mapping = select_func(datapath + filename)
else:
    data = select_func(datapath + filename)
    node_mapping = None


def Collect_info(data, eps, eps1_part, eps2_part, eps3_part, seed=None):
    time_stamp = len(data)
    user_num = len(data[0].nodes())
    
    
    User = users.User(data=data)
    
    sensitive = 1
    eps_1 = eps * eps1_part
    print("eps_1:", eps_1, end=",")
    
    user_degree = []
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for t in range(time_stamp):
        user_degree_slice = np.zeros(user_num)
        for i in range(user_num):
            user_degree_slice[i] = data[t].degree(i) 
            
        
        scale = sensitive / eps_1

        
        noise = np.random.laplace(scale=scale, size=user_num)
        user_degree_slice = user_degree_slice + noise
        
        
        user_degree_slice = utils.norm_laplace(user_degree_slice)
        
        
        user_degree.append(user_degree_slice)
        
    User.perb_degree = np.array(user_degree)
    
    eps_2 = eps * eps2_part
    print("eps_2:", eps_2, end=",")
    
    eps_3 = eps * eps3_part
    print("eps_3:", eps_3)
    return User


import time

def sort_matrix(coo):
    sorted_indices = np.argsort(-coo.data)
    coo.data = coo.data[sorted_indices]
    coo.row = coo.row[sorted_indices]
    coo.col = coo.col[sorted_indices]
    return coo

def Graph_Reconstrut(User, prob_graph, t):
    
    sorted_coo = sort_matrix(prob_graph)
    user_num = len(User.data[t].nodes())
    perb_degree = np.array(User.perb_degree[t])

    
    graph_nx = nx.Graph()
    
    for i in range(user_num):
        graph_nx.add_node(i)
    
    
    for i in range(sorted_coo.nnz):
        
        row = int(sorted_coo.row[i])
        col = int(sorted_coo.col[i])
        if graph_nx.degree(row) < perb_degree[row] and graph_nx.degree(col) < perb_degree[col]:
            graph_nx.add_edge(row,col)
           
    return graph_nx

def create_binary_map(hmm, w):
    prob_map = {}  

    for i in range(2**w):
        binary_string = bin(i)[2:].zfill(w)  
        bool_list = [True if c == '1' else False for c in binary_string]
        bool_array = np.array(bool_list, dtype=np.bool_)
        prob = hmm.cal_prob(np.array(bool_list, dtype=np.int_))
        prob_map[tuple(bool_array)] = prob  
    
    return prob_map

time_list = []
def Extract_info(User, eps, eps2_part, eps3_part, w, delta, seed=None):
    
    print(f"Start Extract_info, time steps: {len(User.data)}, input window size w = {w}")
    time_stamp = len(User.data)
    user_num = len(User.data[0].nodes())

    
    in_window_data = []

    
    warmup = max(0, w - 1)
    print(f"Warm-up placeholder frames: {warmup} (algorithm uses w-1 history frames), window param w = {w}")
    for _ in range(warmup):
        graph = np.zeros((user_num, user_num), dtype=np.bool_)
        in_window_data.append(graph)
    print("Warm-up done")

    
    print("Init HMM model")
    hmm = model.HMM()
    print("HMM model ready")

    
    print(f"Process data by time step, count: {time_stamp}")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for t in range(time_stamp):
        print(f"Processing time step {t}")
        
        graph = User.Random_Response(eps * eps3_part, t, delta)
        in_window_data.append(graph)

        
        transprob = User.cal_trans(eps * eps2_part, t, delta)

        
        eps3 = eps * eps3_part
        p = np.exp(eps3) / (1 + np.exp(eps3))
        emissionprob = np.array([[p, 1 - p], [1 - p, p]])

        report_num = np.sum(User.perb_degree[t] > delta)

        now_graph = np.zeros((user_num, user_num), dtype=np.float32)

        time1 = time.time()

        
        for i in range(user_num):
            di = User.perb_degree[t][i]
            if di <= 0:
                continue

            p1 = di / report_num
            startprob = np.array([1 - p1, p1])

            
            hmm.pi = np.array(startprob)
            hmm.A = np.array(transprob)
            hmm.B = np.array(emissionprob)

            
            prob_map = create_binary_map(hmm, w)

            
            for j in range(user_num):
                dj = User.perb_degree[t][j]
                if dj <= 0:
                    continue

                X = []
                for k in range(w):
                    X.append(in_window_data[k][i, j])
                prob = prob_map[tuple(X)]
                now_graph[i, j] = prob

        time2 = time.time()
        time_list.append(time2 - time1)

        
        in_window_data = in_window_data[1:]
        now_graph = coo_matrix(now_graph)
        synethic_graph_nx = Graph_Reconstrut(User, now_graph, t=t)

        
        if not hasattr(User, 'final_hmg_graphs'):
            User.final_hmg_graphs = {}
        User.final_hmg_graphs[t] = synethic_graph_nx

        print(f"Time step {t} done, nodes: {synethic_graph_nx.number_of_nodes()}, edges: {synethic_graph_nx.number_of_edges()}")
        print(f"Step runtime: {time2 - time1}")
        print(f"Progress: {t + 1}/{time_stamp}")

    print("Extract_info finished, all time steps processed")
    return

def generic_hmg_pipeline(dataset_paths, output_base_path, eps, eps1_part, eps2_part, eps3_part, w, delta):
    

    for dataset_path in dataset_paths:
        
        dataset_name = os.path.basename(dataset_path)
        print(f"Processing dataset: {dataset_name}")
        
        
        data, node_mapping = utils.read_as733(dataset_path)
        
        
        output_dir = os.path.join(output_base_path, f"{dataset_name}_HMG")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Creating output directory: {output_dir}")
        
        
        print(f"Starting the HMG perturbation pipeline, privacy budget: {eps}")
        User = Collect_info(data, eps, eps1_part, eps2_part, eps3_part)
        Extract_info(User, eps, eps2_part, eps3_part, w, delta)
        
        
        if hasattr(User, 'final_hmg_graphs'):
            print(f"Saving HMG-perturbed results, number of time steps: {len(User.final_hmg_graphs)}")
            print(f"Output directory: {output_dir}")
            
            for t, graph in User.final_hmg_graphs.items():
                
                filename = f"{dataset_name}_final_hmg_t{t}.txt"
                filepath = os.path.join(output_dir, filename)
                print(f"Saving file to: {filepath}")
                
                
                nodes_with_edges = set()
                for edge in graph.edges():
                    nodes_with_edges.add(edge[0])
                    nodes_with_edges.add(edge[1])
                nodes = len(nodes_with_edges)  
                edges = graph.number_of_edges()  
                
                
                with open(filepath, 'w') as f:
                    
                    f.write(f"# Undirected graph (each pair of nodes is saved twice):{filename}\n")
                    f.write(f"# {dataset_name} dataset - HMG reconstructed\n")
                    f.write(f"# Nodes:{nodes}\tEdges: {edges}\n")
                    f.write("# FromNodeId\tToNodeId\n")
                    
                    
                    for edge in graph.edges():
                        
                        if node_mapping is not None:
                            
                            reverse_mapping = {v: k for k, v in node_mapping.items()}
                            
                            original_u = reverse_mapping.get(edge[0], edge[0])
                            original_v = reverse_mapping.get(edge[1], edge[1])
                            f.write(f"{original_u}\t{original_v}\n")
                        else:
                            f.write(f"{edge[0]}\t{edge[1]}\n")
            
            print(f"Saved final HMG-perturbed results for all time steps to {output_dir}")
        else:
            print("No HMG-perturbed results found")
        
        print(f"Dataset {dataset_name} processed successfully")
    
    print("All datasets processed")
    return
