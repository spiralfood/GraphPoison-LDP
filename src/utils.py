import networkx as nx 
import numpy as np
import math
import re
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import minimize_scalar
import community
from sklearn.metrics.cluster import normalized_mutual_info_score


def read_action(datapath,group_num,time_val=None):
    print("...Preparing to read data...")

    
    f = open(datapath,encoding="UTF-8")
    time_list = []
    for line in f:
        s = line.strip().split('\t')
        temp = s[0].split()
        temp = [int(t) for t in temp]
        time_list.append(temp)
    f.close()
    time_list = sorted(time_list,key=lambda x: x[0])
    
    
    time_info = [time_list[0][0],time_list[0][0]]
    user_list = {}

    num = 0
    for i in time_list:
        time_info[0] = min(time_info[0],i[0])
        time_info[1] = max(time_info[1],i[0])
        if i[1] not in user_list:
            user_list[i[1]] = num
            num = num+1
        if i[2] not in user_list:
            user_list[i[2]] = num
            num = num+1

    time_val = (time_info[1] - time_info[0]) / group_num
    time_info.append(time_val)

    print("Start time:",time_info[0])
    print("End time:",time_info[1])
    print("Number of",len(user_list),"users") 
    print("Using a time interval of",time_val,"we sampled",(time_info[1]-time_info[0])/time_info[2],"groups")
       
    
    G = []
    real_index = 0
    start_time = time_info[0]
    end_time = time_info[0]
    for i in range(group_num):
        
        slice_graph  = nx.Graph()
        H = nx.path_graph(len(user_list))
        slice_graph.add_nodes_from(H)

        
        start_time = end_time 
        end_time = end_time + time_info[2]

        while start_time <= time_list[real_index][0] < end_time:
            slice_graph.add_edge(user_list[time_list[real_index][1]],user_list[time_list[real_index][2]])
            real_index = real_index + 1
            if real_index >= (len(time_list)-1):
                break
        G.append(slice_graph)
        
    return G

def read_email(datapath,group_num = 20,time_val=None):
    print("...Preparing to read data...")

    
    f = open(datapath,encoding="UTF-8")
    time_list = []
    for line in f:
        s = line.strip().split('\t')
        temp = s[0].split()
        temp[0],temp[2] = temp[2],temp[0]
        temp = [int(t) for t in temp]
        time_list.append(temp)
    f.close()
    time_list = sorted(time_list,key=lambda x: x[0])
    
    
    time_info = [time_list[0][0],time_list[0][0]]

    user_list = {}

    num = 0
    for i in time_list:
        time_info[0] = min(time_info[0],i[0])
        time_info[1] = max(time_info[1],i[0])
        if i[1] not in user_list:
            user_list[i[1]] = num
            num = num+1
        if i[2] not in user_list:
            user_list[i[2]] = num
            num = num+1

    time_val = (time_info[1] - time_info[0]) / group_num
    time_info.append(time_val)

    print("Start time:",time_info[0])
    print("End time:",time_info[1])
    print("Number of",len(user_list),"users") 
    print("Using a time interval of",time_val,"we sampled",(time_info[1]-time_info[0])/time_info[2],"groups")
       
    
    G = []
    real_index = 0
    start_time = time_info[0]
    end_time = time_info[0]
    for i in range(group_num):
        
        slice_graph  = nx.Graph()
        H = nx.path_graph(len(user_list))
        slice_graph.add_nodes_from(H)

        
        start_time = end_time 
        end_time = end_time + time_info[2]

        while start_time <= time_list[real_index][0] < end_time:
            slice_graph.add_edge(user_list[time_list[real_index][1]],user_list[time_list[real_index][2]])
            real_index = real_index + 1
            if real_index >= (len(time_list)-1):
                break
        G.append(slice_graph)
    G = G[:12]
    
    return G


def read_contact(datapath,group_num,time_val=None):
    print("...Preparing to read data...")

    
    f = open(datapath,encoding="UTF-8")
    time_list = []
    for line in f:
        temp = line.strip().split(',')
        temp[0],temp[2] = temp[2],temp[0]
        temp = [int(t) for t in temp]
        time_list.append(temp)
        
    f.close()

    time_list = sorted(time_list,key=lambda x: x[0])
    
    
    time_info = [time_list[0][0],time_list[0][0]]

    user_list = {}

    num = 0
    for i in time_list:
        time_info[0] = min(time_info[0],i[0])
        time_info[1] = max(time_info[1],i[0])
        if i[1] not in user_list:
            user_list[i[1]] = num
            num = num+1
        if i[2] not in user_list:
            user_list[i[2]] = num
            num = num+1
    time_val = (time_info[1] - time_info[0]) / group_num
    time_info.append(time_val)

    print("Start time:",time_info[0])
    print("End time:",time_info[1])
    print("Number of",len(user_list),"users") 
    print("Using a time interval of",time_val,"we sampled",(time_info[1]-time_info[0])/time_info[2],"groups")
       
    
    G = []
    real_index = 0
    start_time = time_info[0]
    end_time = time_info[0]
    for i in range(group_num):
        
        slice_graph  = nx.Graph()
        H = nx.path_graph(len(user_list))
        slice_graph.add_nodes_from(H)

        
        start_time = end_time 
        end_time = end_time + time_info[2]

        while start_time <= time_list[real_index][0] < end_time:
            slice_graph.add_edge(user_list[time_list[real_index][1]],user_list[time_list[real_index][2]])
            real_index = real_index + 1
            if real_index >= (len(time_list)-1):
                break
        G.append(slice_graph)

    return G

def read_reply(datapath,group_num = 20,time_val=None):
    print("...Preparing to read data...")

    
    f = open(datapath,encoding="UTF-8")
    time_list = []
    for line in f:
        temp = re.split(r'\s+', line.strip())
        del temp[2]
        temp[0],temp[2] = temp[2],temp[0]
        temp = [int(t) for t in temp]
        time_list.append(temp)
    f.close()
    time_list = sorted(time_list,key=lambda x: x[0])
    
    
    time_info = [time_list[0][0],time_list[0][0]]

    user_list = {}

    num = 0
    for i in time_list:
        time_info[0] = min(time_info[0],i[0])
        time_info[1] = max(time_info[1],i[0])
        if i[1] not in user_list:
            user_list[i[1]] = num
            num = num+1
        if i[2] not in user_list:
            user_list[i[2]] = num
            num = num+1
    time_val = (time_info[1] - time_info[0]) / group_num
    time_info.append(time_val)

    print("Start time:",time_info[0])
    print("End time:",time_info[1])
    print("Number of",len(user_list),"users") 
    print("Using a time interval of",time_val,"we sampled",(time_info[1]-time_info[0])/time_info[2],"groups")
       
    
    G = []
    real_index = 0
    start_time = time_info[0]
    end_time = time_info[0]
    for i in range(group_num):
        
        slice_graph  = nx.Graph()
        H = nx.path_graph(len(user_list))
        slice_graph.add_nodes_from(H)

        
        start_time = end_time 
        end_time = end_time + time_info[2]

        while start_time <= time_list[real_index][0] < end_time:
            slice_graph.add_edge(user_list[time_list[real_index][1]],user_list[time_list[real_index][2]])
            real_index = real_index + 1
            if real_index >= (len(time_list)-1):
                break
        G.append(slice_graph)

    return G

def read_SMS(datapath,group_num,time_val=None):

    print("...Preparing to read data...")

    
    f = open(datapath,encoding="UTF-8")
    time_list = []
    for line in f:
        temp = re.split(r'\s+', line.strip())
        temp[0],temp[2] = temp[2],temp[0]
        temp = [int(t) for t in temp]
        time_list.append(temp)
        
    f.close()

    time_list = sorted(time_list,key=lambda x: x[0])
    
    
    time_info = [time_list[0][0],time_list[0][0]]

    user_list = {}

    num = 0
    for i in time_list:
        time_info[0] = min(time_info[0],i[0])
        time_info[1] = max(time_info[1],i[0])
        if i[1] not in user_list:
            user_list[i[1]] = num
            num = num+1
        if i[2] not in user_list:
            user_list[i[2]] = num
            num = num+1
    time_val = (time_info[1] - time_info[0]) / group_num
    time_info.append(time_val)

    print("Start time:",time_info[0])
    print("End time:",time_info[1])
    print("Number of",len(user_list),"users") 
    print("Using a time interval of",time_val,"we sampled",(time_info[1]-time_info[0])/time_info[2],"groups")
       
    
    G = []
    real_index = 0
    start_time = time_info[0]
    end_time = time_info[0]
    for i in range(group_num):
        
        slice_graph  = nx.Graph()
        H = nx.path_graph(len(user_list))
        slice_graph.add_nodes_from(H)

        
        start_time = end_time 
        end_time = end_time + time_info[2]

        while start_time <= time_list[real_index][0] < end_time:
            slice_graph.add_edge(user_list[time_list[real_index][1]],user_list[time_list[real_index][2]])
            real_index = real_index + 1
            if real_index >= (len(time_list)-1):
                break
        G.append(slice_graph)

    return G

def read_facebook(datapath,group_num,time_val=None):

    print("...Preparing to read data...")

    
    f = open(datapath,encoding="UTF-8")
    time_list = []
    for line in f:
        temp = re.split(r'\s+', line.strip())
        del temp[2]
        temp[0],temp[2] = temp[2],temp[0]
        temp = [int(t) for t in temp]
        time_list.append(temp)
        
    f.close()

    time_list = sorted(time_list,key=lambda x: x[0])
    
    
    time_info = [time_list[0][0],time_list[0][0]]

    user_list = {}

    num = 0
    for i in time_list:
        time_info[0] = min(time_info[0],i[0])
        time_info[1] = max(time_info[1],i[0])
        if i[1] not in user_list:
            user_list[i[1]] = num
            num = num+1
        if i[2] not in user_list:
            user_list[i[2]] = num
            num = num+1
    time_val = (time_info[1] - time_info[0]) / group_num
    time_info.append(time_val)

    print("Start time:",time_info[0])
    print("End time:",time_info[1])
    print("Number of",len(user_list),"users") 
    print("Using a time interval of",time_val,"we sampled",(time_info[1]-time_info[0])/time_info[2],"groups")
       
    
    G = []
    real_index = 0
    start_time = time_info[0]
    end_time = time_info[0]
    for i in range(group_num):
        
        slice_graph  = nx.Graph()
        H = nx.path_graph(len(user_list))
        slice_graph.add_nodes_from(H)

        
        start_time = end_time 
        end_time = end_time + time_info[2]

        while start_time <= time_list[real_index][0] < end_time:
            slice_graph.add_edge(user_list[time_list[real_index][1]],user_list[time_list[real_index][2]])
            real_index = real_index + 1
            if real_index >= (len(time_list)-1):
                break
        G.append(slice_graph)

    return G

def read_yahoo(datapath,group_num,time_val=None):

    print("...Preparing to read data...")

    
    f = open(datapath,encoding="UTF-8")
    time_list = []
    for line in f:
        temp = re.split(r'\s+', line.strip())
        del temp[2]
        temp[0],temp[2] = temp[2],temp[0]
        temp = [int(t) for t in temp]
        time_list.append(temp)
        
    f.close()

    time_list = sorted(time_list,key=lambda x: x[0])
    
    
    time_info = [time_list[0][0],time_list[0][0]]

    user_list = {}

    num = 0
    for i in time_list:
        time_info[0] = min(time_info[0],i[0])
        time_info[1] = max(time_info[1],i[0])
        if i[1] not in user_list:
            user_list[i[1]] = num
            num = num+1
        if i[2] not in user_list:
            user_list[i[2]] = num
            num = num+1
    time_val = (time_info[1] - time_info[0]) / group_num
    time_info.append(time_val)

    print("Start time:",time_info[0])
    print("End time:",time_info[1])
    print("Number of",len(user_list),"users") 
    print("Using a time interval of",time_val,"we sampled",(time_info[1]-time_info[0])/time_info[2],"groups")
       
    
    G = []
    real_index = 0
    start_time = time_info[0]
    end_time = time_info[0]
    for i in range(group_num):
        
        slice_graph  = nx.Graph()
        H = nx.path_graph(len(user_list))
        slice_graph.add_nodes_from(H)

        
        start_time = end_time 
        end_time = end_time + time_info[2]

        while start_time <= time_list[real_index][0] < end_time:
            slice_graph.add_edge(user_list[time_list[real_index][1]],user_list[time_list[real_index][2]])
            real_index = real_index + 1
            if real_index >= (len(time_list)-1):
                break
        G.append(slice_graph)

    return G


def read_as733(datapath):
    print("...Preparing to read the AS-733 dataset...")
    
    
    import os
    files = [f for f in os.listdir(datapath) if f.endswith('.txt')]
    files.sort()  
    
    
    all_nodes = set()
    
    for file in files:
        with open(os.path.join(datapath, file), 'r') as f:
            for line in f:
                if line.startswith('#'):  
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        all_nodes.add(u)
                        all_nodes.add(v)
                    except ValueError:
                        continue  
    
    
    sorted_nodes = sorted(all_nodes)
    user_list = {node: i for i, node in enumerate(sorted_nodes)}
    
    print(f"There are {len(user_list)} users in this dataset")
    
    G = []
    
    
    for file in files:
        slice_graph = nx.Graph()
        
        slice_graph.add_nodes_from(range(len(user_list)))
        
        with open(os.path.join(datapath, file), 'r') as f:
            for line in f:
                if line.startswith('#'):  
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        
                        mapped_u = user_list[u]
                        mapped_v = user_list[v]
                        slice_graph.add_edge(mapped_u, mapped_v)
                    except ValueError:
                        continue  
        G.append(slice_graph)
    
    print(f"Successfully read graph data for {len(G)} time steps")
    return G, user_list


def norm_laplace(vec):
    non_smooth_func = lambda x:np.abs(np.sum([max(i+x,0) for i in vec]) - np.sum(vec))
    res = minimize_scalar(non_smooth_func, method='brent')
    min_val = np.round(res.fun)
    min_val = min(2,min_val)
    return np.round([max(i + min_val , 0) for i in vec])

def remove_common_isolated_nodes(graph1, graph2):
    
    isolated_nodes_g1 = [node for node in graph1.nodes() if graph1.degree(node) == 0]
    isolated_nodes_g2 = [node for node in graph2.nodes() if graph2.degree(node) == 0]

    
    common_isolated_nodes = set(isolated_nodes_g1).intersection(isolated_nodes_g2)

    
    graph1.remove_nodes_from(common_isolated_nodes)
    graph2.remove_nodes_from(common_isolated_nodes)

    return graph1, graph2


def kl_divergence(degree1,degree2):
    
    width = math.ceil(max(np.max(degree1),np.max(degree2)))
    
    p , _ = np.histogram(degree1, bins=range(width+2),density=True)    
    
    q , _ = np.histogram(degree2, bins=range(width+2),density=True)
    
    p = p + 1e-10
    q = q + 1e-10
    return entropy(p,q)

def compare_deg(graph1,graph2):
    

    degree1 = [degree for node, degree in graph1.degree()]
    degree2 = [degree for node, degree in graph2.degree()]
    return kl_divergence(degree1,degree2)   


def compare_communities(graph_nx,synethic_graph_nx):

    
    partition1 = community.best_partition(graph_nx)

    
    partition2 = community.best_partition(synethic_graph_nx)

    
    nodes = graph_nx.nodes()
    labels1 = []
    labels2 = []
    for node in nodes:
        labels1.append(partition1[node])
        labels2.append(partition2[node])
    
    
    nmi = normalized_mutual_info_score(labels1, labels2)

    return nmi

    
def compare_vec(G1, G2, num_nodes):
    
    
    try:
        evc_scores1 = nx.eigenvector_centrality(G1, max_iter=100)
    except nx.PowerIterationFailedConvergence:
        print("Eigenvector centrality did not converge within 100 iterations.")
        
        evc_scores1 = dict(G1.degree())
    sorted_nodes1 = sorted(evc_scores1.items(), key=lambda x: x[1], reverse=True)
    
    
    top1_percent_nodes1 = sorted_nodes1[:int(num_nodes * 0.01)]
    top1_percent_node_ids1 = [n[0] for n in top1_percent_nodes1]

    
    try:
        evc_scores2 = nx.eigenvector_centrality(G2, max_iter=100)
    except nx.PowerIterationFailedConvergence:
        print("Eigenvector centrality did not converge within 100 iterations.")
        
        evc_scores2 = dict(G2.degree())
    sorted_nodes1 = sorted(evc_scores1.items(), key=lambda x: x[1], reverse=True)

    
    sorted_nodes2 = sorted(evc_scores2.items(), key=lambda x: x[1], reverse=True)

    
    top1_percent_nodes2 = sorted_nodes2[:int(num_nodes * 0.01)]
    top1_percent_node_ids2 = [n[0] for n in top1_percent_nodes2]

    
    common_nodes_percent = len(set(top1_percent_node_ids1).intersection(set(top1_percent_node_ids2))) / len(top1_percent_node_ids1)

    return common_nodes_percent
                     

def calculate_rediam(G , G_synthetic):

    
    components = list(nx.connected_components(G))
    components_hat = list(nx.connected_components(G_synthetic))

    D_max = 0
    for component in components:
        D = nx.diameter(G.subgraph(component))
        D_max = max(D_max,D)

    s_D_max = 0 
    for component_hat in components_hat:
        D = nx.diameter(G_synthetic.subgraph(component_hat))
        s_D_max = max(s_D_max,D)

    return np.abs(D_max - s_D_max) / max(1,D_max)


def compare_coeff(graph1,graph2):

    
    clustering_coefficient1 = nx.average_clustering(graph1)
    clustering_coefficient2 = nx.average_clustering(graph2)

    return np.abs(clustering_coefficient1 - clustering_coefficient2) / max(1e-5,clustering_coefficient1)

def compare_coeff1(graph1,graph2):

    
    triangles = sum(nx.triangles(graph1).values()) / 3 
    star_count = sum([d * (d-1)  for n, d in graph1.degree()]) / 2 
    global_clustering_coefficient1 = triangles / max(star_count,1) 
    
    
    triangles = sum(nx.triangles(graph2).values() ) /3 
    star_count = sum([d * (d-1)  for n, d in graph2.degree()]) /2 
    global_clustering_coefficient2 = triangles / max(star_count,1) 

    return np.abs(global_clustering_coefficient1 - global_clustering_coefficient2) / max(1e-5,global_clustering_coefficient1)
