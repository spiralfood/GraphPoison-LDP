import os
import sys
from typing import List, Dict
import networkx as nx
import numpy as np
import random
import math


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import main
import utils


BASE_SEED = 2025


DEFAULT_BETA = 0.0
DEFAULT_EPS = 4.0
DEFAULT_EPS1_PART = 0.3
DEFAULT_EPS2_PART = 0.1
DEFAULT_EPS3_PART = 0.6
DEFAULT_W = 5
DEFAULT_DELTA = 1

def compose_hmg_seed(attack_method: str) -> int:
    

    _ = attack_method  
    return int(BASE_SEED)

def seed_random_generators(attack_method: str) -> int:
    

    seed = compose_hmg_seed(attack_method)
    random.seed(seed)
    np.random.seed(seed)
    return seed

def read_attack_graph_file(file_path: str, beta: float = 0.0) -> nx.Graph:
    

    graph = nx.Graph()
    nodes_count_hint = None

    with open(file_path, 'r') as f:
        for line in f:
            
            if line.startswith('#'):
                
                if "Nodes:" in line:
                    parts = line.replace('\t', ' ').split()
                    for part in parts:
                        if part.startswith("Nodes:"):
                            try:
                                nodes_count_hint = int(part.split("Nodes:")[1])
                            except ValueError:
                                pass
                continue

            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    graph.add_edge(u, v)
                except ValueError:
                    continue  

    
    max_edge_id = max(graph.nodes()) if graph.nodes else -1
    node_count = nodes_count_hint if nodes_count_hint is not None else (max_edge_id + 1)
    if node_count > 0:
        graph.add_nodes_from(range(node_count))

    return graph

def read_attack_graph_directory(dir_path: str, beta: float = 0.0) -> List[nx.Graph]:
    

    files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
    
    
    def extract_timestep(filename):
        try:
            
            parts = filename.split('_')
            for part in parts:
                if part.startswith('TimeStep'):
                    return int(part.replace('TimeStep', '').replace('.txt', ''))
            return 0
        except:
            return 0
    
    files.sort(key=extract_timestep)
    
    graphs = []
    for file in files:
        file_path = os.path.join(dir_path, file)
        graph = read_attack_graph_file(file_path, beta=beta)
        graphs.append(graph)
    
    return graphs

def create_node_mapping_for_graphs(graphs: List[nx.Graph]) -> Dict[int, int]:
    

    all_nodes = set()
    for graph in graphs:
        all_nodes.update(graph.nodes())
    
    
    sorted_nodes = sorted(all_nodes)
    node_mapping = {node: i for i, node in enumerate(sorted_nodes)}
    
    return node_mapping

def apply_node_mapping_to_graphs(graphs: List[nx.Graph], node_mapping: Dict[int, int]) -> List[nx.Graph]:
    

    mapped_graphs = []
    
    for graph in graphs:
        mapped_graph = nx.Graph()
        
        
        max_mapped_id = max(node_mapping.values()) if node_mapping else 0
        mapped_graph.add_nodes_from(range(max_mapped_id + 1))
        
        
        for u, v in graph.edges():
            if u in node_mapping and v in node_mapping:
                mapped_u = node_mapping[u]
                mapped_v = node_mapping[v]
                mapped_graph.add_edge(mapped_u, mapped_v)
        
        
        mapped_graph.add_nodes_from(range(len(node_mapping)))
        
        mapped_graphs.append(mapped_graph)
    
    return mapped_graphs

def process_attack_graphs_with_hmg(
    input_dir: str,
    output_dir: str,
    attack_type: str,
    attack_method: str,
    beta: float = DEFAULT_BETA,
    eps: float = DEFAULT_EPS,
    eps1_part: float = DEFAULT_EPS1_PART,
    eps2_part: float = DEFAULT_EPS2_PART,
    eps3_part: float = DEFAULT_EPS3_PART,
    w: int = DEFAULT_W,
    delta: int = DEFAULT_DELTA
) -> None:
    

    seed = seed_random_generators(attack_method)
    
    print(f"Processing {attack_type}/{attack_method} attack graphs...")
    
    
    graphs = read_attack_graph_directory(input_dir, beta=beta)
    if not graphs:
        print(f"Warning: no attack graph files found in {input_dir}")
        return
    
    print(f"Found attack graphs for {len(graphs)} time steps")
    
    
    node_mapping = create_node_mapping_for_graphs(graphs)
    print(f"Created node mapping, total nodes: {len(node_mapping)}")
    
    
    mapped_graphs = apply_node_mapping_to_graphs(graphs, node_mapping)
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Creating output directory: {output_dir}")
    
    
    print(f"Starting the HMG perturbation pipeline, privacy budget: {eps}")
    User = main.Collect_info(
        mapped_graphs,
        eps,
        eps1_part,
        eps2_part,
        eps3_part
    )
    main.Extract_info(
        User,
        eps,
        eps2_part,
        eps3_part,
        w,
        delta
    )
    
    
    if hasattr(User, 'final_hmg_graphs'):
        print(f"Saving HMG-perturbed results, number of time steps: {len(User.final_hmg_graphs)}")
        
        
        reverse_mapping = {v: k for k, v in node_mapping.items()}
        
        for t, graph in User.final_hmg_graphs.items():
            
            filename = f"HMG_{attack_method}_{attack_type}_Dataset0_TimeStep{t}.txt"
            filepath = os.path.join(output_dir, filename)
            
            
            nodes_with_edges = set()
            for edge in graph.edges():
                nodes_with_edges.add(edge[0])
                nodes_with_edges.add(edge[1])
            nodes = len(nodes_with_edges)
            edges = graph.number_of_edges()
            
            
            with open(filepath, 'w') as f:
                
                f.write(f"# Undirected graph (each pair of nodes is saved twice):{filename}\n")
                f.write(f"# HMG processed {attack_method} {attack_type} attack graph\n")
                f.write(f"# Nodes:{nodes}\tEdges: {edges}\n")
                f.write("# FromNodeId\tToNodeId\n")
                
                
                for edge in graph.edges():
                    original_u = reverse_mapping.get(edge[0], edge[0])
                    original_v = reverse_mapping.get(edge[1], edge[1])
                    f.write(f"{original_u}\t{original_v}\n")
        
        print(f"Saved HMG-perturbed results for all time steps to {output_dir}")
    else:
        print("No HMG-perturbed results found")
    
    print(f"Finished processing {attack_type}/{attack_method}")

def _choose_dataset_dir(base_root: str) -> str:
    

    if not os.path.isdir(base_root):
        print(f"Input root {base_root} does not exist.")
        return None

    candidates = sorted(
        d for d in os.listdir(base_root)
        if os.path.isdir(os.path.join(base_root, d))
    )
    if not candidates:
        print(f"No dataset folders found under {base_root}.")
        return None

    print("\n=== Select dataset folder to add noise (from output_py) ===")
    for idx, name in enumerate(candidates, 1):
        print(f"{idx} - {name}")

    choice = input(f"Enter dataset id (1-{len(candidates)}, press Enter for 1): ").strip()
    try:
        sel = int(choice) if choice else 1
        if sel < 1 or sel > len(candidates):
            raise ValueError
    except ValueError:
        sel = 1
        print(f"Invalid choice, default to 1 - {candidates[0]}")

    return candidates[sel - 1]


def process_all_attack_graphs(
    input_root: str = "./output_py",
    output_root: str = "./HMG_attack",
    dataset_tag: str = None,
    beta: float = DEFAULT_BETA,
    eps: float = DEFAULT_EPS,
    eps1_part: float = DEFAULT_EPS1_PART,
    eps2_part: float = DEFAULT_EPS2_PART,
    eps3_part: float = DEFAULT_EPS3_PART,
    w: int = DEFAULT_W,
    delta: int = DEFAULT_DELTA
) -> None:
    

    if dataset_tag is None:
        dataset_tag = _choose_dataset_dir(input_root)
        if dataset_tag is None:
            return

    input_base_dir = os.path.join(input_root, dataset_tag)
    output_base_dir = os.path.join(output_root, dataset_tag)

    if not os.path.exists(input_base_dir):
        print(f"Input directory {input_base_dir} does not exist.")
        return

    os.makedirs(output_base_dir, exist_ok=True)

    attack_types = ['DC', 'LCC']
    attack_methods = ['RNA']  

    print("\n=== Start HMG processing ===")
    print(f"Dataset folder: {dataset_tag}")
    print(f"Attack types: {attack_types}")
    print(f"Attack methods: {attack_methods}")
    print(f"Epsilon: {eps}")

    for attack_type in attack_types:
        for attack_method in attack_methods:
            input_dir = os.path.join(input_base_dir, attack_type, attack_method)
            output_dir = os.path.join(output_base_dir, attack_type, f"{attack_method}_HMG")

            if os.path.exists(input_dir):
                try:
                    process_attack_graphs_with_hmg(
                        input_dir=input_dir,
                        output_dir=output_dir,
                        attack_type=attack_type,
                        attack_method=attack_method,
                        beta=beta,
                        eps=eps,
                        eps1_part=eps1_part,
                        eps2_part=eps2_part,
                        eps3_part=eps3_part,
                        w=w,
                        delta=delta
                    )
                except Exception as e:
                    print(f"Error processing {attack_type}/{attack_method}: {str(e)}")
            else:
                print(f"Skip: input directory {input_dir} does not exist.")

    print("All HMG processing completed.")
if __name__ == "__main__":
    
    print("Starting HMG attack processing...")
    process_all_attack_graphs()
    print("HMG attack processing finished.")
