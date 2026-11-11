import os
import random
import math
import numpy as np
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional, Union
from allocation_strategy import allocate_fake_nodes, get_strategy_name
import utils


DEFAULT_BETA = 0.05
DEFAULT_TARGET_RATIO = 0.05


class SimpleGraph:
    

    def __init__(self):
        self.adjacency_list = {}
        self.vertices = set()
    
    def add_vertex(self, vertex: int) -> None:
        

        self.vertices.add(vertex)
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = set()
    
    def add_edge(self, u: int, v: int) -> None:
        

        self.add_vertex(u)
        self.add_vertex(v)
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)
    
    def has_edge(self, u: int, v: int) -> bool:
        

        return u in self.adjacency_list and v in self.adjacency_list[u]
    
    def degree(self, vertex: int) -> int:
        

        return len(self.adjacency_list.get(vertex, set()))
    
    def get_vertices(self) -> Set[int]:
        

        return self.vertices.copy()
    
    def get_neighbors(self, vertex: int) -> Set[int]:
        

        return self.adjacency_list.get(vertex, set()).copy()
    
    def local_clustering_coefficient(self, vertex: int) -> float:
        

        neighbors = self.get_neighbors(vertex)
        k = len(neighbors)
        
        if k < 2:
            return 0.0  
        
        
        edge_count = 0
        for u in neighbors:
            for v in neighbors:
                if u < v and self.has_edge(u, v):  
                    edge_count += 1
        
        
        possible_edges = k * (k - 1) / 2
        
        return edge_count / possible_edges
    
    def average_clustering_coefficient(self) -> float:
        

        if not self.vertices:
            return 0.0
        
        total_coefficient = 0.0
        for vertex in self.vertices:
            total_coefficient += self.local_clustering_coefficient(vertex)
        
        return total_coefficient / len(self.vertices)


class DynamicAttackLCC:
    

    SEED = 2025
    
    def __init__(self):
        
        self.time_series_graphs = []  
        self.time_steps = 0           
        self.current_time_step = 0     
        
        
        self.dataset = 0
        self.dataset_tag = "as-733"
        self.metric = 2  
        
        self.epsilon = 1.0
        self.percentage_for_matrix = 0.9
        self.percentage = 0.9
        
        
        self.node_mappings = []
        self.reverse_node_mappings = []
        
        self._seeded_methods = set()

    def _compose_seed(self, attack_method: str, time_step: int) -> int:
        
        _ = (attack_method, time_step)  
        return int(self.SEED)

    def _seed_rng_for_step(self, attack_method: str, time_step: int) -> int:
        
        seed = self._compose_seed(attack_method, time_step)
        if attack_method not in self._seeded_methods:
            random.seed(seed)
            np.random.seed(seed)
            self._seeded_methods.add(attack_method)
        return seed
    
    def load_time_series_graphs(self, dataset_path: str, max_time_steps: int = 0) -> Dict:
        

        print("=== Load Dynamic Dataset ===")
        print(f"Dataset path: {dataset_path}")
        
        result = {
            'time_series_graphs': [],
            'dataset_infos': [],
            'node_mappings': [],  
            'reverse_node_mappings': [],  
            'max_node_ids': []  
        }
        
        try:
            
            if not os.path.exists(dataset_path):
                print(f"Error：Dataset directory does not exist: {dataset_path}")
                
                print("Creating sample data for testing...")
                return self._create_sample_data(result)
            
            if not os.path.isdir(dataset_path):
                print(f"Error：{dataset_path} is not a directory")
                return self._create_sample_data(result)
            
            
            files = [f for f in os.listdir(dataset_path) if f.endswith('.txt')]
            if not files:
                print("Error：No .txt files found in the dataset directory")
                print("Creating sample data for testing...")
                return self._create_sample_data(result)
            
            
            files.sort()
            
            
            files_to_load = len(files) if max_time_steps == 0 else min(max_time_steps, len(files))
            print(f"Found {len(files)} files; loading the first {files_to_load} files")
            
            
            for i in range(files_to_load):
                file_path = os.path.join(dataset_path, files[i])
                print(f"Loading file {i+1}/{files_to_load}: {files[i]}")
                
                info = self._parse_dataset_info(file_path)
                
                
                current_node_ids = set()
                self._collect_node_ids(file_path, current_node_ids)
                
                
                current_mapping = {}
                current_reverse_mapping = {}
                current_max_node_id = self._create_node_mapping_for_time_step(current_node_ids, current_mapping, current_reverse_mapping)
                
                
                graph = self._load_single_graph(file_path, current_mapping, current_max_node_id + 1)
                
                if graph is not None:
                    result['time_series_graphs'].append(graph)
                    result['dataset_infos'].append(info)
                    result['node_mappings'].append(current_mapping)
                    result['reverse_node_mappings'].append(current_reverse_mapping)
                    result['max_node_ids'].append(current_max_node_id)
                    
                    print(f"Time step {i}: node count = {len(current_node_ids)}, maximum node ID = {current_max_node_id}")
            
            self.time_series_graphs = result['time_series_graphs']
            self.time_steps = len(self.time_series_graphs)
            self.node_mappings = result['node_mappings']
            self.reverse_node_mappings = result['reverse_node_mappings']
            
        except Exception as e:
            print(f"Failed to load dataset: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"Dataset loading complete; successfully loaded {len(result['time_series_graphs'])} time steps")
        return result
    
    def _collect_node_ids(self, file_path: str, node_ids: Set[int]) -> None:
        

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            node_ids.add(int(parts[0]))
                            node_ids.add(int(parts[1]))
        except Exception as e:
            print(f"Error collecting node IDs: {str(e)}")
    
    def _create_node_mapping(self, node_ids: Set[int], result: Dict) -> None:
        

        sorted_ids = sorted(node_ids)
        for i, node_id in enumerate(sorted_ids):
            result['node_mapping'][node_id] = i
            result['reverse_node_mapping'][i] = node_id
        
        result['max_node_id'] = len(sorted_ids) - 1
    
    def _create_node_mapping_for_time_step(self, node_ids: Set[int], node_mapping: Dict, reverse_node_mapping: Dict) -> int:
        

        for idx, node_id in enumerate(sorted(node_ids)):
            node_mapping[node_id] = idx
            reverse_node_mapping[idx] = node_id
        
        
        max_node_id = max(node_ids) if node_ids else 0
        
        return max_node_id
    
    def _create_sample_data(self, result: Dict) -> Dict:
        

        print("Creating sample time-series graph data...")
        
        
        time_steps = 20
        node_count = 20  
        
        
        for t in range(time_steps):
            
            current_node_ids = set(range(node_count))
            
            
            current_mapping = {}
            current_reverse_mapping = {}
            current_max_node_id = self._create_node_mapping_for_time_step(current_node_ids, current_mapping, current_reverse_mapping)
            
            
            adj_matrix = np.zeros((node_count, node_count), dtype=bool)
            
            
            edge_probability = 0.3
            for i in range(node_count):
                for j in range(i+1, node_count):  
                    if random.random() < edge_probability:
                        adj_matrix[i, j] = True
                        adj_matrix[j, i] = True  
            
            
            result['time_series_graphs'].append(adj_matrix)
            
            
            result['dataset_infos'].append({
                'filename': f'sample_graph_{t}.txt',
                'description': 'Sample graph data',
                'node_count': node_count,
                'edge_count': np.sum(adj_matrix) // 2,  
                'date': ''
            })
            
            
            result['node_mappings'].append(current_mapping)
            result['reverse_node_mappings'].append(current_reverse_mapping)
            result['max_node_ids'].append(current_max_node_id)
            
            print(f"Time step {t}: node count = {len(current_node_ids)}, maximum node ID = {current_max_node_id}")
        
        self.time_series_graphs = result['time_series_graphs']
        self.time_steps = len(self.time_series_graphs)
        self.node_mappings = result['node_mappings']
        self.reverse_node_mappings = result['reverse_node_mappings']
        
        print(f"Created {time_steps} time-step sample graphs, each with {node_count} nodes")
        return result
    
    def _parse_dataset_info(self, file_path: str) -> Dict:
        

        filename = os.path.basename(file_path)
        info = {
            'filename': filename,
            'description': '',
            'node_count': 0,
            'edge_count': 0,
            'date': ''
        }
        
        
        if 'as' in filename:
            date_part = filename.replace('as', '')
            if date_part.isdigit() and len(date_part) == 8:
                year = date_part[:4]
                month = date_part[4:6]
                day = date_part[6:8]
                info['date'] = f"{year}-{month}-{day}"
        
        
        nodes = set()
        edge_count = 0
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            nodes.add(int(parts[0]))
                            nodes.add(int(parts[1]))
                            edge_count += 1
        except Exception as e:
            print(f"Error parsing dataset information: {str(e)}")
        
        info['node_count'] = len(nodes)
        info['edge_count'] = edge_count
        
        return info
    
    def _load_single_graph(self, file_path: str, node_mapping: Dict[int, int], max_node_id: int) -> Optional[np.ndarray]:
        

        try:
            
            node_count = len(node_mapping)
            graph = np.zeros((node_count, node_count), dtype=bool)
            
            
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            u = int(parts[0])
                            v = int(parts[1])
                            
                            if u in node_mapping and v in node_mapping:
                                u_mapped = node_mapping[u]
                                v_mapped = node_mapping[v]
                                
                                graph[u_mapped][v_mapped] = True
                                graph[v_mapped][u_mapped] = True  
            
            return graph
            
        except Exception as e:
            print(f"Error loading graph data: {str(e)}")
            return None
    
    def read_average_degree_for_time_step(self, time_step: int, dataset_name: str = None, reconstructed_dir: str = "./reconstructed_graph/") -> int:
        

        dataset_names_to_try = []
        
        if dataset_name is not None:
            dataset_names_to_try.append(dataset_name)
        else:
            
            hmg_degrees_dir = "./hmg_average_degrees/"
            if os.path.exists(hmg_degrees_dir):
                for filename in os.listdir(hmg_degrees_dir):
                    if filename.endswith("_average_degrees.txt"):
                        auto_dataset_name = filename.replace("_average_degrees.txt", "")
                        dataset_names_to_try.append(auto_dataset_name)
                        print(f"    Auto-discovered dataset: {auto_dataset_name}")
        
        
        for try_dataset_name in dataset_names_to_try:
            try:
                avg_degrees_file = os.path.join("./hmg_average_degrees/", f"{try_dataset_name}_average_degrees.txt")
                print(f"    Trying to read precomputed average-degree file: {avg_degrees_file}")
                
                if os.path.exists(avg_degrees_file):
                    with open(avg_degrees_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            
                            if line.startswith('#'):
                                continue
                            
                            
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                try:
                                    file_time_step = int(parts[0])
                                    avg_degree = int(parts[1])
                                    
                                    if file_time_step == time_step:
                                        print(f"    Read average degree from precomputed file: {avg_degree} (time step: {time_step}, dataset: {try_dataset_name})")
                                        return avg_degree
                                except ValueError:
                                    continue  
                    
                    print(f"    Warning：No data found for time step {time_step} in the precomputed file")
                else:
                    print(f"    Warning：Precomputed average-degree file does not exist: {avg_degrees_file}")
            except Exception as e:
                print(f"    Warning：Error reading precomputed average-degree file: {str(e)}")
                continue
        
        
        if not dataset_names_to_try:
            print(f"    Warning：No dataset name was provided and automatic discovery failed; pass the dataset_name parameter when calling the attack function")
        
        
        if dataset_name is not None:
            try:
                hmg_file = os.path.join(reconstructed_dir, f"{dataset_name}_HMG", f"{dataset_name}_final_hmg_t{time_step}.txt")
                print(f"    Fallback：Trying to calculate average degree from HMG file: {hmg_file}")
                
                if os.path.exists(hmg_file):
                    with open(hmg_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith("# Nodes:") and "Edges:" in line:
                                
                                parts = line.strip().split()
                                nodes = int(parts[1].split(":")[1])
                                edges = int(parts[3])
                                
                                if nodes > 0:
                                    avg_degree = max(1, int(2 * edges / nodes))
                                    print(f"    Calculated average degree from HMG noisy graph data: {avg_degree} (node count: {nodes}, edge count: {edges})")
                                    return avg_degree
                                else:
                                    print(f"    Warning：HMG noisy graph has no valid nodes")
                                    break
                else:
                    print(f"    Warning：HMG noisy graph file does not exist: {hmg_file}")
            except Exception as e:
                print(f"    Warning：Error reading HMG file: {str(e)}")
        
        
        print("    Final fallback：Calculating average degree from original graph data") if dataset_name is None else print("    Final fallback：Calculating average degree from original graph data")
        
        if hasattr(self, 'time_series_graphs') and time_step < len(self.time_series_graphs):
            graph = self.time_series_graphs[time_step]
            node_count = graph.shape[0]
            edge_count = np.sum(graph) // 2  
            
            if node_count > 0:
                avg_degree = max(1, int(2 * edge_count / node_count))
                print(f"    Calculated average degree from original graph data: {avg_degree}")
                return avg_degree
        
        
        print(f"    Using default average degree value: 5")
        return 5
    
    def get_neighbors(self, graph: np.ndarray, node: int) -> List[int]:
        

        if node < 0 or node >= graph.shape[0]:
            return []
        
        return [i for i in range(graph.shape[0]) if graph[node][i]]
    
    def get_degree(self, graph: np.ndarray, node: int) -> int:
        

        if node < 0 or node >= graph.shape[0]:
            return 0
        
        return np.sum(graph[node])
    
    def generate_fake_nodes_excluding_targets(self, graph_nodes: List[int], target_nodes: List[int], num_fake_nodes: int) -> List[int]:
        

        if num_fake_nodes <= 0:
            return []

        available_nodes = [n for n in graph_nodes if n not in set(target_nodes)]
        if not available_nodes:
            return []

        pick_count = min(num_fake_nodes, len(available_nodes))
        return random.sample(available_nodes, pick_count)

    def _execute_mga_strategy(self,
                              time_step: int,
                              fake_nodes: List[int],
                              current_target_nodes: List[int],
                              current_graph: np.ndarray,
                              fake_connection_matrix: np.ndarray,
                              fake_edge_counts: Dict[int, int],
                              edge_limit: float) -> List[List[int]]:
        

        new_edges: List[List[int]] = []

        if edge_limit <= 0:
            print("    MGA stage：Average-degree limit is 0; skipping new edge generation.")
            return new_edges

        if not fake_nodes:
            print("    MGA stage：No fake nodes in the current time step; skipping MGA logic.")
            return new_edges

        total_nodes = current_graph.shape[0]

        def can_add_edge(u: int, v: int) -> bool:
            
            if u == v:
                return False
            if fake_connection_matrix[u][v] or current_graph[u][v]:
                return False
            if u in fake_edge_counts and fake_edge_counts[u] >= edge_limit:
                return False
            if v in fake_edge_counts and fake_edge_counts[v] >= edge_limit:
                return False
            return True

        def add_edge(u: int, v: int) -> bool:
            
            if not can_add_edge(u, v):
                return False
            fake_connection_matrix[u][v] = True
            fake_connection_matrix[v][u] = True
            if u in fake_edge_counts:
                fake_edge_counts[u] += 1
            if v in fake_edge_counts:
                fake_edge_counts[v] += 1
            new_edges.append([u, v])
            return True

        def has_fake_partner(node: int) -> bool:
            
            for other in fake_nodes:
                if other != node and fake_connection_matrix[node][other]:
                    return True
            return False

        def attempt_connect_target(node: int) -> bool:
            
            candidates = [
                target for target in current_target_nodes
                if can_add_edge(node, target)
            ]
            random.shuffle(candidates)
            for target in candidates:
                if add_edge(node, target):
                    print(f"    MGA stage：Fake node {node} connected to target node {target}。")
                    return True
            return False

        def attempt_connect_fake(node: int) -> bool:
            
            candidates = [
                other for other in fake_nodes
                if other != node and can_add_edge(node, other)
            ]
            random.shuffle(candidates)
            for other in candidates:
                if add_edge(node, other):
                    print(f"    MGA stage：Fake node {node} connected to another fake node {other}。")
                    return True
            return False

        def attempt_connect_other(node: int) -> bool:
            
            candidates = [
                other for other in range(total_nodes)
                if other != node and can_add_edge(node, other)
            ]
            random.shuffle(candidates)
            for other in candidates:
                if add_edge(node, other):
                    print(f"    MGA stage：Fake node {node} connected to another node {other}。")
                    return True
            return False

        def edges_needed(node: int) -> int:
            
            current = fake_edge_counts.get(node, 0)
            remaining = edge_limit - current
            if remaining <= 0:
                return 0
            epsilon = 1e-9
            needed = math.ceil(max(0.0, remaining - epsilon))
            return max(needed, 0)

        def connect_pair_to_common_target(u: int, v: int) -> bool:
            

            if not current_target_nodes:
                return False

            progressed = False
            targets = current_target_nodes[:]
            random.shuffle(targets)

            for target in targets:
                
                if fake_edge_counts.get(u, 0) >= edge_limit or fake_edge_counts.get(v, 0) >= edge_limit:
                    break

                u_has = fake_connection_matrix[u][target]
                v_has = fake_connection_matrix[v][target]
                if u_has and v_has:
                    continue

                added = False
                if (not u_has and fake_edge_counts.get(u, 0) < edge_limit and
                        add_edge(u, target)):
                    print(f"    MGA: paired node {u} connects to target {target}.")
                    added = True

                if (not v_has and fake_edge_counts.get(v, 0) < edge_limit and
                        add_edge(v, target)):
                    print(f"    MGA: paired node {v} connects to target {target}.")
                    added = True

                if added:
                    progressed = True
                    print(f"    MGA: pair {u}-{v} linked to common target {target}.")
                    
                    if fake_edge_counts.get(u, 0) >= edge_limit or fake_edge_counts.get(v, 0) >= edge_limit:
                        break

            return progressed

        print(f"    MGA stage：Starting to process fake nodes for time step {time_step}.")
        print(f"    MGA stage：Fake node list {fake_nodes}，target node list {current_target_nodes}，edge quota {edge_limit}。")

        pair_edges: Set[Tuple[int, int]] = set()

        
        shuffled_fake_nodes = fake_nodes[:]
        random.shuffle(shuffled_fake_nodes)
        for node in shuffled_fake_nodes:
            if fake_edge_counts.get(node, 0) >= edge_limit:
                continue
            if has_fake_partner(node):
                continue

            candidates = [
                other for other in fake_nodes
                if other != node and not has_fake_partner(other) and can_add_edge(node, other)
            ]
            random.shuffle(candidates)
            partner = None
            for candidate in candidates:
                if add_edge(node, candidate):
                    partner = candidate
                    break
            if partner is None:
                continue

            ordered_pair = tuple(sorted((node, partner)))
            pair_edges.add(ordered_pair)
            print(f"    MGA stage：Fake node {node} and {partner} formed an initial pair.")

        
        if pair_edges:
            pair_list = list(pair_edges)
            random.shuffle(pair_list)
            for u, v in pair_list:
                connect_pair_to_common_target(u, v)
        else:
            print("    MGA stage：No fake-node pairs were formed; skipping paired target connections.")

        
        def process_multi_need_nodes() -> bool:
            need_nodes = [node for node in fake_nodes if edges_needed(node) >= 2]
            if len(need_nodes) < 2:
                return False
            random.shuffle(need_nodes)
            progress = False
            for i in range(0, len(need_nodes) - 1, 2):
                u = need_nodes[i]
                v = need_nodes[i + 1]
                if connect_pair_to_common_target(u, v):
                    progress = True
            return progress

        while process_multi_need_nodes():
            pass

        remaining_nodes = [node for node in fake_nodes if fake_edge_counts.get(node, 0) < edge_limit]
        if remaining_nodes:
            random.shuffle(remaining_nodes)
            for node in remaining_nodes:
                while fake_edge_counts.get(node, 0) < edge_limit:
                    if attempt_connect_target(node):
                        continue
                    if attempt_connect_fake(node):
                        continue
                    if attempt_connect_other(node):
                        continue
                    break

        remaining = [
            node for node in fake_nodes
            if fake_edge_counts.get(node, 0) < edge_limit
        ]
        if remaining:
            print(f"    Notice：Some fake nodes did not fully meet the average-degree limit：{remaining}")

        return new_edges

    def _save_fake_node_info(self, fake_nodes: List[int], fake_edges: List[Tuple[int, int]],
                             output_file: str, result: Dict = None, time_step: int = 0) -> None:
        

        try:
            reverse_mapping = None
            if result and 'reverse_node_mappings' in result and time_step < len(result['reverse_node_mappings']):
                reverse_mapping = result['reverse_node_mappings'][time_step]

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                filename = os.path.basename(output_file)
                f.write(f"# Fake Node Information for LCC Attack: {filename}\n")
                f.write(f"# TimeStep: {time_step}\n")
                f.write(f"# Total Fake Nodes: {len(fake_nodes)}\n")
                f.write(f"# Total New Fake Edges: {len(fake_edges)}\n")
                f.write("# Format: Node1 Node2 (using original node IDs)\n")
                f.write("#" + "=" * 50 + "\n")

                f.write("\n# Fake Nodes List:\n")
                for node in fake_nodes:
                    original_id = reverse_mapping.get(node, node) if reverse_mapping else node
                    line = f"# Fake Node: {node}"
                    if reverse_mapping:
                        line += f" (Original ID: {original_id})"
                    f.write(line + "\n")

                f.write("\n# New Fake Edges Added by Attack:\n")
                for u, v in fake_edges:
                    u_orig = reverse_mapping.get(u, u) if reverse_mapping else u
                    v_orig = reverse_mapping.get(v, v) if reverse_mapping else v
                    f.write(f"{u_orig} {v_orig}\n")

            print(f"    Successfully saved fake node information file: {output_file}")
        except Exception as exc:
            print(f"Error saving fake node information: {exc}")
            import traceback
            traceback.print_exc()

    def _save_graph_to_file(self, graph: np.ndarray, output_file: str,
                            result: Dict = None, time_step: int = 0, fake_nodes_count: int = 0) -> None:
        

        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            edge_count = 0
            for i in range(graph.shape[0]):
                for j in range(i, graph.shape[0]):
                    if graph[i][j]:
                        edge_count += 1

            with open(output_file, 'w', encoding='utf-8') as f:
                filename = os.path.basename(output_file)
                f.write(f"# Undirected graph (each pair of nodes is saved twice):{filename}\n")
                f.write("# Attacked graph generated by LCC attack\n")
                f.write(f"# Nodes:{graph.shape[0]}    Edges: {edge_count}    FakeNodes: {fake_nodes_count}\n")
                f.write("# FromNodeId    ToNodeId\n")

                reverse_mapping = None
                if result and 'reverse_node_mappings' in result and time_step < len(result['reverse_node_mappings']):
                    reverse_mapping = result['reverse_node_mappings'][time_step]

                for i in range(graph.shape[0]):
                    for j in range(i, graph.shape[0]):
                        if graph[i][j]:
                            node_i = reverse_mapping.get(i, i) if reverse_mapping else i
                            node_j = reverse_mapping.get(j, j) if reverse_mapping else j
                            if i == j:
                                f.write(f"{node_i} {node_j}\n")
                            else:
                                f.write(f"{node_i} {node_j}\n")
                                f.write(f"{node_j} {node_i}\n")

            print(f"  Successfully saved graph to file: {output_file}")
        except Exception as exc:
            print(f"Error saving graph to file: {exc}")

    def apply_attack(self, time_series_graphs: List[np.ndarray], attack_result: Dict,
                     output_dir: str, attack_type: str = "RVA", result: Dict = None) -> List[np.ndarray]:
        

        print(f"=== Applying {attack_type} attack ===")
        print(f"Output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        if "output_py" in output_dir:
            fake_node_output_dir = output_dir.replace("output_py", "FakeNodeID")
        else:
            fake_node_output_dir = os.path.join("FakeNodeID", "LCC", attack_type)
        os.makedirs(fake_node_output_dir, exist_ok=True)

        attacked_graphs: List[np.ndarray] = []

        for time_step, original_graph in enumerate(time_series_graphs):
            fake_nodes = attack_result.get('time_step_fake_nodes', {}).get(time_step, [])
            fake_edges = attack_result.get('time_step_fake_edges', {}).get(time_step, [])

            
            max_index = original_graph.shape[0] - 1
            for node in fake_nodes:
                max_index = max(max_index, node)
            for edge in fake_edges:
                if len(edge) != 2:
                    continue
                max_index = max(max_index, edge[0], edge[1])
            new_size = max_index + 1

            new_graph = np.zeros((new_size, new_size), dtype=bool)
            new_graph[:original_graph.shape[0], :original_graph.shape[0]] = original_graph[:]
            for edge in fake_edges:
                if len(edge) != 2:
                    continue
                u, v = edge
                if u < new_size and v < new_size:
                    new_graph[u][v] = True
                    new_graph[v][u] = True

            attacked_graphs.append(new_graph)

            
            result_for_save = result
            if result and 'reverse_node_mappings' in result and time_step < len(result['reverse_node_mappings']):
                reverse_mapping = result['reverse_node_mappings'][time_step]
                extended_mapping = dict(reverse_mapping) if reverse_mapping else {}
                max_original_id = max(extended_mapping.values()) if extended_mapping else (new_size - 1)
                next_original_id = max_original_id + 1
                for node in range(original_graph.shape[0], new_size):
                    if node not in extended_mapping:
                        extended_mapping[node] = next_original_id
                        next_original_id += 1
                updated_reverse = list(result['reverse_node_mappings'])
                updated_reverse[time_step] = extended_mapping
                result_for_save = dict(result)
                result_for_save['reverse_node_mappings'] = updated_reverse

            output_file = os.path.join(
                output_dir,
                f"AttackedGraph_{attack_type}_LCC_Dataset{self.dataset}_TimeStep{time_step}.txt"
            )
            self._save_graph_to_file(new_graph, output_file, result_for_save, time_step, len(fake_nodes))

            fake_node_file = os.path.join(
                fake_node_output_dir,
                f"FakeNodeID_{attack_type}_LCC_Dataset{self.dataset}_TimeStep{time_step}.txt"
            )
            self._save_fake_node_info(fake_nodes, fake_edges, fake_node_file, result_for_save, time_step)

            print(f"  Time step {time_step}: original node count = {original_graph.shape[0]}, node count after attack = {new_graph.shape[0]}, fake nodes = {len(fake_nodes)}, fake edges = {len(fake_edges)}")
            print(f"  Graph saved to: {output_file}")
            print(f"  Fake node information saved to: {fake_node_file}")

        return attacked_graphs

    def _record_target_nodes(self, strategy: int, initial_target_original_ids: List[int],
                             time_step_target_original_ids: Dict[int, List[int]]) -> None:
        

        target_dir = os.path.join("target_nodes_record", "LCC", str(self.dataset_tag))
        os.makedirs(target_dir, exist_ok=True)
        initial_path = os.path.join(target_dir, f"{self.dataset_tag}_target_nodes_original_ids.txt")
        per_time_path = os.path.join(target_dir, f"{self.dataset_tag}_target_nodes_original_ids_by_time_step.txt")

        try:
            with open(initial_path, 'w', encoding='utf-8') as f:
                f.write("# Target node original ID records\n")
                f.write(f"# Allocation strategy: {get_strategy_name(strategy)}\n")
                f.write(f"# Target node count: {len(initial_target_original_ids)}\n")
                for node_id in initial_target_original_ids:
                    f.write(f"{node_id}\n")
        except Exception as exc:
            print(f"Error saving target node original ID file: {exc}")

        try:
            with open(per_time_path, 'w', encoding='utf-8') as f:
                f.write("# Target node original ID records for each time step\n")
                f.write(f"# Allocation strategy: {get_strategy_name(strategy)}\n")
                f.write(f"# Initial target node count: {len(initial_target_original_ids)}\n")
                for time_step in sorted(time_step_target_original_ids.keys()):
                    ids = time_step_target_original_ids.get(time_step, [])
                    ids_str = ", ".join(str(node_id) for node_id in ids)
                    f.write(f"TimeStep {time_step}: {ids_str}\n")
            print(f"Target node original ID records saved to: {initial_path} and {per_time_path}")
        except Exception as exc:
            print(f"Error saving per-time-step target node original ID files: {exc}")
    
    def local_clustering_coefficient(self, graph: np.ndarray, node: int) -> float:
        

        neighbors = self.get_neighbors(graph, node)
        k = len(neighbors)
        
        if k < 2:
            return 0.0  
        
        
        edge_count = 0
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                u = neighbors[i]
                v = neighbors[j]
                if graph[u][v]:
                    edge_count += 1
        
        
        possible_edges = k * (k - 1) / 2
        
        return edge_count / possible_edges
    
    def average_clustering_coefficient(self, graph: np.ndarray) -> float:
        

        node_count = graph.shape[0]
        if node_count == 0:
            return 0.0
        
        total_coefficient = 0.0
        valid_nodes = 0
        
        for node in range(node_count):
            
            if self.get_degree(graph, node) > 1:
                total_coefficient += self.local_clustering_coefficient(graph, node)
                valid_nodes += 1
        
        return total_coefficient / valid_nodes if valid_nodes > 0 else 0.0
    
    def time_decay_based_lcc_fake_node_allocation_with_dynamic_targets(self, 
                                                                   time_series_graphs: List[np.ndarray], 
                                                                   time_steps: int, 
                                                                   beta: float, 
                                                                   alpha: int,
                                                                   initial_target_nodes: List[int], 
                                                                   a1: float, 
                                                                   d: float,
                                                                   strategy: int = 0,
                                                                   dataset_name: str = None,
                                                                   reconstructed_dir: str = "./reconstructed_graph/",
                                                                   original_target_nodes: List[int] = None,
                                                                   predefined_fake_nodes_per_time_step: Dict[int, int] = None,
                                                                   predefined_fake_nodes: Dict[int, List[int]] = None) -> Dict:
        

        print("=== Clustering Coefficient Attack Allocation Strategy（Dynamic Target Nodes）===")
        print(f"Attack strategy: {['RVA', 'RNA', 'MGA'][alpha] if alpha < 3 else 'Unknown'}")
        print(f"Allocation strategy: {get_strategy_name(strategy)}")
        print(f"Number of time steps: {time_steps}")
        print(f"Fake user ratio β: {beta}")
        attack_method_name = ['RVA', 'RNA', 'MGA'][alpha] if alpha in (0, 1, 2) else str(alpha)
        
        if original_target_nodes is not None:
            initial_target_original_ids = list(original_target_nodes)
        else:
            initial_target_original_ids = []
            if self.reverse_node_mappings and len(self.reverse_node_mappings) > 0:
                reverse_mapping_t0 = self.reverse_node_mappings[0]
                for mapped_node in initial_target_nodes:
                    initial_target_original_ids.append(reverse_mapping_t0.get(mapped_node, mapped_node))
            else:
                initial_target_original_ids = list(initial_target_nodes)
        
        print(f"Initial target node count: {len(initial_target_original_ids)}")
        if initial_target_original_ids:
            print(f"Target node original IDs: {initial_target_original_ids}")
        else:
            print("No target node original IDs were provided")
        
        
        result = {
            'time_step_fake_node_counts': {},
            'time_step_fake_nodes': {},
            'time_step_fake_edges': {},
            'original_target_nodes': initial_target_original_ids
        }
        
        
        L = time_series_graphs[0].shape[0] if time_series_graphs and len(time_series_graphs) > 0 else 0
        print(f"Original graph node count L: {L}")
        
        
        total_fake_nodes = int(beta * L)
        
        print(f"Total fake node count l = β * L = {total_fake_nodes}")
        
        
        if time_steps <= 0:
            return result
        
        
        fake_nodes_per_time_step = allocate_fake_nodes(total_fake_nodes, time_steps, strategy)
        print(f"Fake node count list for allocation strategy {get_strategy_name(strategy)}: {fake_nodes_per_time_step}")
        
        for t in range(time_steps):
            print(f"Time step {t}: allocated fake node count = {fake_nodes_per_time_step[t]}")
        
        
        global_fake_node_id = L  
        
        
        time_step_target_nodes = {}
        time_step_target_original_ids = {}
        
        
        for time_step in range(time_steps):
            step_seed = self._seed_rng_for_step(attack_method_name, time_step)
            fake_nodes_for_current_step = fake_nodes_per_time_step[time_step]
            current_graph = time_series_graphs[time_step]
            
            mapping_for_time = None
            if self.node_mappings and time_step < len(self.node_mappings):
                mapping_for_time = self.node_mappings[time_step]
            
            current_target_nodes = []
            current_target_original_ids = []
            if mapping_for_time:
                for original_id in initial_target_original_ids:
                    mapped_id = mapping_for_time.get(original_id)
                    if mapped_id is not None and mapped_id < current_graph.shape[0]:
                        current_target_nodes.append(mapped_id)
                        current_target_original_ids.append(original_id)
            else:
                for original_id in initial_target_original_ids:
                    if original_id < current_graph.shape[0]:
                        current_target_nodes.append(original_id)
                        current_target_original_ids.append(original_id)
            
            time_step_target_nodes[time_step] = current_target_nodes
            time_step_target_original_ids[time_step] = current_target_original_ids
            
            print(f"Time step {time_step} target node count: {len(current_target_nodes)}")
            if current_target_original_ids:
                print(f"  Target node original IDs: {current_target_original_ids}")
            else:
                print("  No target nodes are available in the current time step")
            
            current_nodes = list(range(current_graph.shape[0]))
            current_time_step_fake_nodes = self.generate_fake_nodes_excluding_targets(
                current_nodes, current_target_nodes, fake_nodes_for_current_step)
            if len(current_time_step_fake_nodes) < fake_nodes_for_current_step:
                print(f"  Warning：Expected {fake_nodes_for_current_step} fake nodes, but only {len(current_time_step_fake_nodes)} are available（insufficient nodes remain after excluding target nodes）")
            result['time_step_fake_node_counts'][time_step] = len(current_time_step_fake_nodes)
            
            print(f"  Generated fake nodes: {current_time_step_fake_nodes}")
            
            
            original_node_count = current_graph.shape[0]
            
            
            current_time_step_fake_edges = []
            base_graph_with_fakes = current_graph.copy()
            fake_connection_matrix = base_graph_with_fakes.copy()
            fake_edge_counts = {node: 0 for node in current_time_step_fake_nodes}
            
            
            if alpha == 2:
                max_edges_per_fake_node = self.read_average_degree_for_time_step(time_step, dataset_name, reconstructed_dir)
                print(f"    Time step {time_step} average-degree limit after perturbation: {max_edges_per_fake_node}")
                print("    Using the MGA strategy to process all fake nodes in the current time step.")

                mga_new_edges = self._execute_mga_strategy(
                    time_step=time_step,
                    fake_nodes=list(current_time_step_fake_nodes),
                    current_target_nodes=current_target_nodes,
                    current_graph=base_graph_with_fakes,
                    fake_connection_matrix=fake_connection_matrix,
                    fake_edge_counts=fake_edge_counts,
                    edge_limit=max_edges_per_fake_node
                )
                current_time_step_fake_edges.extend(mga_new_edges)
            else:
                edge_budget_rng = random.Random(step_seed)
                for v_new in current_time_step_fake_nodes:
                    new_edges = []

                    max_edges_per_fake_node = self.read_average_degree_for_time_step(time_step, dataset_name, reconstructed_dir)
                    print(f"    Time step {time_step} average-degree limit after perturbation: {max_edges_per_fake_node}")

                    if alpha == 0:  
                        print(f"    Processing fake node with RVA strategy {v_new}")

                        total_num = fake_connection_matrix.shape[0]
                        possible_targets = [j for j in range(total_num) if j != v_new]
                        random.shuffle(possible_targets)

                        max_edges_budget = max(1, max_edges_per_fake_node)
                        random_edge_cap = edge_budget_rng.randint(1, max_edges_budget)
                        print(f"      RVA strategy：Random edge cap for this fake node: {random_edge_cap}")

                        added_edges = 0
                        for target_node in possible_targets:
                            if fake_connection_matrix[v_new][target_node]:
                                continue
                            if added_edges >= random_edge_cap:
                                break
                            new_edges.append([v_new, target_node])
                            fake_connection_matrix[v_new][target_node] = True
                            fake_connection_matrix[target_node][v_new] = True
                            added_edges += 1

                    elif alpha == 1:  
                        print(f"    Processing fake node with RNA strategy {v_new}")

                        shuffled_targets = current_target_nodes[:]
                        random.shuffle(shuffled_targets)
                        connected = False
                        for target_node in shuffled_targets:
                            if fake_connection_matrix[v_new][target_node]:
                                continue
                            new_edges.append([v_new, target_node])
                            fake_connection_matrix[v_new][target_node] = True
                            fake_connection_matrix[target_node][v_new] = True
                            connected = True
                            break

                        
                        if not connected:
                            available_fake_nodes = [
                                fake for fake in current_time_step_fake_nodes
                                if fake != v_new and not fake_connection_matrix[v_new][fake]
                            ]
                            if available_fake_nodes:
                                random.shuffle(available_fake_nodes)
                                fallback_node = available_fake_nodes[0]
                                new_edges.append([v_new, fallback_node])
                                fake_connection_matrix[v_new][fallback_node] = True
                                fake_connection_matrix[fallback_node][v_new] = True
                                connected = True
                            else:
                                real_candidates = [
                                    node for node in range(original_node_count)
                                    if node != v_new
                                    and node not in current_time_step_fake_nodes
                                    and not fake_connection_matrix[v_new][node]
                                ]
                                if real_candidates:
                                    random.shuffle(real_candidates)
                                    fallback_node = real_candidates[0]
                                    new_edges.append([v_new, fallback_node])
                                    fake_connection_matrix[v_new][fallback_node] = True
                                    fake_connection_matrix[fallback_node][v_new] = True
                                    connected = True

                    current_time_step_fake_edges.extend(new_edges)
            
            result['time_step_fake_nodes'][time_step] = list(current_time_step_fake_nodes)
            result['time_step_fake_edges'][time_step] = list(current_time_step_fake_edges)
        
        result['time_step_target_original_ids'] = time_step_target_original_ids
        self._record_target_nodes(strategy, initial_target_original_ids, time_step_target_original_ids)
        return result


def load_dataset_by_selection_lcc(attack_obj, selection: str):
    

    dataset_options = {
        "1": {
            "tag": "email",
            "path": os.path.join("dataset", "email", "email.txt"),
            "reader": utils.read_email,
        },
        "2": {
            "tag": "reply",
            "path": os.path.join("dataset", "reply", "reply.txt"),
            "reader": utils.read_reply,
        },
    }
    dir_dataset_options = {
        "3": "as-733",
        "4": "Synthetic",
    }

    if selection in dataset_options:
        option = dataset_options[selection]
        dataset_tag = option["tag"]
        data_path = option["path"]
        reader = option["reader"]
        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
            return {"time_series_graphs": [], "dataset_infos": [], "node_mappings": [], "reverse_node_mappings": [], "max_node_ids": []}, dataset_tag
        print(f"Loading {dataset_tag} dataset from: {data_path}")
        graphs_nx = reader(data_path)
        time_series_graphs = []
        dataset_infos = []
        node_mappings = []
        reverse_node_mappings = []
        max_node_ids = []
        for idx, g in enumerate(graphs_nx):
            nodes = sorted(g.nodes())
            mapping = {node: i for i, node in enumerate(nodes)}
            reverse_mapping = {i: node for node, i in mapping.items()}
            adj = nx.to_numpy_array(g, dtype=bool, nodelist=nodes)
            time_series_graphs.append(adj)
            dataset_infos.append({
                "filename": f"{dataset_tag}_t{idx}.txt",
                "description": "",
                "node_count": len(nodes),
                "edge_count": g.number_of_edges(),
                "date": ""
            })
            node_mappings.append(mapping)
            reverse_node_mappings.append(reverse_mapping)
            max_node_ids.append(len(nodes) - 1 if nodes else -1)

        attack_obj.time_series_graphs = time_series_graphs
        attack_obj.time_steps = len(time_series_graphs)
        attack_obj.node_mappings = node_mappings
        attack_obj.reverse_node_mappings = reverse_node_mappings
        attack_obj.dataset_tag = dataset_tag
        attack_obj.dataset = dataset_tag
        return {
            "time_series_graphs": time_series_graphs,
            "dataset_infos": dataset_infos,
            "node_mappings": node_mappings,
            "reverse_node_mappings": reverse_node_mappings,
            "max_node_ids": max_node_ids,
        }, dataset_tag

    dataset_tag = dir_dataset_options.get(selection, "as-733")
    dataset_path = os.path.join("dataset", dataset_tag)
    result = attack_obj.load_time_series_graphs(dataset_path, max_time_steps=0)
    attack_obj.dataset_tag = dataset_tag
    attack_obj.dataset = dataset_tag
    return result, dataset_tag


def main() -> None:
    

    attack = DynamicAttackLCC()
    attack_labels = ['RVA', 'RNA', 'MGA']

    print("=== Clustering Coefficient Attack Fake Node Allocation Strategy Selection ===")
    print("0 - Even allocation")
    print("1 - Random allocation")
    print("2 - Decay allocation(θ=0.6)")

    strategy = 0
    while True:
        try:
            strategy = int(input("Select fake node allocation strategy (0/1/2): "))
            if strategy in (0, 1, 2):
                print(f"Selected: {get_strategy_name(strategy)}")
                break
            print("Invalid choice, please enter 0, 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")

    attack.current_strategy = strategy

    
    print("\n=== Select dataset ===")
    print("1 - email (dataset/email/email.txt)")
    print("2 - reply (dataset/reply/reply.txt)")
    print("3 - as-733 (dataset/as-733) [default]")
    print("4 - Synthetic (dataset/Synthetic)")
    ds_choice = input("Enter dataset id (1/2/3/4, press Enter for 3): ").strip()
    if ds_choice not in ("1", "2", "3", "4"):
        ds_choice = "3"
    
        ds_choice = "3"
    
    result, dataset_tag = load_dataset_by_selection_lcc(attack, ds_choice)
    dataset_name = dataset_tag
    time_series_graphs = result.get('time_series_graphs', [])
    if not time_series_graphs:
        print("No time-series graphs loaded; abort.")
        return

    time_steps = len(time_series_graphs)
    print(f"Total time steps: {time_steps}.")

    beta = DEFAULT_BETA 
    print(f"Using fake node ratio beta = {beta}")

    node_mapping_t0 = result['node_mappings'][0] if result.get('node_mappings') else {}
    total_nodes_t0 = time_series_graphs[0].shape[0]
    available_original_ids = list(node_mapping_t0.keys()) if node_mapping_t0 else list(range(total_nodes_t0))

    if not available_original_ids:
        print("No original nodes are available in the initial time step; the program is ending.")
        return

    target_count = max(1, int(DEFAULT_TARGET_RATIO * len(available_original_ids)))
    if target_count > len(available_original_ids):
        target_count = len(available_original_ids)

    random.seed(attack.SEED)
    initial_target_original_ids = random.sample(available_original_ids, target_count)
    if node_mapping_t0:
        initial_target_nodes = [node_mapping_t0.get(original_id, original_id) for original_id in initial_target_original_ids]
    else:
        initial_target_nodes = initial_target_original_ids.copy()

    result['original_target_nodes'] = initial_target_original_ids

    print(f"Initial target node original IDs: {initial_target_original_ids}")
    print(f"Initial target node mapped IDs: {initial_target_nodes}")

    reconstructed_dir = "./reconstructed_graph/"

    
    L = time_series_graphs[0].shape[0]
    total_fake_nodes = int(beta * L)
    print(f"Unified preallocation：total fake node count l = β * L = {total_fake_nodes}")
    fake_nodes_per_time_step = allocate_fake_nodes(total_fake_nodes, time_steps, strategy)
    print(f"Allocation strategy {get_strategy_name(strategy)} fake node count list: {fake_nodes_per_time_step}")

    time_step_target_nodes: Dict[int, List[int]] = {}
    time_step_target_original_ids: Dict[int, List[int]] = {}
    unified_fake_nodes: Dict[int, List[int]] = {}

    for time_step in range(time_steps):
        mapping_for_time = result['node_mappings'][time_step] if result.get('node_mappings') and time_step < len(result['node_mappings']) else {}
        current_graph = time_series_graphs[time_step]

        current_target_nodes: List[int] = []
        current_target_original_ids: List[int] = []
        if mapping_for_time:
            for original_id in initial_target_original_ids:
                mapped_id = mapping_for_time.get(original_id)
                if mapped_id is not None and mapped_id < current_graph.shape[0]:
                    current_target_nodes.append(mapped_id)
                    current_target_original_ids.append(original_id)
        else:
            for original_id in initial_target_original_ids:
                if original_id < current_graph.shape[0]:
                    current_target_nodes.append(original_id)
                    current_target_original_ids.append(original_id)

        time_step_target_nodes[time_step] = current_target_nodes
        time_step_target_original_ids[time_step] = current_target_original_ids

        step_seed = attack._seed_rng_for_step("UNIFIED_FAKE", time_step)
        fake_nodes_for_step = fake_nodes_per_time_step[time_step]
        current_nodes = list(range(current_graph.shape[0]))
        allocated_fake_nodes = attack.generate_fake_nodes_excluding_targets(
            current_nodes, current_target_nodes, fake_nodes_for_step)
        if len(allocated_fake_nodes) < fake_nodes_for_step:
            print(f"Time step {time_step}: Not enough nodes available; expected {fake_nodes_for_step}，actual {len(allocated_fake_nodes)}（target nodes excluded）")
        unified_fake_nodes[time_step] = allocated_fake_nodes

        print(f"Time step {time_step}: target node count = {len(current_target_nodes)}，preallocated fake node count = {fake_nodes_for_step}，result: {allocated_fake_nodes}")

    attack._record_target_nodes(strategy, initial_target_original_ids, time_step_target_original_ids)

    for alpha, attack_name in enumerate(attack_labels):
        print(f"\n=== Running {attack_name} attack（unified fake nodes） ===")
        attack_result = {
            'time_step_fake_node_counts': {t: len(unified_fake_nodes.get(t, [])) for t in range(time_steps)},
            'time_step_fake_nodes': unified_fake_nodes,
            'time_step_fake_edges': {},
            'original_target_nodes': initial_target_original_ids,
            'time_step_target_original_ids': time_step_target_original_ids
        }

        for time_step in range(time_steps):
            step_seed = attack._seed_rng_for_step(attack_name, time_step)
            current_graph = time_series_graphs[time_step]
            current_target_nodes = time_step_target_nodes.get(time_step, [])
            current_fake_nodes = unified_fake_nodes.get(time_step, [])

            current_time_step_fake_edges: List[List[int]] = []
            original_node_count = current_graph.shape[0]
            base_graph_with_fakes = current_graph.copy()
            fake_connection_matrix = base_graph_with_fakes.copy()
            fake_edge_counts = {node: 0 for node in current_fake_nodes}

            if alpha == 2:  
                max_edges_per_fake_node = attack.read_average_degree_for_time_step(time_step, dataset_name, reconstructed_dir)
                print(f"    Time step {time_step} average-degree limit after perturbation: {max_edges_per_fake_node}")
                mga_new_edges = attack._execute_mga_strategy(
                    time_step=time_step,
                    fake_nodes=list(current_fake_nodes),
                    current_target_nodes=current_target_nodes,
                    current_graph=base_graph_with_fakes,
                    fake_connection_matrix=fake_connection_matrix,
                    fake_edge_counts=fake_edge_counts,
                    edge_limit=max_edges_per_fake_node
                )
                current_time_step_fake_edges.extend(mga_new_edges)
            else:
                edge_budget_rng = random.Random(step_seed)
                for v_new in current_fake_nodes:
                    new_edges: List[List[int]] = []
                    max_edges_per_fake_node = attack.read_average_degree_for_time_step(time_step, dataset_name, reconstructed_dir)
                    print(f"    Time step {time_step} average-degree limit after perturbation: {max_edges_per_fake_node}")

                    if alpha == 0:  
                        print(f"    Processing fake node with RVA strategy {v_new}")
                        total_num = fake_connection_matrix.shape[0]
                        possible_targets = [j for j in range(total_num) if j != v_new]
                        random.shuffle(possible_targets)
                        max_edges_budget = max(1, max_edges_per_fake_node)
                        random_edge_cap = edge_budget_rng.randint(1, max_edges_budget)
                        print(f"      RVA strategy：Random edge cap for this fake node: {random_edge_cap}")
                        added_edges = 0
                        for target_node in possible_targets:
                            if fake_connection_matrix[v_new][target_node]:
                                continue
                            if added_edges >= random_edge_cap:
                                break
                            new_edges.append([v_new, target_node])
                            fake_connection_matrix[v_new][target_node] = True
                            fake_connection_matrix[target_node][v_new] = True
                            added_edges += 1

                    elif alpha == 1:  
                        print(f"    Processing fake node with RNA strategy {v_new}")
                        shuffled_targets = current_target_nodes[:]
                        random.shuffle(shuffled_targets)
                        connected = False
                        for target_node in shuffled_targets:
                            if fake_connection_matrix[v_new][target_node]:
                                continue
                            new_edges.append([v_new, target_node])
                            fake_connection_matrix[v_new][target_node] = True
                            fake_connection_matrix[target_node][v_new] = True
                            connected = True
                            break
                        if not connected:
                            available_fake_nodes = [
                                fake for fake in current_fake_nodes
                                if fake != v_new and not fake_connection_matrix[v_new][fake]
                            ]
                            if available_fake_nodes:
                                random.shuffle(available_fake_nodes)
                                fallback_node = available_fake_nodes[0]
                                new_edges.append([v_new, fallback_node])
                                fake_connection_matrix[v_new][fallback_node] = True
                                fake_connection_matrix[fallback_node][v_new] = True
                                connected = True
                            else:
                                real_candidates = [
                                    node for node in range(original_node_count)
                                    if node != v_new
                                    and node not in current_fake_nodes
                                    and not fake_connection_matrix[v_new][node]
                                ]
                                if real_candidates:
                                    random.shuffle(real_candidates)
                                    fallback_node = real_candidates[0]
                                    new_edges.append([v_new, fallback_node])
                                    fake_connection_matrix[v_new][fallback_node] = True
                                    fake_connection_matrix[fallback_node][v_new] = True
                                    connected = True

                    current_time_step_fake_edges.extend(new_edges)

            attack_result['time_step_fake_edges'][time_step] = list(current_time_step_fake_edges)

        output_dir = os.path.join("output_py", str(dataset_tag), "LCC", attack_name)
        attacked_graphs = attack.apply_attack(
            time_series_graphs,
            attack_result,
            output_dir,
            attack_name,
            result
        )

        print(f"{attack_name} attack complete; generated {len(attacked_graphs)} graphs after the attack。")


if __name__ == "__main__":
    main()
