import os
import csv
import json
import math
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from utils import read_email, read_reply

class OriginalVsAttackGainCalculator:
    

    def __init__(self):
        

        self.original_graphs = {}  
        self.attacked_graphs = {}  
        self.results = defaultdict(dict)  
        self.graph_metadata = {}
        self.decay_factor = 0.6
        self.k_star_size = 3
        self.target_metrics_records = []
        self.dataset_tag: Optional[str] = None
    
    def load_graph_from_file(self, file_path: str) -> nx.Graph:
        

        graph = nx.Graph()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    
                    if line.startswith('#'):
                        continue
                    
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            u, v = int(parts[0]), int(parts[1])
                            graph.add_edge(u, v)
                        except ValueError:
                            continue  
        except FileNotFoundError:
            print(f"Warning: file {file_path} does not exist")
            return nx.Graph()
        except Exception as e:
            print(f"Failed to read file {file_path}: {str(e)}")
            return nx.Graph()
        
        return graph
    
    def calculate_clustering_coefficient(self, graph: nx.Graph, node: int) -> float:
        

        if node not in graph.nodes():
            return 0.0
        
        neighbors = list(graph.neighbors(node))
        k_i = len(neighbors)
        
        
        if k_i < 2:
            return 0.0
        
        
        E_i = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if graph.has_edge(neighbors[i], neighbors[j]):
                    E_i += 1
        
        
        clustering_coefficient = (2 * E_i) / (k_i * (k_i - 1))
        return clustering_coefficient
    
    def calculate_degree_centrality(self, graph: nx.Graph, node: int) -> float:
        

        if node not in graph.nodes():
            return 0.0
        
        k_i = graph.degree(node)
        N = graph.number_of_nodes()
        
        
        if N <= 1:
            return 0.0
        
        degree_centrality = k_i / (N - 1)
        return degree_centrality
    
    def calculate_graph_metrics(self, graph: nx.Graph, metric_type: str) -> Dict[int, float]:
        

        metrics = {}
        
        for node in graph.nodes():
            if metric_type == 'clustering':
                metrics[node] = self.calculate_clustering_coefficient(graph, node)
            elif metric_type == 'degree_centrality':
                metrics[node] = self.calculate_degree_centrality(graph, node)
            else:
                raise ValueError(f"Unsupported metric type: {metric_type}")
        
        return metrics

    def get_graph_metadata(self, directory: str, prefix: str, time_step: int) -> Optional[Dict[str, object]]:
        key = (os.path.abspath(directory), prefix)
        return self.graph_metadata.get(key, {}).get(time_step)
    
    def load_target_nodes_from_file(self, attack_type: str, dataset_tag: Optional[str] = None) -> List[int]:
        

        target_nodes: List[int] = []
        dataset_tag = dataset_tag or self.dataset_tag

        file_path = None
        if dataset_tag:
            candidate = os.path.join("./target_nodes_record", attack_type, dataset_tag,
                                     f"{dataset_tag}_target_nodes_original_ids.txt")
            if os.path.exists(candidate):
                file_path = candidate
        if file_path is None:
            if attack_type == 'DC':
                file_path = "./target_nodes_record/DC/target_nodes_original_ids.txt"
            elif attack_type == 'LCC':
                file_path = "./target_nodes_record/LCC/target_nodes_original_ids.txt"
            else:
                print(f"Unsupported attack type: {attack_type}")
                return target_nodes

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    try:
                        node_id = int(line)
                        target_nodes.append(node_id)
                    except ValueError:
                        continue
        except FileNotFoundError:
            print(f"Target-node file not found: {file_path}")
        except Exception as e:
            print(f"Error reading target-node file {file_path}: {e}")

        print(f"Loaded {len(target_nodes)} target nodes from {file_path}")
        return target_nodes

    def identify_target_nodes(self, original_graph: nx.Graph, attacked_graph: nx.Graph, 
                             attack_type: str = None, top_k: int = 10) -> List[int]:
        

        if attack_type:
            
            return self.load_target_nodes_from_file(attack_type, self.dataset_tag)
        
        
        if original_graph.number_of_nodes() == 0:
            return []
        
        
        node_degrees = [(node, original_graph.degree(node)) for node in original_graph.nodes()]
        
        
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        
        
        target_nodes = [node for node, degree in node_degrees[:top_k]]
        
        return target_nodes
    
    def calculate_weighted_target_gain(self, target_gains: List[float], decay_factor: float = 0.6) -> float:
        

        if not target_gains:
            return 0.0
        
        total_weighted_gain = 0.0
        total_weight = 0.0
        for idx, gain in enumerate(target_gains):
            weight = decay_factor ** (len(target_gains) - 1 - idx)
            total_weighted_gain += gain * weight
            total_weight += weight
        
        return total_weighted_gain / total_weight if total_weight > 0 else 0.0
    
    def calculate_time_slice_gain(self, original_values: Dict[int, float], 
                                 attacked_values: Dict[int, float], 
                                 target_nodes: List[int] = None) -> Tuple[float, float]:
        

        all_nodes = set(original_values.keys()) | set(attacked_values.keys())
        total_gain = 0.0
        
        for node in all_nodes:
            original_val = original_values.get(node, 0.0)
            attacked_val = attacked_values.get(node, 0.0)
            total_gain += abs(original_val - attacked_val)
        
        
        target_gain = 0.0
        if target_nodes:
            for node in target_nodes:
                if node in original_values or node in attacked_values:
                    original_val = original_values.get(node, 0.0)
                    attacked_val = attacked_values.get(node, 0.0)
                    target_gain += abs(original_val - attacked_val)
        
        return total_gain, target_gain
    
    def compute_average_shortest_path_length(self, graph: nx.Graph, node: int) -> float:
        

        if node not in graph:
            return 0.0
        lengths = nx.single_source_shortest_path_length(graph, node)
        if len(lengths) <= 1:
            return 0.0
        total = 0.0
        count = 0
        for target, dist in lengths.items():
            if target == node:
                continue
            total += dist
            count += 1
        return total / count if count > 0 else 0.0
    
    def compute_target_graph_metrics(self, graph: nx.Graph, target_nodes: List[int]) -> Dict[str, float]:
        

        nodes = [node for node in target_nodes if node in graph]
        if not nodes:
            return {
                'avg_clustering': 0.0,
                'avg_degree_centrality': 0.0,
                'avg_two_star': 0.0,
                'avg_k_star': 0.0,
                'avg_triangle_count': 0.0,
                'avg_shortest_path': 0.0
            }
        
        triangles = nx.triangles(graph)
        clustering_vals = []
        degree_vals = []
        two_star_vals = []
        k_star_vals = []
        triangle_vals = []
        shortest_vals = []
        
        for node in nodes:
            degree = graph.degree(node)
            clustering_vals.append(self.calculate_clustering_coefficient(graph, node))
            degree_vals.append(self.calculate_degree_centrality(graph, node))
            two_star_vals.append(float(math.comb(degree, 2)) if degree >= 2 else 0.0)
            k_star_vals.append(float(math.comb(degree, self.k_star_size)) if degree >= self.k_star_size else 0.0)
            triangle_vals.append(float(triangles.get(node, 0)))
            shortest_vals.append(self.compute_average_shortest_path_length(graph, node))
        
        def avg(values: List[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0
        
        return {
            'avg_clustering': avg(clustering_vals),
            'avg_degree_centrality': avg(degree_vals),
            'avg_two_star': avg(two_star_vals),
            'avg_k_star': avg(k_star_vals),
            'avg_triangle_count': avg(triangle_vals),
            'avg_shortest_path': avg(shortest_vals)
        }
    
    def collect_target_metrics_record(self,
                                      attack_type: str,
                                      attack_method: str,
                                      time_step: int,
                                      original_graph: nx.Graph,
                                      attacked_graph: nx.Graph,
                                      target_nodes: List[int]) -> Dict[str, float]:
        

        original_metrics = self.compute_target_graph_metrics(original_graph, target_nodes)
        attacked_metrics = self.compute_target_graph_metrics(attacked_graph, target_nodes)
        
        record = {
            'attack_type': attack_type,
            'attack_method': attack_method,
            'time_step': time_step
        }
        
        for key in ['avg_clustering', 'avg_degree_centrality', 'avg_two_star',
                    'avg_k_star', 'avg_triangle_count', 'avg_shortest_path']:
            record[f'orig_{key}'] = original_metrics[key]
            record[f'attacked_{key}'] = attacked_metrics[key]
            record[f'{key}_gain'] = attacked_metrics[key] - original_metrics[key]
        
        return record
    
    def save_target_metrics_summary(self, output_path: str) -> None:
        

        if not self.target_metrics_records:
            print("No target-node metric data to summarize")
            return
        
        scenario_records = defaultdict(list)
        for record in self.target_metrics_records:
            key = (record['attack_type'], record['attack_method'])
            scenario_records[key].append(record)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fieldnames = [
            'attack_type',
            'attack_method',
            'time_steps',
            'orig_avg_clustering',
            'attacked_avg_clustering',
            'avg_clustering_gain',
            'orig_avg_degree_centrality',
            'attacked_avg_degree_centrality',
            'avg_degree_centrality_gain',
            'orig_avg_two_star',
            'attacked_avg_two_star',
            'avg_two_star_gain',
            'orig_avg_k_star',
            'attacked_avg_k_star',
            'avg_k_star_gain',
            'orig_avg_triangle_count',
            'attacked_avg_triangle_count',
            'avg_triangle_count_gain',
            'orig_avg_shortest_path',
            'attacked_avg_shortest_path',
            'avg_shortest_path_gain'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for (attack_type, attack_method), records in scenario_records.items():
                if not records:
                    continue
                records = sorted(records, key=lambda r: r.get('time_step', 0))
                time_steps_count = len(records)
                weights = []
                for idx in range(time_steps_count):
                    weight = self.decay_factor ** (time_steps_count - 1 - idx)
                    weights.append(weight)
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights] if total_weight > 0 else [0.0] * time_steps_count
                
                def weighted_average(field: str) -> float:
                    return sum(record[field] * normalized_weights[idx] for idx, record in enumerate(records))
                
                orig_avg_clustering = weighted_average('orig_avg_clustering')
                attacked_avg_clustering = weighted_average('attacked_avg_clustering')
                orig_avg_degree = weighted_average('orig_avg_degree_centrality')
                attacked_avg_degree = weighted_average('attacked_avg_degree_centrality')
                orig_avg_two_star = weighted_average('orig_avg_two_star')
                attacked_avg_two_star = weighted_average('attacked_avg_two_star')
                orig_avg_k_star = weighted_average('orig_avg_k_star')
                attacked_avg_k_star = weighted_average('attacked_avg_k_star')
                orig_avg_triangle = weighted_average('orig_avg_triangle_count')
                attacked_avg_triangle = weighted_average('attacked_avg_triangle_count')
                orig_avg_shortest_path = weighted_average('orig_avg_shortest_path')
                attacked_avg_shortest_path = weighted_average('attacked_avg_shortest_path')
                
                writer.writerow({
                    'attack_type': attack_type,
                    'attack_method': attack_method,
                    'time_steps': time_steps_count,
                    'orig_avg_clustering': orig_avg_clustering,
                    'attacked_avg_clustering': attacked_avg_clustering,
                    'avg_clustering_gain': attacked_avg_clustering - orig_avg_clustering,
                    'orig_avg_degree_centrality': orig_avg_degree,
                    'attacked_avg_degree_centrality': attacked_avg_degree,
                    'avg_degree_centrality_gain': attacked_avg_degree - orig_avg_degree,
                    'orig_avg_two_star': orig_avg_two_star,
                    'attacked_avg_two_star': attacked_avg_two_star,
                    'avg_two_star_gain': attacked_avg_two_star - orig_avg_two_star,
                    'orig_avg_k_star': orig_avg_k_star,
                    'attacked_avg_k_star': attacked_avg_k_star,
                    'avg_k_star_gain': attacked_avg_k_star - orig_avg_k_star,
                    'orig_avg_triangle_count': orig_avg_triangle,
                    'attacked_avg_triangle_count': attacked_avg_triangle,
                    'avg_triangle_count_gain': attacked_avg_triangle - orig_avg_triangle,
                    'orig_avg_shortest_path': orig_avg_shortest_path,
                    'attacked_avg_shortest_path': attacked_avg_shortest_path,
                    'avg_shortest_path_gain': attacked_avg_shortest_path - orig_avg_shortest_path
                })
        
        print(f"Target-node metrics summary saved to: {output_path}")
    
    def load_original_dataset_graphs(self, directory: str, dataset_tag: str) -> Dict[int, nx.Graph]:
        

        graphs: Dict[int, nx.Graph] = {}

        if not os.path.exists(directory):
            print(f"Warning: directory {directory} not found")
            return graphs

        tag = dataset_tag or ''
        reader = None
        data_file = None
        if tag == 'email':
            data_file = os.path.join(directory, 'email.txt')
            reader = read_email
        elif tag == 'reply':
            data_file = os.path.join(directory, 'reply.txt')
            reader = read_reply

        if data_file:
            if not os.path.exists(data_file):
                print(f"Warning: raw data file not found: {data_file}")
                return graphs
            try:
                series = reader(data_file)
                metadata_key = (os.path.abspath(directory), os.path.basename(data_file))
                self.graph_metadata[metadata_key] = {}
                for idx, g in enumerate(series):
                    graphs[idx] = g
                    self.graph_metadata[metadata_key][idx] = {
                        'file_name': os.path.basename(data_file),
                        'file_path': data_file,
                        'dataset': tag,
                        'time_step': idx
                    }
                print(f"Loaded {len(graphs)} time slices from original dataset (dataset={tag})")
                return graphs
            except Exception as e:
                print(f"Error reading raw data file {data_file}: {e}")
                return graphs

        files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        as_files = [f for f in files if f.startswith('as')]
        if as_files:
            files = as_files
        if not files:
            print(f"Warning: no txt files found in {directory}")
            return graphs

        files.sort()
        metadata_key = (os.path.abspath(directory), 'original')
        self.graph_metadata[metadata_key] = {}

        print(f"Found {len(files)} original dataset files")
        for i, file in enumerate(files):
            try:
                file_path = os.path.join(directory, file)
                graph = self.load_graph_from_file(file_path)
                if graph.number_of_nodes() > 0:
                    graphs[i] = graph
                    self.graph_metadata[metadata_key][i] = {
                        'file_name': file,
                        'file_path': file_path,
                        'dataset': tag or None,
                        'time_step': i
                    }
                    print(f"Loaded {file} as time step {i}, nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
                else:
                    print(f"Skip empty graph file: {file}")
            except Exception as e:
                print(f"Error loading file {file}: {str(e)}")
                continue

        print(f"Successfully loaded {len(graphs)} time steps of original dataset")
        return graphs

    def load_time_series_graphs(self, directory: str, prefix: str) -> Dict[int, nx.Graph]:
        

        graphs = {}
        metadata_key = (os.path.abspath(directory), prefix)
        self.graph_metadata[metadata_key] = {}
        
        if not os.path.exists(directory):
            print(f"Warning: directory {directory} does not exist")
            return graphs
        
        
        files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.txt')]
        
        for file in sorted(files):
            
            try:
                
                if '_t' in file:
                    
                    time_step = int(file.split('_t')[1].replace('.txt', ''))
                elif 'TimeStep' in file:
                    
                    time_step = int(file.split('TimeStep')[1].replace('.txt', ''))
                else:
                    continue
                
                dataset = None
                name_parts = file.replace('.txt', '').split('_')
                for part in name_parts:
                    if part.startswith('Dataset'):
                        try:
                            dataset = int(part.replace('Dataset', ''))
                        except ValueError:
                            dataset = None
                        break

                file_path = os.path.join(directory, file)
                graph = self.load_graph_from_file(file_path)
                graphs[time_step] = graph
                self.graph_metadata[metadata_key][time_step] = {
                    'file_name': file,
                    'file_path': file_path,
                    'dataset': dataset,
                    'time_step': time_step
                }
                
            except (ValueError, IndexError) as e:
                print(f"Unable to parse filename {file}: {str(e)}")
                continue
        
        return graphs
    
    def process_attack_scenario(self, original_dir: str, attacked_dir: str,                               attack_type: str, attack_method: str,                               metric_type: str, dataset_tag: Optional[str] = None,                              collect_target_metrics: bool = False) -> Dict[str, float]:
        

        tag = dataset_tag or self.dataset_tag
        print(f"Processing attack scenario: {attack_type}/{attack_method}, metric: {metric_type}, dataset: {tag}")
        
        original_graphs = self.load_original_dataset_graphs(original_dir, tag or "")
        attacked_prefix = f'AttackedGraph_{attack_method}_{attack_type}'
        attacked_graphs = self.load_time_series_graphs(attacked_dir, attacked_prefix)
        
        if not original_graphs:
            print(f"Warning: original-dataset graph data not found: {original_dir}")
            return {}
        
        if not attacked_graphs:
            print(f"Warning: attacked-graph data not found: {attacked_dir}")
            return {}
        
        print(f"Found {len(original_graphs)} original time steps and {len(attacked_graphs)} attacked time steps")
        
        gains = []
        total_gain_values = []
        target_gain_values = []
        
        common_time_steps = set(original_graphs.keys()) & set(attacked_graphs.keys())
        
        for time_step in sorted(common_time_steps):
            original_graph = original_graphs[time_step]
            attacked_graph = attacked_graphs[time_step]
            target_nodes = self.identify_target_nodes(original_graph, attacked_graph, attack_type)
            original_metrics = self.calculate_graph_metrics(original_graph, metric_type)
            attacked_metrics = self.calculate_graph_metrics(attacked_graph, metric_type)
            total_gain, target_gain = self.calculate_time_slice_gain(original_metrics, attacked_metrics, target_nodes)
            gains.append({
                'time_step': time_step,
                'total_gain': total_gain,
                'target_gain': target_gain
            })
            total_gain_values.append(total_gain)
            target_gain_values.append(target_gain)
            
            if collect_target_metrics:
                metrics_record = self.collect_target_metrics_record(
                    attack_type=attack_type,
                    attack_method=attack_method,
                    time_step=time_step,
                    original_graph=original_graph,
                    attacked_graph=attacked_graph,
                    target_nodes=target_nodes
                )
                self.target_metrics_records.append(metrics_record)
            
            print(f"Time step {time_step}: total-graph attack gain = {total_gain:.6f}, target-node attack gain = {target_gain:.6f}")
        
        aggregated_result = None
        if gains:
            weights = []
            time_steps_count = len(gains)
            for idx in range(time_steps_count):
                weight = self.decay_factor ** (time_steps_count - 1 - idx)
                weights.append(weight)
            total_weight = sum(weights)
            for idx, record in enumerate(gains):
                if total_weight > 0:
                    normalized = weights[idx] / total_weight
                else:
                    normalized = 0.0
                record['decay_weight'] = normalized
                record['decay_weight_percent'] = normalized * 100.0
                record['weighted_total_gain'] = record['total_gain'] * normalized
                record['weighted_target_gain'] = record['target_gain'] * normalized
            aggregated_result = {
                'time_steps': time_steps_count,
                'total_gain_sum': float(sum(total_gain_values)),
                'total_gain_avg': float(np.mean(total_gain_values)) if total_gain_values else 0.0,
                'total_gain_max': float(max(total_gain_values)) if total_gain_values else 0.0,
                'total_gain_min': float(min(total_gain_values)) if total_gain_values else 0.0,
                'target_gain_sum': float(sum(target_gain_values)),
                'target_gain_avg': float(np.mean(target_gain_values)) if target_gain_values else 0.0,
                'target_gain_max': float(max(target_gain_values)) if target_gain_values else 0.0,
                'target_gain_min': float(min(target_gain_values)) if target_gain_values else 0.0,
                'decay_weighted_total_gain': float(sum(record['weighted_total_gain'] for record in gains)),
                'decay_weighted_target_gain': float(sum(record['weighted_target_gain'] for record in gains))
            }
        
        return aggregated_result or {}

    def save_results_to_csv(self, result: Dict[str, float],                            output_path: str, attack_type: str,                            attack_method: str, metric_type: str) -> None:
        

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['# Original Dataset vs. Attacked Graph Gain Calculation Results'])
            writer.writerow([f'# Attack type: {attack_type}'])
            writer.writerow([f'# Attack method: {attack_method}'])
            writer.writerow([f'# Metric type: {metric_type}'])
            writer.writerow(['# Summary includes only the overall decay-weighted results'])
            writer.writerow([
                'TimeSteps',
                'TotalGainSum',
                'TotalGainAvg',
                'TotalGainMax',
                'TotalGainMin',
                'TargetGainSum',
                'TargetGainAvg',
                'TargetGainMax',
                'TargetGainMin',
                'DecayWeightedTotalGain',
                'DecayWeightedTargetGain'
            ])
            writer.writerow([
                int(result.get('time_steps', 0)),
                f"{result.get('total_gain_sum', 0.0):.6f}",
                f"{result.get('total_gain_avg', 0.0):.6f}",
                f"{result.get('total_gain_max', 0.0):.6f}",
                f"{result.get('total_gain_min', 0.0):.6f}",
                f"{result.get('target_gain_sum', 0.0):.6f}",
                f"{result.get('target_gain_avg', 0.0):.6f}",
                f"{result.get('target_gain_max', 0.0):.6f}",
                f"{result.get('target_gain_min', 0.0):.6f}",
                f"{result.get('decay_weighted_total_gain', 0.0):.6f}",
                f"{result.get('decay_weighted_target_gain', 0.0):.6f}"
            ])
        
        print(f"Results saved to: {output_path}")

    def _choose_dataset_tag(self) -> Optional[str]:
        

        base = "./output_py"
        if not os.path.isdir(base):
            return None
        candidates = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        if not candidates:
            return None
        print("\n=== Select a dataset ===")
        for idx, name in enumerate(candidates, 1):
            print(f"{idx} - {name}")
        choice = input(f"Enter a number (1-{len(candidates)}; press Enter for default 1): ").strip()
        try:
            sel = int(choice) if choice else 1
            if sel < 1 or sel > len(candidates):
                raise ValueError
        except ValueError:
            sel = 1
            print(f"Invalid input; using default {candidates[0]}")
        return candidates[sel - 1]

    def process_all_scenarios(self, 
                            dataset_tag: Optional[str] = None,
                            output_base_dir: str = "./original_vs_attack_gain_results/") -> None:
        

        if dataset_tag is None:
            dataset_tag = self._choose_dataset_tag()
        if not dataset_tag:
            print("No dataset found, aborting.")
            return
        self.dataset_tag = dataset_tag

        original_base_dir = os.path.join("./dataset", dataset_tag)
        attacked_base_dir = os.path.join("./output_py", dataset_tag)
        output_base_dir = os.path.join(output_base_dir, dataset_tag)

        print("Starting processing all attack scenarios...")
        print(f"Dataset: {dataset_tag}")
        self.target_metrics_records = []
        
        attack_types = ['DC', 'LCC']
        attack_methods = ['MGA', 'RNA', 'RVA']
        metric_types = ['clustering', 'degree_centrality']
        
        os.makedirs(output_base_dir, exist_ok=True)
        
        total_scenarios = len(attack_types) * len(attack_methods) * len(metric_types)
        current_scenario = 0
        
        for attack_type in attack_types:
            for attack_method in attack_methods:
                for metric_type in metric_types:
                    current_scenario += 1
                    print(f"\nScenario {current_scenario}/{total_scenarios}: {attack_type}/{attack_method}/{metric_type}")
                    
                    attacked_dir = os.path.join(attacked_base_dir, attack_type, attack_method)
                    
                    if not os.path.exists(attacked_dir):
                        print(f"Skip: attack dir {attacked_dir} not found")
                        continue
                    
                    try:
                        collect_metrics = (metric_type == 'clustering')
                        results = self.process_attack_scenario(
                            original_dir=original_base_dir,
                            attacked_dir=attacked_dir,
                            attack_type=attack_type,
                            attack_method=attack_method,
                            metric_type=metric_type,
                            dataset_tag=dataset_tag,
                            collect_target_metrics=collect_metrics
                        )
                        
                        if results:
                            output_dir = os.path.join(output_base_dir, attack_type)
                            os.makedirs(output_dir, exist_ok=True)
                            
                            filename = f"{attack_method}_{metric_type}_gain.csv"
                            output_path = os.path.join(output_dir, filename)
                            
                            self.save_results_to_csv(
                                result=results,
                                output_path=output_path,
                                attack_type=attack_type,
                                attack_method=attack_method,
                                metric_type=metric_type
                            )
                        else:
                            print(f"Warning: no results for {attack_type}/{attack_method}/{metric_type}")
                    
                    except Exception as e:
                        print(f"Error processing {attack_type}/{attack_method}/{metric_type}: {str(e)}")
        
        target_metrics_path = os.path.join(output_base_dir, 'target_metrics_summary.csv')
        self.save_target_metrics_summary(target_metrics_path)
        print("\nAll scenarios processed.")

    def generate_summary_report(self, output_base_dir: str = "./original_vs_attack_gain_results/", dataset_tag: Optional[str] = None) -> None:
        

        tag = dataset_tag or self.dataset_tag
        if tag:
            output_base_dir = os.path.join(output_base_dir, tag)
        print("Generating summary report...")
        summary_data = []
        for attack_type in ['DC', 'LCC']:
            attack_dir = os.path.join(output_base_dir, attack_type)
            if not os.path.exists(attack_dir):
                continue
            for file in os.listdir(attack_dir):
                if not file.endswith('_gain.csv'):
                    continue
                file_path = os.path.join(attack_dir, file)
                parts = file.replace('_gain.csv', '').split('_')
                if len(parts) < 2:
                    continue
                attack_method = parts[0]
                metric_type = '_'.join(parts[1:])
                try:
                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                        reader = csv.reader(f)
                        header = None
                        header_index = {}
                        aggregated_row = None
                        for row in reader:
                            if not row:
                                continue
                            row[0] = row[0].lstrip('﻿')
                            if row[0].startswith('#'):
                                continue
                            if header is None:
                                header = row
                                header_index = {name: idx for idx, name in enumerate(row)}
                                continue
                            aggregated_row = row
                            break
                        if header and aggregated_row:
                            def get_value(name: str, default: float = 0.0) -> float:
                                try:
                                    idx = header_index[name]
                                    return float(aggregated_row[idx])
                                except (KeyError, ValueError, IndexError):
                                    return default
                            summary_entry = {
                                'attack_type': attack_type,
                                'attack_method': attack_method,
                                'metric_type': metric_type,
                                'total_gain_sum': get_value('TotalGainSum'),
                                'total_gain_avg': get_value('TotalGainAvg'),
                                'total_gain_max': get_value('TotalGainMax'),
                                'total_gain_min': get_value('TotalGainMin'),
                                'target_gain_sum': get_value('TargetGainSum'),
                                'target_gain_avg': get_value('TargetGainAvg'),
                                'target_gain_max': get_value('TargetGainMax'),
                                'target_gain_min': get_value('TargetGainMin'),
                                'target_gain_weighted': get_value('DecayWeightedTargetGain'),
                                'time_steps': int(get_value('TimeSteps', 0.0))
                            }
                            summary_entry['total_gain_decay_weighted'] = get_value('DecayWeightedTotalGain')
                            summary_entry['target_gain_decay_weighted'] = get_value('DecayWeightedTargetGain')
                            summary_data.append(summary_entry)
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
        summary_path = os.path.join(output_base_dir, 'summary_report.csv')
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            if summary_data:
                writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
                writer.writeheader()
                writer.writerows(summary_data)
        print(f"Summary report saved to: {summary_path}")

def main():
    

    print("Original Dataset vs. Attacked Graph Gain Calculator starting...")
    
    
    calculator = OriginalVsAttackGainCalculator()
    
    
    calculator.process_all_scenarios()
    
    
    calculator.generate_summary_report()
    
    print("Original Dataset vs. Attacked Graph Gain Calculation completed")

if __name__ == "__main__":
    main()
