import os
import csv
import json
import networkx as nx
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import utils


DEFAULT_LDP_EPSILON = 8.0

class AttackGainCalculator:
    

    def __init__(self, dataset_tag: Optional[str] = None):
        

        self.original_graphs = {}  
        self.attacked_graphs = {}  
        self.results = defaultdict(dict)  
        self.graph_metadata = {}
        self.ldp_epsilon = DEFAULT_LDP_EPSILON
        self.ldp_seed = 2025  
        self.decay_factor = 0.6  
        self.k_star_size = 3
        self.target_metrics_records = []
        self.target_metrics_seen: Set[Tuple[str, str, int]] = set()
        self.target_metrics_timeseries = []
        self.dataset_tag = dataset_tag
    
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

    def calculate_degree_values(self, graph: nx.Graph) -> Dict[int, float]:
        

        if graph.number_of_nodes() == 0:
            return {}
        return {node: float(graph.degree(node)) for node in graph.nodes()}

    def normalize_degree_values(self, degree_values: Dict[int, float], node_count: int) -> Dict[int, float]:
        

        if not degree_values:
            return {}
        if node_count <= 1:
            return {node: 0.0 for node in degree_values}
        denominator = float(node_count - 1)
        return {node: value / denominator for node, value in degree_values.items()}

    def ensure_non_negative_metrics(self, metrics: Dict[int, float]) -> Dict[int, float]:
        

        if not metrics:
            return metrics
        for node_id, value in list(metrics.items()):
            if value < 0.0:
                metrics[node_id] = 0.0
        return metrics

    def get_graph_metadata(self, directory: str, prefix: str, time_step: int) -> Optional[Dict[str, object]]:
        

        key = (os.path.abspath(directory), prefix)
        return self.graph_metadata.get(key, {}).get(time_step)

    def _parse_fake_node_file(self, file_path: str) -> Dict[int, int]:
        

        mapping: Dict[int, int] = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith('#'):
                        continue
                    if "# Fake Node:" in line and "(Original ID:" in line:
                        try:
                            fake_segment = line.split("# Fake Node:", 1)[1]
                            fake_part = fake_segment.split("(Original ID:", 1)[0].strip()
                            original_part = line.split("(Original ID:", 1)[1].rstrip(')').strip()
                            fake_id = int(fake_part)
                            original_id = int(original_part)
                            mapping[fake_id] = original_id
                        except (IndexError, ValueError):
                            continue
        except FileNotFoundError:
            return mapping
        except Exception as exc:
            print(f"Failed to read fake-node file {file_path}: {exc}")
        return mapping

    def load_fake_node_mapping(self, attack_type: str, attack_method: str,
                               time_step: int, dataset: Optional[int] = None) -> Dict[int, int]:
        

        dataset_prefix = self.dataset_tag or ""
        base_dir = os.path.join("./FakeNodeID", dataset_prefix, attack_type, attack_method)
        mapping: Dict[int, int] = {}

        if not os.path.exists(base_dir):
            print(f"Warning: fake-node directory does not exist: {base_dir}")
            return mapping

        candidate_paths = []
        if dataset is not None:
            file_name = f"FakeNodeID_{attack_method}_{attack_type}_Dataset{dataset}_TimeStep{time_step}.txt"
            candidate_paths.append(os.path.join(base_dir, file_name))

        
        for file in sorted(os.listdir(base_dir)):
            if not file.startswith(f"FakeNodeID_{attack_method}_{attack_type}_"):
                continue
            if f"TimeStep{time_step}" not in file:
                continue
            candidate_paths.append(os.path.join(base_dir, file))

        seen = set()
        for path in candidate_paths:
            if path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            mapping = self._parse_fake_node_file(path)
            if mapping:
                return mapping

        if not mapping:
            print(f"Warning: no fake-node file found for time step {time_step} ({attack_type}/{attack_method})")
        return mapping

    def _compose_ldp_seed(self, attack_method: str, time_step: int, context: str) -> int:
        

        _ = (attack_method, time_step, context)  
        return int(self.ldp_seed)

    def apply_ldp_noise_to_degrees(self, graph: nx.Graph,
                                   degree_values: Dict[int, float],
                                   node_ids: Optional[List[int]] = None,
                                   seed: Optional[int] = None) -> Dict[int, float]:
        

        if graph.number_of_nodes() <= 1:
            return degree_values
        if node_ids is None:
            node_ids = list(graph.nodes())
        elif not node_ids:
            return degree_values
        rng_seed = seed if seed is not None else self.ldp_seed
        rng = np.random.default_rng(rng_seed)
        updated_values = degree_values.copy()
        scale = 1.0 / max(self.ldp_epsilon, 1e-9)

        for node_id in sorted(set(node_ids)):
            try:
                node_key = int(node_id)
            except (TypeError, ValueError):
                node_key = node_id

            if not graph.has_node(node_key):
                continue

            true_degree = graph.degree(node_key)
            noisy_degree = true_degree + rng.laplace(0.0, scale)
            updated_values[node_key] = max(0.0, float(noisy_degree))

        return self.ensure_non_negative_metrics(updated_values)
    
    def load_target_nodes_from_file(self, attack_type: str) -> List[int]:
        

        target_nodes = []
        
        
        dataset_prefix = self.dataset_tag or ""
        if attack_type in ('DC', 'DC_HMG'):
            file_path = os.path.join("target_nodes_record", "DC", dataset_prefix,
                                     f"{dataset_prefix}_target_nodes_original_ids.txt")
        elif attack_type == 'LCC':
            file_path = os.path.join("target_nodes_record", "LCC", dataset_prefix,
                                     f"{dataset_prefix}_target_nodes_original_ids.txt")
        else:
            print(f"Warning: unsupported attack type {attack_type}")
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
            print(f"Warning: target-node file {file_path} does not exist")
        except Exception as e:
            print(f"Failed to read target-node file {file_path}: {str(e)}")
        
        print(f"From {file_path}, loaded {len(target_nodes)} target nodes")
        return target_nodes
    
    def identify_target_nodes(self, original_graph: nx.Graph, attacked_graph: nx.Graph, 
                             attack_type: str = None, top_k: int = 10) -> List[int]:
        

        if not attack_type:
            raise ValueError("attack_type must be provided to read the target-node file.")
        
        return self.load_target_nodes_from_file(attack_type)

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
    
    def compute_target_metric_diffs(self, original_graph: nx.Graph, attacked_graph: nx.Graph, target_nodes: List[int]) -> Dict[str, float]:
        

        diffs = {
            'clustering_gain': 0.0,
            'degree_centrality_gain': 0.0,
            'two_star_gain': 0.0,
            'k_star_gain': 0.0,
            'triangle_count_gain': 0.0,
            'shortest_path_gain': 0.0
        }
        if not target_nodes:
            return diffs

        orig_triangles = nx.triangles(original_graph)
        atk_triangles = nx.triangles(attacked_graph)

        for node in target_nodes:
            orig_clust = self.calculate_clustering_coefficient(original_graph, node)
            atk_clust = self.calculate_clustering_coefficient(attacked_graph, node)
            diffs['clustering_gain'] += abs(atk_clust - orig_clust)

            orig_deg_cent = self.calculate_degree_centrality(original_graph, node)
            atk_deg_cent = self.calculate_degree_centrality(attacked_graph, node)
            diffs['degree_centrality_gain'] += abs(atk_deg_cent - orig_deg_cent)

            orig_deg = original_graph.degree(node) if node in original_graph else 0
            atk_deg = attacked_graph.degree(node) if node in attacked_graph else 0
            diffs['two_star_gain'] += abs((math.comb(orig_deg, 2) if orig_deg >= 2 else 0) -
                                          (math.comb(atk_deg, 2) if atk_deg >= 2 else 0))
            diffs['k_star_gain'] += abs((math.comb(orig_deg, self.k_star_size) if orig_deg >= self.k_star_size else 0) -
                                        (math.comb(atk_deg, self.k_star_size) if atk_deg >= self.k_star_size else 0))

            diffs['triangle_count_gain'] += abs(float(atk_triangles.get(node, 0)) - float(orig_triangles.get(node, 0)))

            orig_sp = self.compute_average_shortest_path_length(original_graph, node)
            atk_sp = self.compute_average_shortest_path_length(attacked_graph, node)
            diffs['shortest_path_gain'] += abs(atk_sp - orig_sp)

        return diffs
    
    def collect_target_metrics_record(self,
                                      attack_type: str,
                                      attack_method: str,
                                      time_step: int,
                                      original_graph: nx.Graph,
                                      attacked_graph: nx.Graph,
                                      target_nodes: List[int]) -> Dict[str, float]:
        

        diffs = self.compute_target_metric_diffs(original_graph, attacked_graph, target_nodes)

        record = {
            'attack_type': attack_type,
            'attack_method': attack_method,
            'time_step': time_step
        }
        
        for key, val in diffs.items():
            record[key] = val
        
        return record
    
    def save_target_metrics_summary(self, output_path: str) -> None:
        

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fieldnames = [
            'attack_type',
            'attack_method',
            'time_steps',
            'clustering_gain_weighted',
            'degree_centrality_gain_weighted',
            'two_star_gain_weighted',
            'k_star_gain_weighted',
            'triangle_count_gain_weighted',
            'shortest_path_gain_weighted'
        ]

        if not self.target_metrics_records:
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            print(f"No target-node metrics yet; created empty file: {output_path}")
            return

        scenario_records = defaultdict(list)
        for record in self.target_metrics_records:
            key = (
                record.get('attack_type'),
                record.get('attack_method')
            )
            scenario_records[key].append(record)

        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for (attack_type, attack_method), records in scenario_records.items():
                if not records:
                    continue

                records = sorted(records, key=lambda r: r.get('time_step', 0))
                time_steps_count = len(records)
                weights = [self.decay_factor ** (time_steps_count - 1 - idx) for idx in range(time_steps_count)]
                total_weight = sum(weights)
                normalized = [w / total_weight for w in weights] if total_weight > 0 else [0.0] * time_steps_count

                def weighted_gain(field: str) -> float:
                    return float(sum(abs(rec[field]) * normalized[idx] for idx, rec in enumerate(records)))

                writer.writerow({
                    'attack_type': attack_type,
                    'attack_method': attack_method,
                    'time_steps': time_steps_count,
                    'clustering_gain_weighted': weighted_gain('clustering_gain'),
                    'degree_centrality_gain_weighted': weighted_gain('degree_centrality_gain'),
                    'two_star_gain_weighted': weighted_gain('two_star_gain'),
                    'k_star_gain_weighted': weighted_gain('k_star_gain'),
                    'triangle_count_gain_weighted': weighted_gain('triangle_count_gain'),
                    'shortest_path_gain_weighted': weighted_gain('shortest_path_gain')
                })

        print(f"Target-node metrics summary saved to: {output_path}")

    def save_target_metrics_timeseries(self, output_path: str) -> None:
        

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fieldnames = [
            'attack_type',
            'attack_method',
            'time_step',
            'clustering_gain',
            'degree_centrality_gain',
            'two_star_gain',
            'k_star_gain',
            'triangle_count_gain',
            'shortest_path_gain'
        ]

        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in sorted(self.target_metrics_timeseries, key=lambda r: (r['attack_type'], r['attack_method'], r['time_step'])):
                writer.writerow({
                    'attack_type': rec.get('attack_type'),
                    'attack_method': rec.get('attack_method'),
                    'time_step': rec.get('time_step'),
                    'clustering_gain': rec.get('clustering_gain', 0.0),
                    'degree_centrality_gain': rec.get('degree_centrality_gain', 0.0),
                    'two_star_gain': rec.get('two_star_gain', 0.0),
                    'k_star_gain': rec.get('k_star_gain', 0.0),
                    'triangle_count_gain': rec.get('triangle_count_gain', 0.0),
                    'shortest_path_gain': rec.get('shortest_path_gain', 0.0)
                })

        print(f"Target-node metrics time series saved to: {output_path}")

    def load_time_series_graphs(self, directory: str, prefix: Optional[str] = None) -> Dict[int, nx.Graph]:
        

        graphs = {}
        normalized_prefix = prefix or ""
        metadata_key = (os.path.abspath(directory), normalized_prefix)
        self.graph_metadata[metadata_key] = {}

        if not os.path.exists(directory):
            print(f"Warning: directory {directory} does not exist")
            return graphs

        
        files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        if normalized_prefix:
            files = [f for f in files if f.startswith(normalized_prefix)]

        fallback_index = 0
        for file in sorted(files):
            
            try:
                time_step = None
                
                if '_t' in file:
                    
                    time_step = int(file.split('_t')[1].replace('.txt', ''))
                elif file.startswith('timeslice_'):
                    
                    time_step = int(file.replace('timeslice_', '').replace('.txt', ''))
                elif 'TimeStep' in file:
                    
                    time_step = int(file.split('TimeStep')[1].replace('.txt', ''))
                if time_step is None:
                    time_step = fallback_index
                    fallback_index += 1
                
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
    
    def _load_original_dc_graphs(self, directory: str) -> Dict[int, nx.Graph]:
        

        if self.dataset_tag in ("email", "reply"):
            file_map = {
                "email": os.path.join(directory, "email.txt"),
                "reply": os.path.join(directory, "reply.txt"),
            }
            reader_map = {
                "email": utils.read_email,
                "reply": utils.read_reply,
            }
            file_path = file_map.get(self.dataset_tag)
            reader = reader_map.get(self.dataset_tag)
            if not file_path or not reader or not os.path.exists(file_path):
                print(f"Warning: no {self.dataset_tag} data file found: {file_path}")
                return {}
            graphs = reader(file_path)
            return {idx: g for idx, g in enumerate(graphs)}
        
        return self.load_time_series_graphs(directory)

    def process_attack_scenario(self, original_dir: str, attacked_dir: str, 
                              attack_type: str, attack_method: str, 
                              metric_type: str,
                              original_prefix: Optional[str] = None,
                              attacked_prefix: Optional[str] = None,
                              collect_target_metrics: bool = False) -> List[Dict[str, float]]:
        

        print(f"Processing attack scenario: {attack_type}/{attack_method}, dataset: {self.dataset_tag or 'default'}, metric: {metric_type}")
        
        
        if original_prefix is None:
            if attack_type == 'DC':
                original_prefix = None
            else:
                
                original_prefix = None  
        if attacked_prefix is None:
            if attack_type == 'DC':
                attacked_prefix = f"AttackedGraph_{attack_method}_{attack_type}"
            else:
                attacked_prefix = f"Merged_{attack_method}_{attack_type}"
        
        
        if attack_type == 'DC':
            original_graphs = self._load_original_dc_graphs(original_dir)
        else:
            original_graphs = self.load_time_series_graphs(original_dir, original_prefix)
        
        
        attacked_graphs = self.load_time_series_graphs(attacked_dir, attacked_prefix)
        
        if not original_graphs:
            print(f"Warning: original graph data not found at {original_dir}")
            return []
        
        if not attacked_graphs:
            print(f"Warning: attacked graph data not found at {attacked_dir}")
            return []
        
        print(f"Found {len(original_graphs)} original time steps and {len(attacked_graphs)} attacked time steps")
        
        
        gains = []
        
        
        common_time_steps = set(original_graphs.keys()) & set(attacked_graphs.keys())
        
        for time_step in sorted(common_time_steps):
            original_graph = original_graphs[time_step]
            attacked_graph = attacked_graphs[time_step]
            
            
            target_nodes = self.identify_target_nodes(original_graph, attacked_graph, attack_type)
            
            
            if attack_type == 'DC' and metric_type == 'degree_centrality':
                original_degree_values = self.calculate_degree_values(original_graph)
                original_seed = self._compose_ldp_seed(attack_method, time_step, 'original')
                original_degree_values = self.apply_ldp_noise_to_degrees(
                    original_graph,
                    original_degree_values,
                    list(original_graph.nodes()),
                    seed=original_seed
                )
                original_metrics = self.normalize_degree_values(
                    original_degree_values,
                    original_graph.number_of_nodes()
                )

                attacked_degree_values = self.calculate_degree_values(attacked_graph)
                attacked_seed = self._compose_ldp_seed(attack_method, time_step, 'attacked')
                attacked_degree_values = self.apply_ldp_noise_to_degrees(
                    attacked_graph,
                    attacked_degree_values,
                    list(attacked_graph.nodes()),
                    seed=attacked_seed
                )
                attacked_metrics = self.normalize_degree_values(
                    attacked_degree_values,
                    attacked_graph.number_of_nodes()
                )
            else:
                original_metrics = self.calculate_graph_metrics(original_graph, metric_type)
                attacked_metrics = self.calculate_graph_metrics(attacked_graph, metric_type)
            
            
            total_gain, target_gain = self.calculate_time_slice_gain(original_metrics, attacked_metrics, target_nodes)
            gains.append({
                'time_step': time_step,
                'total_gain': total_gain,
                'target_gain': target_gain
            })
            
            if collect_target_metrics:
                key = (attack_type, attack_method, time_step)
                if key not in self.target_metrics_seen:
                    metrics_record = self.collect_target_metrics_record(
                        attack_type=attack_type,
                        attack_method=attack_method,
                        time_step=time_step,
                        original_graph=original_graph,
                        attacked_graph=attacked_graph,
                        target_nodes=target_nodes
                    )
                    self.target_metrics_seen.add(key)
                    self.target_metrics_records.append(metrics_record)
                    self.target_metrics_timeseries.append(metrics_record.copy())
            
            print(f"Time step {time_step}: total-graph attack gain = {total_gain:.6f}, target-node attack gain = {target_gain:.6f}")
        
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

        return gains
    
    def save_results_to_csv(self, results: List[Dict[str, float]], 
                           output_path: str, attack_type: str, 
                           attack_method: str, metric_type: str) -> None:
        

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            
            writer.writerow(['# Attack Gain Calculation Results'])
            writer.writerow([f'# Attack type: {attack_type}'])
            writer.writerow([f'# Attack method: {attack_method}'])
            writer.writerow([f'# Metric type: {metric_type}'])
            writer.writerow(['# Time Step', 'Total-Graph Attack Gain', 'Target-Node Attack Gain'])
            
            
            writer.writerow(['TimeStep', 'TotalGraphGain', 'TargetNodesGain', 'DecayWeightPercent', 'WeightedTotalGain', 'WeightedTargetGain'])
            for record in results:
                time_step = record['time_step']
                total_gain = record['total_gain']
                target_gain = record['target_gain']
                weight_percent = record.get('decay_weight_percent', 0.0)
                weighted_total = record.get('weighted_total_gain', 0.0)
                weighted_target = record.get('weighted_target_gain', 0.0)
                writer.writerow([
                    time_step,
                    f'{total_gain:.6f}',
                    f'{target_gain:.6f}',
                    f'{weight_percent:.6f}',
                    f'{weighted_total:.6f}',
                    f'{weighted_target:.6f}'
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
                              output_base_dir: str = "./attack_gain_results/") -> None:
        

        if dataset_tag is None:
            dataset_tag = self._choose_dataset_tag()
        if not dataset_tag:
            print("No available dataset found; exiting.")
            return
        self.dataset_tag = dataset_tag

        dc_original_dir = os.path.join("./dataset", dataset_tag)
        lcc_original_dir = os.path.join("./reconstructed_graph", f"{dataset_tag}_HMG")
        dc_attacked_base_dir = os.path.join("./output_py", dataset_tag, "DC")
        lcc_attacked_base_dir = os.path.join("./HMG_FakeNode_Merged", dataset_tag)

        print("Attack gain calculator starting...")
        self.target_metrics_records = []
        self.target_metrics_seen = set()
        self.target_metrics_timeseries = []
        
        attack_types = ['DC', 'LCC', 'DC_HMG']
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
                    
                    if attack_type == 'DC':
                        original_dir = dc_original_dir
                        attacked_dir = os.path.join(dc_attacked_base_dir, attack_method)
                        original_prefix = None
                        attacked_prefix = f"AttackedGraph_{attack_method}_{attack_type}"
                    elif attack_type == 'LCC':
                        original_dir = lcc_original_dir
                        attacked_dir = os.path.join(lcc_attacked_base_dir, attack_type, f"{attack_method}_Merged")
                        original_prefix = None  
                        attacked_prefix = f"Merged_{attack_method}_{attack_type}"
                    else:
                        original_dir = lcc_original_dir
                        attacked_dir = os.path.join(lcc_attacked_base_dir, 'DC', f"{attack_method}_Merged")
                        original_prefix = None
                        attacked_prefix = f"Merged_{attack_method}_DC"
                    
                    if not os.path.exists(original_dir):
                        print(f"Warning: original directory does not exist: {original_dir}")
                        continue
                    
                    if not os.path.exists(attacked_dir):
                        print(f"Warning: attacked directory does not exist: {attacked_dir}")
                        continue
                    
                    try:
                        collect_metrics = True
                        results = self.process_attack_scenario(
                            original_dir=original_dir,
                            attacked_dir=attacked_dir,
                            attack_type=attack_type,
                            attack_method=attack_method,
                            metric_type=metric_type,
                            original_prefix=original_prefix,
                            attacked_prefix=attacked_prefix,
                            collect_target_metrics=collect_metrics
                        )
                        
                        if results:
                            output_dir = os.path.join(output_base_dir, attack_type)
                            os.makedirs(output_dir, exist_ok=True)
                            filename = f"{attack_method}_{metric_type}_gain.csv"
                            output_path = os.path.join(output_dir, filename)
                            self.save_results_to_csv(
                                results=results,
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
        target_metrics_timeseries_path = os.path.join(output_base_dir, 'target_metrics_timeseries.csv')
        self.save_target_metrics_timeseries(target_metrics_timeseries_path)
        print("\nTarget metrics summary generated.")

    def calculate_weighted_target_gain(self, target_gains: list, decay_factor: float = 0.6) -> float:
        

        if not target_gains:
            return 0.0
        
        total_weighted_gain = 0.0
        total_weight = 0.0
        
        
        for i, gain in enumerate(target_gains):
            
            
            weight = decay_factor ** (len(target_gains) - 1 - i)
            total_weighted_gain += gain * weight
            total_weight += weight
        
        
        return total_weighted_gain / total_weight if total_weight > 0 else 0.0
    
    def generate_summary_report(self, output_base_dir: str = "./attack_gain_results/") -> None:
        

        print("Generating summary report...")
        
        summary_data = []
        
        
        for attack_type in ['DC', 'LCC']:
            attack_dir = os.path.join(output_base_dir, attack_type)
            if not os.path.exists(attack_dir):
                continue
            
            for file in os.listdir(attack_dir):
                if file.endswith('_gain.csv'):
                    file_path = os.path.join(attack_dir, file)
                    
                    
                    parts = file.replace('_gain.csv', '').split('_')
                    if len(parts) >= 2:
                        attack_method = parts[0]
                        metric_type = '_'.join(parts[1:])
                        
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8-sig') as f:
                                reader = csv.reader(f)
                                header = None
                                header_index = {}
                                total_gains = []
                                target_gains = []
                                weighted_total_gains = []
                                weighted_target_gains = []

                                for row in reader:
                                    if not row:
                                        continue
                                    if row[0].startswith('#'):
                                        continue
                                    if header is None:
                                        header = row
                                        header_index = {name: idx for idx, name in enumerate(row)}
                                        continue

                                    try:
                                        total_gain = float(row[header_index.get('TotalGraphGain', 1)])
                                        target_gain = float(row[header_index.get('TargetNodesGain', 2)])
                                        total_gains.append(total_gain)
                                        target_gains.append(target_gain)

                                        if 'WeightedTotalGain' in header_index:
                                            weighted_total_gains.append(float(row[header_index['WeightedTotalGain']]))
                                        if 'WeightedTargetGain' in header_index:
                                            weighted_target_gains.append(float(row[header_index['WeightedTargetGain']]))
                                    except (ValueError, KeyError, IndexError):
                                        continue

                                if total_gains and target_gains:
                                    weighted_target_gain = self.calculate_weighted_target_gain(target_gains)
                                    summary_entry = {
                                        'attack_type': attack_type,
                                        'attack_method': attack_method,
                                        'metric_type': metric_type,
                                        'total_gain_sum': sum(total_gains),
                                        'total_gain_avg': np.mean(total_gains),
                                        'total_gain_max': max(total_gains),
                                        'total_gain_min': min(total_gains),
                                        'target_gain_sum': sum(target_gains),
                                        'target_gain_avg': np.mean(target_gains),
                                        'target_gain_max': max(target_gains),
                                        'target_gain_min': min(target_gains),
                                        'target_gain_weighted': weighted_target_gain,
                                        'time_steps': len(total_gains)
                                    }

                                    summary_entry['total_gain_decay_weighted'] = sum(weighted_total_gains) if weighted_total_gains else 0.0
                                    summary_entry['target_gain_decay_weighted'] = sum(weighted_target_gains) if weighted_target_gains else 0.0

                                    summary_data.append(summary_entry)

                        except Exception as e:
                            print(f"Failed to read file {file_path}: {str(e)}")
        
        
        summary_path = os.path.join(output_base_dir, 'summary_report.csv')
        with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
            if summary_data:
                writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
                writer.writeheader()
                writer.writerows(summary_data)
        
        print(f"Summary report saved to: {summary_path}")

def main():
    

    print("Attack gain calculator starting...")
    
    
    calculator = AttackGainCalculator()
    
    
    calculator.process_all_scenarios()
    
    
    calculator.generate_summary_report()
    
    print("Attack gain calculation completed")

if __name__ == "__main__":
    main()
