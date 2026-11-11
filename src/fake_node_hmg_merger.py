import os
import sys
from typing import List, Dict, Set, Tuple
import networkx as nx
import utils


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FakeNodeHMGMerger:
    

    def __init__(self, 
                 fake_node_root: str = "./FakeNodeID/",
                 hmg_attack_root: str = "./HMG_attack/",
                 output_root: str = "./HMG_FakeNode_Merged/",
                 dataset_tag: str = None):
        

        self.fake_node_root = fake_node_root
        self.hmg_attack_root = hmg_attack_root
        self.output_root = output_root

        
        self.dataset_tag = dataset_tag or self._choose_dataset()
        if not self.dataset_tag:
            raise ValueError("No valid dataset selected; stopping processing.")

        
        self.fake_node_dir = os.path.join(self.fake_node_root, self.dataset_tag)
        self.hmg_attack_dir = os.path.join(self.hmg_attack_root, self.dataset_tag)
        self.output_dir = os.path.join(self.output_root, self.dataset_tag)
        
        
        os.makedirs(self.output_dir, exist_ok=True)

    def _choose_dataset(self) -> str:
        

        def list_dirs(root):
            if not os.path.isdir(root):
                return set()
            return {d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}

        fake_sets = list_dirs(self.fake_node_root)
        hmg_sets = list_dirs(self.hmg_attack_root)
        candidates = sorted(fake_sets & hmg_sets)

        if not candidates:
            print(f"No intersecting dataset directory found in {self.fake_node_root} and {self.hmg_attack_root}.")
            return None

        print("\n=== Select the dataset directories to merge ===")
        for idx, name in enumerate(candidates, 1):
            print(f"{idx} - {name}")

        choice = input(f"Enter a number (1-{len(candidates)}; press Enter for default 1): ").strip()
        try:
            sel = int(choice) if choice else 1
            if sel < 1 or sel > len(candidates):
                raise ValueError
        except ValueError:
            sel = 1
            print(f"Invalid input; selecting {candidates[0]} by default")

        return candidates[sel - 1]
    
    def read_fake_node_info(self, file_path: str) -> Tuple[Dict[int, int], Set[Tuple[int, int]]]:
        

        fake_node_mapping = {}  
        fake_edges = set()
        
        if not os.path.exists(file_path):
            print(f"Warning: fake-node file does not exist: {file_path}")
            return fake_node_mapping, fake_edges
        
        in_edges_section = False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                
                if not line:
                    continue
                
                
                if line.startswith('#'):
                    
                    if "# Fake Node:" in line and "(Original ID:" in line:
                        try:
                            
                            parts = line.split()
                            fake_id = int(parts[3])  
                            
                            original_id_str = line.split("(Original ID:")[1].strip().rstrip(')')
                            original_id = int(original_id_str)
                            fake_node_mapping[fake_id] = original_id
                        except (ValueError, IndexError) as e:
                            print(f"Warning: failed to parse fake-node information: {line}, error: {e}")
                            continue
                    
                    
                    elif "# New Fake Edges" in line:
                        in_edges_section = True
                    
                    continue
                
                
                if in_edges_section:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            u, v = int(parts[0]), int(parts[1])
                            fake_edges.add((u, v))
                            fake_edges.add((v, u))  
                        except ValueError:
                            print(f"Warning: failed to parse edge information: {line}")
                            continue
        
        print(f"Successfully read fake-node file: {file_path}")
        print(f"  - Fake-node mappings: {len(fake_node_mapping)} entries")
        print(f"  - Fake edges: {len(fake_edges)//2} (stored bidirectionally)")
        
        return fake_node_mapping, fake_edges

    def load_original_graph(self, time_step: int) -> nx.Graph:
        

        graph = nx.Graph()
        tag = str(self.dataset_tag)

        try:
            if tag == "as-733":
                dataset_dir = os.path.join("dataset", "as-733")
                if not os.path.isdir(dataset_dir):
                    print(f"Warning: original dataset directory does not exist: {dataset_dir}")
                    return graph
                files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".txt")])
                if time_step < 0 or time_step >= len(files):
                    print(f"Warning: time step {time_step} is outside the as-733 file-count range ({len(files)})")
                    return graph
                file_path = os.path.join(dataset_dir, files[time_step])
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                u, v = int(parts[0]), int(parts[1])
                                graph.add_edge(u, v)
                            except ValueError:
                                continue
            elif tag in ("email", "reply"):
                reader = utils.read_email if tag == "email" else utils.read_reply
                data_path = os.path.join("dataset", tag, f"{tag}.txt")
                if not os.path.exists(data_path):
                    print(f"Warning: original data file does not exist: {data_path}")
                    return graph
                graphs = reader(data_path)
                if time_step < 0 or time_step >= len(graphs):
                    print(f"Warning: time step {time_step} exceeds the available slice range for {tag} ({len(graphs)})")
                    return graph
                graph = graphs[time_step]
            else:
                print(f"Warning: unknown dataset tag {tag}; cannot load the original graph")
        except Exception as exc:
            print(f"Warning: failed to load original graph (dataset {tag}, time step {time_step}): {exc}")

        return graph

    def load_reconstructed_graph(self, time_step: int) -> nx.Graph:
        

        graph = nx.Graph()
        tag = str(self.dataset_tag)
        base_dir = os.path.join("reconstructed_graph", f"{tag}_HMG")
        file_name = f"{tag}_final_hmg_t{time_step}.txt"
        file_path = os.path.join(base_dir, file_name)

        if not os.path.exists(file_path):
            print(f"Warning: reconstructed-graph file does not exist: {file_path}")
            return graph

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            u, v = int(parts[0]), int(parts[1])
                            graph.add_edge(u, v)
                        except ValueError:
                            continue
        except Exception as exc:
            print(f"Warning: failed to load reconstructed graph ({file_path}): {exc}")

        return graph
    

    def read_hmg_graph(self, file_path: str) -> nx.Graph:
        

        graph = nx.Graph()
        
        if not os.path.exists(file_path):
            print(f"Warning: HMG attacked-graph file does not exist: {file_path}")
            return graph
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                
                if line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        graph.add_edge(u, v)
                    except ValueError:
                        continue
        
        return graph

    def _collect_fake_nodes(self, fake_node_mapping: Dict[int, int], fake_edges: Set[Tuple[int, int]]) -> Set[int]:
        

        _ = fake_edges
        return set(fake_node_mapping.keys())

    def _remove_fake_incident_edges(self, graph: nx.Graph, fake_nodes: Set[int]) -> None:
        
        for fake_node in fake_nodes:
            if graph.has_node(fake_node):
                incident = list(graph.edges(fake_node))
                if incident:
                    graph.remove_edges_from(incident)

    def _add_edges_from_graph(self, target: nx.Graph, source: nx.Graph, fake_nodes: Set[int]) -> None:
        

        if source is None:
            return
        for fake_node in fake_nodes:
            if not source.has_node(fake_node):
                continue
            if not target.has_node(fake_node):
                target.add_node(fake_node)
            for nbr in source.neighbors(fake_node):
                if not target.has_node(nbr):
                    target.add_node(nbr)
                target.add_edge(fake_node, nbr)

    def _add_fake_edges(self, target: nx.Graph, fake_edges: Set[Tuple[int, int]]) -> None:
        
        for u, v in fake_edges:
            if not target.has_node(u):
                target.add_node(u)
            if not target.has_node(v):
                target.add_node(v)
            target.add_edge(u, v)

    def merge_mga_rva(self,
                      fake_node_mapping: Dict[int, int],
                      fake_edges: Set[Tuple[int, int]],
                      reconstructed_graph: nx.Graph,
                      original_graph: nx.Graph) -> nx.Graph:
        

        merged_graph = reconstructed_graph.copy()
        fake_nodes = self._collect_fake_nodes(fake_node_mapping, fake_edges)

        
        self._remove_fake_incident_edges(merged_graph, fake_nodes)

        
        self._add_edges_from_graph(merged_graph, original_graph, fake_nodes)

        
        self._add_fake_edges(merged_graph, fake_edges)

        return merged_graph

    def merge_rna(self,
                  fake_node_mapping: Dict[int, int],
                  fake_edges: Set[Tuple[int, int]],
                  reconstructed_graph: nx.Graph,
                  hmg_graph: nx.Graph) -> nx.Graph:
        

        merged_graph = reconstructed_graph.copy()
        fake_nodes = self._collect_fake_nodes(fake_node_mapping, fake_edges)

        
        self._remove_fake_incident_edges(merged_graph, fake_nodes)

        
        self._add_edges_from_graph(merged_graph, hmg_graph, fake_nodes)

        return merged_graph

    def write_merged_graph(self, graph: nx.Graph, output_path: str, 
                          attack_type: str, attack_method: str, 
                          dataset: int, time_step: int):
        

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            
            f.write(f"# Undirected graph (each pair of nodes is saved twice):Merged_{attack_method}_{attack_type}_Dataset{dataset}_TimeStep{time_step}.txt\n")
            f.write(f"# HMG processed {attack_method} {attack_type} attack graph merged with fake nodes\n")
            f.write(f"# Nodes:{graph.number_of_nodes()}\tEdges: {graph.number_of_edges()}\n")
            f.write("# FromNodeId\tToNodeId\n")
            
            
            edges = list(graph.edges())
            edges.sort(key=lambda x: (x[0], x[1]))
            
            for u, v in edges:
                f.write(f"{u}\t{v}\n")
                f.write(f"{v}\t{u}\n")  
    
    def process_attack_type(self, attack_type: str, attack_method: str):
        

        print(f"Processing {attack_type} {attack_method} attack (single-time-step mode)...")
        
        fake_node_subdir = os.path.join(self.fake_node_dir, attack_type, attack_method)
        hmg_subdir = os.path.join(self.hmg_attack_dir, attack_type, f"{attack_method}_HMG")
        output_subdir = os.path.join(self.output_dir, attack_type, f"{attack_method}_Merged")

        if not os.path.exists(fake_node_subdir):
            print(f"Warning: fake-node directory does not exist: {fake_node_subdir}")
            return

        if attack_method == "RNA" and not os.path.exists(hmg_subdir):
            print(f"Warning: RNA attacked-graph noise directory does not exist: {hmg_subdir}")
            return
        
        
        os.makedirs(output_subdir, exist_ok=True)
        
        
        fake_files = [f for f in os.listdir(fake_node_subdir) if f.startswith('FakeNodeID_')]
        time_step_files = []
        
        for fake_file in fake_files:
            
            
            parts = fake_file.replace('.txt', '').split('_')
            if len(parts) >= 5:
                dataset_token = parts[3].replace('Dataset', '')
                time_step = int(parts[4].replace('TimeStep', ''))
                time_step_files.append((time_step, dataset_token, fake_file))
        
        
        time_step_files.sort(key=lambda x: x[0])
        
        
        for time_step, dataset_token, fake_file in time_step_files:
            print(f"\nProcessing time step {time_step} (single-time-step mode)...")

            
            reconstructed_graph = self.load_reconstructed_graph(time_step)
            if reconstructed_graph.number_of_nodes() == 0 and reconstructed_graph.number_of_edges() == 0:
                print(f"Warning: reconstructed graph is empty; skipping time step {time_step}")
                continue

            
            fake_path = os.path.join(fake_node_subdir, fake_file)
            fake_node_mapping, fake_edges = self.read_fake_node_info(fake_path)

            merged_graph = None

            if attack_method in ("MGA", "RVA"):
                
                original_graph = self.load_original_graph(time_step)
                merged_graph = self.merge_mga_rva(fake_node_mapping, fake_edges, reconstructed_graph, original_graph)
            elif attack_method == "RNA":
                
                hmg_file = f"HMG_{attack_method}_{attack_type}_Dataset{dataset_token}_TimeStep{time_step}.txt"
                hmg_path = os.path.join(hmg_subdir, hmg_file)
                
                if not os.path.exists(hmg_path) and not dataset_token.isdigit():
                    fallback_token = "0"
                    fallback_file = f"HMG_{attack_method}_{attack_type}_Dataset{fallback_token}_TimeStep{time_step}.txt"
                    fallback_path = os.path.join(hmg_subdir, fallback_file)
                    if os.path.exists(fallback_path):
                        hmg_file = fallback_file
                        hmg_path = fallback_path
                        dataset_token = fallback_token

                if not os.path.exists(hmg_path):
                    print(f"Warning: corresponding attacked-graph noise file does not exist: {hmg_path}")
                    continue

                hmg_graph = self.read_hmg_graph(hmg_path)
                merged_graph = self.merge_rna(fake_node_mapping, fake_edges, reconstructed_graph, hmg_graph)
            else:
                print(f"Warning: unknown attack method {attack_method}; skipping")
                continue

            if merged_graph is not None:
                output_path = os.path.join(output_subdir, f"Merged_{attack_method}_{attack_type}_Dataset{dataset_token}_TimeStep{time_step}.txt")
                self.write_merged_graph(merged_graph, output_path, attack_type, attack_method, dataset_token, time_step)
                print(f"Processed: Dataset{dataset_token}_TimeStep{time_step} - Nodes: {merged_graph.number_of_nodes()}, Edges: {merged_graph.number_of_edges()}")
    
    def process_all(self):
        

        print("Starting fake-node and HMG attacked-graph merge...")
        
        
        for attack_type in ['DC', 'LCC']:
            for attack_method in ['RVA', 'MGA', 'RNA']:
                self.process_attack_type(attack_type, attack_method)
        
        print("Fake-node and HMG attacked-graph merge completed!")
        print(f"Results saved to: {self.output_dir}")

def main():
    
    merger = FakeNodeHMGMerger()
    merger.process_all()

if __name__ == "__main__":
    main()
