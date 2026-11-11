import os
import re
from typing import Dict, List, Optional

def parse_hmg_file_header(file_path: str) -> Optional[Dict[str, int]]:
    

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                
                
                if line.startswith("# Nodes:") and "Edges:" in line:
                    
                    nodes_match = re.search(r'Nodes:(\d+)', line)
                    edges_match = re.search(r'Edges:\s*(\d+)', line)
                    
                    if nodes_match and edges_match:
                        nodes = int(nodes_match.group(1))
                        edges = int(edges_match.group(1))
                        return {'nodes': nodes, 'edges': edges}
        
        print(f"Warning: could not parse node and edge counts from file header: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {file_path}, error: {str(e)}")
        return None

def calculate_average_degree_from_header(nodes: int, edges: int) -> int:
    

    if nodes == 0:
        return 0
    
    
    avg_degree = (2 * edges) / nodes
    return max(1, int(avg_degree))  

def extract_time_step_from_filename(filename: str) -> Optional[int]:
    

    match = re.search(r'_t(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    return None

def process_dataset_average_degrees(dataset_hmg_dir: str) -> Dict[int, int]:
    

    time_step_degrees = {}
    
    if not os.path.exists(dataset_hmg_dir):
        print(f"Error: directory does not exist: {dataset_hmg_dir}")
        return time_step_degrees
    
    
    hmg_files = [f for f in os.listdir(dataset_hmg_dir) if f.endswith('.txt')]
    
    print(f"Processing dataset: {os.path.basename(dataset_hmg_dir)}")
    print(f"Found {len(hmg_files)} HMG files")
    
    for filename in hmg_files:
        file_path = os.path.join(dataset_hmg_dir, filename)
        
        
        time_step = extract_time_step_from_filename(filename)
        if time_step is None:
            print(f"Warning: could not extract the time step from filename: {filename}")
            continue
        
        
        header_info = parse_hmg_file_header(file_path)
        if header_info is None:
            print(f"Warning: could not parse file header: {filename}")
            continue
        
        
        avg_degree = calculate_average_degree_from_header(
            header_info['nodes'], header_info['edges']
        )
        
        time_step_degrees[time_step] = avg_degree
        print(f"  Time step {time_step}: nodes={header_info['nodes']}, edges={header_info['edges']}, average degree={avg_degree}")
    
    return time_step_degrees

def save_average_degrees_to_file(dataset_name: str, time_step_degrees: Dict[int, int], output_dir: str) -> None:
    

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Creating output directory: {output_dir}")
    
    
    output_file = os.path.join(output_dir, f"{dataset_name}_average_degrees.txt")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Average degree results for HMG-perturbed graphs\n")
            f.write(f"# Dataset: {dataset_name}\n")
            f.write(f"# Format: Time step\tAverage degree\n")
            f.write("#" + "="*50 + "\n")
            
            
            for time_step in sorted(time_step_degrees.keys()):
                avg_degree = time_step_degrees[time_step]
                f.write(f"{time_step}\t{avg_degree}\n")
        
        print(f"Average degree results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving file: {str(e)}")

def process_all_datasets(
    reconstructed_dir: str = "./reconstructed_graph/",
    output_dir: str = "./hmg_average_degrees/"
) -> None:
    

    print("=== Batch average-degree calculation for HMG-perturbed graphs ===")
    print(f"Input directory: {reconstructed_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    if not os.path.exists(reconstructed_dir):
        print(f"Error: input directory does not exist: {reconstructed_dir}")
        return
    
    
    dataset_dirs = [d for d in os.listdir(reconstructed_dir) 
                   if os.path.isdir(os.path.join(reconstructed_dir, d)) and d.endswith('_HMG')]
    
    if not dataset_dirs:
        print(f"Error: no _HMG directories found under {reconstructed_dir}")
        return
    
    print(f"Found {len(dataset_dirs)} datasets:")
    for dataset_dir in dataset_dirs:
        print(f"  - {dataset_dir}")
    print()
    
    
    for dataset_dir in dataset_dirs:
        dataset_hmg_path = os.path.join(reconstructed_dir, dataset_dir)
        dataset_name = dataset_dir.replace('_HMG', '')
        
        print(f"\n--- Processing dataset: {dataset_name} ---")
        
        
        time_step_degrees = process_dataset_average_degrees(dataset_hmg_path)
        
        if time_step_degrees:
            
            save_average_degrees_to_file(dataset_name, time_step_degrees, output_dir)
            print(f"Dataset {dataset_name} processed successfully; {len(time_step_degrees)} time steps")
        else:
            print(f"Failed to process dataset {dataset_name}: no valid data found")
    
    print(f"\n=== All datasets processed ===")
    print(f"Result files saved in: {output_dir}")

def process_single_dataset(
    dataset_name: str,
    reconstructed_dir: str = "./reconstructed_graph/",
    output_dir: str = "./hmg_average_degrees/"
) -> None:
    

    print(f"=== Processing a single dataset: {dataset_name} ===")
    
    dataset_hmg_path = os.path.join(reconstructed_dir, f"{dataset_name}_HMG")
    
    if not os.path.exists(dataset_hmg_path):
        print(f"Error: dataset directory does not exist: {dataset_hmg_path}")
        return
    
    
    time_step_degrees = process_dataset_average_degrees(dataset_hmg_path)
    
    if time_step_degrees:
        
        save_average_degrees_to_file(dataset_name, time_step_degrees, output_dir)
        print(f"Dataset {dataset_name} processed successfully; {len(time_step_degrees)} time steps")
        
        
        print("\nAverage degree by time step:")
        for time_step in sorted(time_step_degrees.keys()):
            print(f"  Time step {time_step}: average degree = {time_step_degrees[time_step]}")
    else:
        print(f"Failed to process dataset {dataset_name}: no valid data found")

if __name__ == "__main__":
    
    
    process_all_datasets()
