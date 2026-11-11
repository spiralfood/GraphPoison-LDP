import os
import sys
from typing import List
import random
import numpy as np


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import main
import utils


BASE_SEED = 2025
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)


DEFAULT_EPS = 4.0
DEFAULT_EPS1_PART = 0.3
DEFAULT_EPS2_PART = 0.1
DEFAULT_EPS3_PART = 0.6
DEFAULT_W = 5
DEFAULT_DELTA = 1

def run_hmg_for_datasets(
    data_dir: str = "./dataset/",
    output_dir: str = "./reconstructed_graph/",
    eps: float = DEFAULT_EPS,
    eps1_part: float = DEFAULT_EPS1_PART,
    eps2_part: float = DEFAULT_EPS2_PART,
    eps3_part: float = DEFAULT_EPS3_PART,
    w: int = DEFAULT_W,
    delta: int = DEFAULT_DELTA,
) -> None:
    

    if not os.path.exists(data_dir):
        print(f"Error: data directory {data_dir} does not exist")
        return
    
    
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    
    if not subdirs:
        print(f"Error: no datasets found under data directory {data_dir}")
        return
    
    
    dataset_paths = [os.path.join(data_dir, subdir) for subdir in subdirs]
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Creating output directory: {output_dir}")
    
    print(f"Found {len(dataset_paths)} datasets:")
    for path in dataset_paths:
        print(f"  - {path}")
    
    
    try:
        main.generic_hmg_pipeline(
            dataset_paths=dataset_paths,  
            output_base_path=output_dir,
            eps=eps,
            eps1_part=eps1_part,
            eps2_part=eps2_part,
            eps3_part=eps3_part,
            w=w,
            delta=delta
        )
        print("HMG perturbation pipeline completed")
    except Exception as e:
        print(f"Error while running the HMG perturbation pipeline: {str(e)}")


def run_hmg_for_email_or_reply(
    selection: str,
    output_base_dir: str = "./reconstructed_graph/",
    eps: float = DEFAULT_EPS,
    eps1_part: float = DEFAULT_EPS1_PART,
    eps2_part: float = DEFAULT_EPS2_PART,
    eps3_part: float = DEFAULT_EPS3_PART,
    w: int = DEFAULT_W,
    delta: int = DEFAULT_DELTA,
) -> None:
    

    dataset_options = {
        "1": {
            "name": "email",
            "path": os.path.join("dataset", "email", "email.txt"),
            "reader": utils.read_email,
        },
        "2": {
            "name": "reply",
            "path": os.path.join("dataset", "reply", "reply.txt"),
            "reader": utils.read_reply,
        },
    }

    if selection not in dataset_options:
        print("Invalid selection. Enter 1 (email) or 2 (reply).")
        return

    option = dataset_options[selection]
    dataset_name = option["name"]
    data_path = option["path"]
    reader = option["reader"]

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    print(f"Loading the {dataset_name} dataset: {data_path}")
    data = reader(data_path)

    print("Starting the HMG perturbation pipeline")
    User = main.Collect_info(data, eps, eps1_part, eps2_part, eps3_part)
    main.Extract_info(User, eps, eps2_part, eps3_part, w, delta)

    
    output_dir = os.path.join(output_base_dir, f"{dataset_name}_HMG")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Creating output directory: {output_dir}")

    if hasattr(User, "final_hmg_graphs"):
        print(f"Saving HMG reconstructed results, number of time steps: {len(User.final_hmg_graphs)}")
        for t, graph in User.final_hmg_graphs.items():
            filename = f"{dataset_name}_final_hmg_t{t}.txt"
            filepath = os.path.join(output_dir, filename)

            nodes_with_edges = set()
            for edge in graph.edges():
                nodes_with_edges.add(edge[0])
                nodes_with_edges.add(edge[1])
            nodes = len(nodes_with_edges)
            edges = graph.number_of_edges()

            with open(filepath, "w") as f:
                f.write(f"# Undirected graph (each pair of nodes is saved twice):{filename}\n")
                f.write(f"# {dataset_name} dataset - HMG reconstructed\n")
                f.write(f"# Nodes:{nodes}\tEdges: {edges}\n")
                f.write("# FromNodeId\tToNodeId\n")
                for edge in graph.edges():
                    f.write(f"{edge[0]}\t{edge[1]}\n")
            print(f"Saved: {filepath}")
        print("Saving complete")
    else:
        print("No HMG reconstructed results found; the pipeline may have exited early.")


def run_hmg_for_as733(
    output_base_dir: str = "./reconstructed_graph/",
    eps: float = DEFAULT_EPS,
    eps1_part: float = DEFAULT_EPS1_PART,
    eps2_part: float = DEFAULT_EPS2_PART,
    eps3_part: float = DEFAULT_EPS3_PART,
    w: int = DEFAULT_W,
    delta: int = DEFAULT_DELTA,
) -> None:
    

    dataset_path = os.path.join("dataset", "as-733")
    if not os.path.exists(dataset_path):
        print(f"AS-733 dataset directory not found: {dataset_path}")
        return

    print(f"Processing the AS-733 dataset: {dataset_path}")
    
    main.generic_hmg_pipeline(
        dataset_paths=[dataset_path],
        output_base_path=output_base_dir,
        eps=eps,
        eps1_part=eps1_part,
        eps2_part=eps2_part,
        eps3_part=eps3_part,
        w=w,
        delta=delta,
    )


def run_hmg_for_synthetic(
    output_base_dir: str = "./reconstructed_graph/",
    eps: float = DEFAULT_EPS,
    eps1_part: float = DEFAULT_EPS1_PART,
    eps2_part: float = DEFAULT_EPS2_PART,
    eps3_part: float = DEFAULT_EPS3_PART,
    w: int = DEFAULT_W,
    delta: int = DEFAULT_DELTA,
) -> None:
    

    dataset_path = os.path.join("dataset", "Synthetic")
    if not os.path.exists(dataset_path):
        print(f"Synthetic dataset directory not found: {dataset_path}")
        return

    print(f"Processing the Synthetic dataset: {dataset_path}")
    main.generic_hmg_pipeline(
        dataset_paths=[dataset_path],
        output_base_path=output_base_dir,
        eps=eps,
        eps1_part=eps1_part,
        eps2_part=eps2_part,
        eps3_part=eps3_part,
        w=w,
        delta=delta,
    )


if __name__ == "__main__":
    print("Select the HMG pipeline to run:")
    print("1: email dataset (dataset/email/email.txt)")
    print("2: reply dataset (dataset/reply/reply.txt)")
    print("3: AS-733 dataset (dataset/as-733)")
    print("4: Synthetic dataset (dataset/Synthetic)")
    print("5: Batch process all subdirectories under dataset/ (default)")
    choice = input("Enter a number: ").strip()

    if choice in ("1", "2"):
        run_hmg_for_email_or_reply(selection=choice)
    elif choice == "3":
        run_hmg_for_as733()
    elif choice == "4":
        run_hmg_for_synthetic()
    else:
        run_hmg_for_datasets()
