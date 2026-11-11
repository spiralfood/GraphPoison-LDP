import os
import sys
import subprocess
import shutil
import time
import re
from typing import List, Tuple

from allocation_strategy import get_strategy_name


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


SCENARIOS = [
    {
        "name": "strategy0_multi_eps",
        "allocation_strategy": 0,
        "beta_values": [0.001,0.005,0.01,0.1,0.2],
        "target_ratios": [0.05],
        "eps_values": [4.0],
        "ldp_eps_values": [4.0],
        
        "dataset_choice": "3",
        "dataset_tag": "as-733",
        
        "dataset_index": "1",   
    },
    {
        "name": "strategy0_multi_eps",
        "allocation_strategy": 0,
        "beta_values": [0.05],
        "target_ratios": [0.001,0.005,0.01,0.1,0.2],
        "eps_values": [4.0],
        "ldp_eps_values": [4.0],
        
        "dataset_choice": "3",
        "dataset_tag": "as-733",
        
        "dataset_index": "1",   
    },
    {
        "name": "strategy0_multi_eps",
        "allocation_strategy": 0,
        "beta_values": [0.05],
        "target_ratios": [0.05],
        "eps_values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "ldp_eps_values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        
        "dataset_choice": "3",
        "dataset_tag": "as-733",
        
        "dataset_index": "1",   
    },
    {
        "name": "strategy0_multi_eps",
        "allocation_strategy": 1,
        "beta_values": [0.001,0.005,0.01,0.1,0.2],
        "target_ratios": [0.05],
        "eps_values": [4.0],
        "ldp_eps_values": [4.0],
        
        "dataset_choice": "3",
        "dataset_tag": "as-733",
        
        "dataset_index": "1",   
    },
    {
        "name": "strategy0_multi_eps",
        "allocation_strategy": 1,
        "beta_values": [0.05],
        "target_ratios": [0.001,0.005,0.01,0.1,0.2],
        "eps_values": [4.0],
        "ldp_eps_values": [4.0],
        
        "dataset_choice": "3",
        "dataset_tag": "as-733",
        
        "dataset_index": "1",   
    },
    {
        "name": "strategy0_multi_eps",
        "allocation_strategy": 1,
        "beta_values": [0.05],
        "target_ratios": [0.05],
        "eps_values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "ldp_eps_values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        
        "dataset_choice": "3",
        "dataset_tag": "as-733",
        
        "dataset_index": "1",   
    },
    {
        "name": "strategy0_multi_eps",
        "allocation_strategy":2,
        "beta_values": [0.001,0.005,0.01,0.1,0.2],
        "target_ratios": [0.05],
        "eps_values": [4.0],
        "ldp_eps_values": [4.0],
        
        "dataset_choice": "3",
        "dataset_tag": "as-733",
        
        "dataset_index": "1",   
    },
    {
        "name": "strategy0_multi_eps",
        "allocation_strategy": 2,
        "beta_values": [0.05],
        "target_ratios": [0.001,0.005,0.01,0.1,0.2],
        "eps_values": [4.0],
        "ldp_eps_values": [4.0],
        
        "dataset_choice": "3",
        "dataset_tag": "as-733",
        
        "dataset_index": "1",   
    },
    {
        "name": "strategy0_multi_eps",
        "allocation_strategy": 2,
        "beta_values": [0.05],
        "target_ratios": [0.05],
        "eps_values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "ldp_eps_values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        
        "dataset_choice": "3",
        "dataset_tag": "as-733",
        
        "dataset_index": "1",   
    }
]


def count_combinations(beta_values: List[float], target_ratios: List[float], eps_values: List[float]) -> int:
    
    return len(beta_values) * len(target_ratios) * len(eps_values)


FILES_TO_MODIFY = {
    "attackcentre.py": {"beta": "DEFAULT_BETA", "target_ratio": "DEFAULT_TARGET_RATIO"},
    "attacklcc.py": {"beta": "DEFAULT_BETA", "target_ratio": "DEFAULT_TARGET_RATIO"},
    "run_hmg_pipeline.py": {"eps": "DEFAULT_EPS"},
    "hmg_attack_processor.py": {"eps": "DEFAULT_EPS"},
    "attack_gain_calculator.py": {"ldp_epsilon": "DEFAULT_LDP_EPSILON"},
}


def backup_original_files():
    
    print("Backing up original files...")
    for filename in FILES_TO_MODIFY.keys():
        backup_name = f"{filename}.backup"
        if not os.path.exists(backup_name):
            shutil.copy2(filename, backup_name)
            print(f"  Backed up: {filename} -> {backup_name}")

def restore_original_files():
    
    print("Restoring original files...")
    for filename in FILES_TO_MODIFY.keys():
        backup_name = f"{filename}.backup"
        if os.path.exists(backup_name):
            shutil.copy2(backup_name, filename)
            print(f"  Restored: {backup_name} -> {filename}")


def modify_constant_value(lines: List[str], const_name: str, value: float, filename: str) -> List[str]:
    

    updated = False
    pattern = re.compile(rf"^(\s*){re.escape(const_name)}\s*=")
    for idx, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            indent = m.group(1)
            lines[idx] = f"{indent}{const_name} = {value}\n"
            updated = True
            break
    if not updated:
        print(f"  Warning: in file {filename}, constant {const_name} was not found; please check.")
    return lines


def set_parameters(beta: float, target_ratio: float, eps: float, ldp_eps: float):
    
    print(f"Setting parameters: beta={beta}, target_ratio={target_ratio}, eps={eps}, ldp_epsilon={ldp_eps}")

    for filename, param_map in FILES_TO_MODIFY.items():
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if "beta" in param_map:
            lines = modify_constant_value(lines, param_map["beta"], beta, filename)
        if "target_ratio" in param_map:
            lines = modify_constant_value(lines, param_map["target_ratio"], target_ratio, filename)
        if "eps" in param_map:
            lines = modify_constant_value(lines, param_map["eps"], eps, filename)
        if "ldp_epsilon" in param_map:
            lines = modify_constant_value(lines, param_map["ldp_epsilon"], ldp_eps, filename)

        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(lines)

def run_step(step_num: int, command: str, description: str, auto_input: str = None) -> bool:
    
    print(f"\n{'='*60}")
    print(f"Step {step_num}: {description}")
    print(f"Running command: {command}")
    print(f"{'='*60}")
    
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        if auto_input:
            
            process = subprocess.Popen(
                command.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                cwd=script_dir,
                env=env,
            )
            
            
            output, _ = process.communicate(input=auto_input + '\n')
            print(output)
            
            if process.returncode == 0:
                print(f"Step {step_num} succeeded")
                return True
            else:
                print(f"Step {step_num} failed with exit code: {process.returncode}")
                return False
        else:
            
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                cwd=script_dir,
                timeout=3600,  
                env=env,
            )
            
            print("Standard output:")
            print(result.stdout)
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            
            if result.returncode == 0:
                print(f"Step {step_num} succeeded")
                return True
            else:
                print(f"Step {step_num} failed with exit code: {result.returncode}")
                return False
                
    except subprocess.TimeoutExpired:
        print(f"Step {step_num} timed out")
        return False
    except Exception as e:
        print(f"Step {step_num} failed: {str(e)}")
        return False

def run_pipeline_steps(allocation_strategy: int, dataset_choice: str, dataset_index: str) -> bool:
    
    
    steps = [
        
        (1, "python run_hmg_pipeline.py", "Run HMG on original graphs", dataset_choice),
        (2, "python calculate_hmg_average_degrees.py", "Calculate average degrees of HMG graphs"),
        
        (3, "python attackcentre.py", "Run degree-centrality attacks", f"{allocation_strategy}\n{dataset_choice}"),
        
        (4, "python attacklcc.py", "Run clustering-coefficient attacks", f"{allocation_strategy}\n{dataset_choice}"),
        
        (5, "python hmg_attack_processor.py", "Apply HMG to attacked graphs", dataset_index),
        
        (6, "python fake_node_hmg_merger.py", "Merge fake nodes into HMG attack graphs", dataset_index),
        
        (7, "python attack_gain_calculator.py", "Compute noisy attack gain", dataset_index),
        
        (8, "python original_vs_attack_gain_calculator.py", "Compute original-vs-attack gain", dataset_index)
    ]

    for step_info in steps:
        if len(step_info) == 4:
            step_num, command, description, auto_input = step_info
            success = run_step(step_num, command, description, auto_input)
        else:
            step_num, command, description = step_info
            success = run_step(step_num, command, description)

        if not success:
            print(f"Pipeline failed at step {step_num}, aborting")
            return False

        
        time.sleep(2)

    return True


def create_results_backup(beta: float, target_ratio: float, eps: float, allocation_strategy: int, dataset_tag: str):
    
    strategy_label = get_strategy_name(allocation_strategy)
    safe_label = (
        strategy_label.replace(' ', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('θ=0.5', 'theta_0_5')
    )
    folder_name = f"{safe_label}_strategy_{allocation_strategy}_beta_{beta}_target_{target_ratio}_eps_{eps}"
    backup_dir = os.path.join("results_backup", dataset_tag, folder_name)
    os.makedirs(backup_dir, exist_ok=True)
    
    
    dirs_to_backup = [
        "reconstructed_graph",
        "hmg_average_degrees", 
        "output_py",
        "target_nodes_record",
        "HMG_attack",
        "attack_gain_results",
        "original_vs_attack_gain_results",
        "FakeNodeID",
        "HMG_FakeNode_Merged"
    ]
    
    print(f"Backed up results to: {backup_dir}")
    for dir_name in dirs_to_backup:
        if os.path.exists(dir_name):
            backup_path = os.path.join(backup_dir, dir_name)
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            shutil.copytree(dir_name, backup_path)
            print(f"  Backed up: {dir_name}")

def clean_intermediate_results():
    
    dirs_to_clean = [
        "reconstructed_graph",
        "hmg_average_degrees",
        "output_py", 
        "target_nodes_record",
        "HMG_attack",
        "attack_gain_results",
        "original_vs_attack_gain_results",
        "FakeNodeID",
        "HMG_FakeNode_Merged"
    ]
    
    print("Cleaning intermediate results...")
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"  Cleaned: {dir_name}")

def main():
    
    print("=" * 80)
    print("HMG attack analysis automation pipeline")
    print("=" * 80)
    total_combinations = sum(
        count_combinations(scene["beta_values"], scene["target_ratios"], scene["eps_values"])
        for scene in SCENARIOS
    )
    print(f"Total scenarios: {len(SCENARIOS)}")
    print(f"Total parameter combinations: {total_combinations}")
    print("=" * 80)

    backup_original_files()

    current_combination = 0
    successful_runs = 0
    failed_runs = 0

    try:
        for scenario_index, scenario in enumerate(SCENARIOS, start=1):
            allocation_strategy = scenario["allocation_strategy"]
            dataset_tag = scenario.get("dataset_tag", "as-733")
            beta_values = scenario["beta_values"]
            target_ratios = scenario["target_ratios"]
            eps_values = scenario["eps_values"]
            ldp_eps_values = scenario.get("ldp_eps_values", eps_values)
            scenario_total = count_combinations(beta_values, target_ratios, eps_values)
            strategy_name = get_strategy_name(allocation_strategy)

            print(f"\n{'-' * 80}")
            print(f"Scenario {scenario_index}/{len(SCENARIOS)}: {scenario.get('name', f'scenario_{scenario_index}')}")
            print(f"Allocation strategy: {allocation_strategy} ({strategy_name})")
            print(f"Beta values: {beta_values}")
            print(f"Target ratios: {target_ratios}")
            print(f"Epsilon values: {eps_values}")
            print(f"LDP epsilon values: {ldp_eps_values}")
            print(f"Dataset tag: {dataset_tag}")
            print(f"Scenario combinations: {scenario_total}")
            print(f"{'-' * 80}")

            for beta in beta_values:
                for target_ratio in target_ratios:
                    for eps_index, eps in enumerate(eps_values):
                        if ldp_eps_values:
                            ldp_eps = ldp_eps_values[eps_index % len(ldp_eps_values)]
                        else:
                            ldp_eps = eps
                        current_combination += 1
                        print(f"\n{'#' * 80}")
                        print(f"Global combination {current_combination}/{total_combinations}")
                        print(
                            f"Beta: {beta}, Target ratio: {target_ratio}, Eps: {eps}, LDP eps: {ldp_eps}, Strategy: {allocation_strategy} ({strategy_name})"
                        )
                        print(f"{'#' * 80}")

                        set_parameters(beta, target_ratio, eps, ldp_eps)
                        success = run_pipeline_steps(allocation_strategy, scenario.get("dataset_choice", "3"), scenario.get("dataset_index", "1"))

                        if success:
                            print(f"Combination {current_combination} succeeded")
                            successful_runs += 1
                            create_results_backup(beta, target_ratio, eps, allocation_strategy, dataset_tag)
                            if current_combination < total_combinations:
                                clean_intermediate_results()
                        else:
                            print(f"Combination {current_combination} failed")
                            failed_runs += 1

                        remaining = total_combinations - current_combination
                        print(f"Progress: success {successful_runs}, failed {failed_runs}, remaining {remaining}")

    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        restore_original_files()
        print(f"\n{'=' * 80}")
        print("Summary")
        print(f"{'=' * 80}")
        print(f"Total combinations: {total_combinations}")
        print(f"Succeeded: {successful_runs}")
        print(f"Failed: {failed_runs}")
        completion = (successful_runs / total_combinations * 100) if total_combinations else 0
        print(f"Completion: {completion:.1f}%")
        print("All outputs saved under results_backup/")
        print("Source files restored")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
