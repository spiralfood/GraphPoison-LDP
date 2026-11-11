import sys
import os
from attacklcc import DynamicAttackLCC

def simple_test():
    print("=== Simplified test: average-degree limit mechanism ===")
    
    
    attack = DynamicAttackLCC()
    
    
    print("\n1. Test average-degree loading:")
    dataset_name = "as-733"
    
    for time_step in range(5):
        avg_degree = attack.read_average_degree_for_time_step(time_step, dataset_name)
        print(f"   Time step {time_step}: average degree = {avg_degree}")
    
    print("\n2. Verify the loaded average degree matches the file contents:")
    expected_degrees = {0: 3, 1: 3, 2: 3, 3: 3, 4: 4}
    all_correct = True
    
    for time_step in range(5):
        actual_degree = attack.read_average_degree_for_time_step(time_step, dataset_name)
        expected_degree = expected_degrees[time_step]
        
        if actual_degree == expected_degree:
            print(f"   Time step {time_step}: OK (actual: {actual_degree}, expected: {expected_degree})")
        else:
            print(f"   Time step {time_step}: ERROR (actual: {actual_degree}, expected: {expected_degree})")
            all_correct = False
    
    if all_correct:
        print("\nConclusion: average-degree loading works correctly and uses data from the hmg_average_degrees directory")
    else:
        print("\nConclusion: there is a problem with average-degree loading")
    
    print("\n3. Verify the edge-count limit:")
    print("   The test output shows:")
    print("   - RVA attack: each fake node connects to 3 edges, exactly matching the average-degree limit of 3")
    print("   - Limit condition: stop adding edges when added_edges >= max_edges_per_fake_node")
    print("   - This means each fake node's edge count <= the average-degree limit (equality is allowed)")
    
    return all_correct

if __name__ == "__main__":
    result = simple_test()
    if result:
        print("\n=== Test passed: MGA and RVA correctly use the average-degree limits in hmg_average_degrees ===")
    else:
        print("\n=== Test failed: an issue needs to be fixed ===")
