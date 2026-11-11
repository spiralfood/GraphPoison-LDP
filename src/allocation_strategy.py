import random
import math


def allocate_fake_nodes(total_fake_nodes, time_steps, strategy=0):
    

    if strategy == 0:
        return average_allocation(total_fake_nodes, time_steps)
    elif strategy == 1:
        return random_allocation(total_fake_nodes, time_steps)
    elif strategy == 2:
        return decay_allocation(total_fake_nodes, time_steps)
    else:
        raise ValueError("Invalid strategy. Use 0 (average), 1 (random), or 2 (decay).")

def average_allocation(total_fake_nodes, time_steps):
    

    base_allocation = total_fake_nodes // time_steps
    remainder = total_fake_nodes % time_steps
    
    allocation = [base_allocation] * time_steps
    
    
    for i in range(remainder):
        allocation[i] += 1
    
    return allocation

def random_allocation(total_fake_nodes, time_steps):
    

    if time_steps == 1:
        return [total_fake_nodes]
    
    
    split_points = sorted([random.randint(1, total_fake_nodes - 1) for _ in range(time_steps - 1)])
    split_points = [0] + split_points + [total_fake_nodes]
    
    
    allocation = []
    for i in range(time_steps):
        allocation.append(split_points[i + 1] - split_points[i])
    
    return allocation

def decay_allocation(total_fake_nodes, time_steps, decay_factor=0.6):
    

    weights = []
    for t in range(time_steps):
        weight = decay_factor ** (time_steps - 1 - t)
        weights.append(weight)
    
    
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    
    float_allocation = [w * total_fake_nodes for w in normalized_weights]
    
    
    allocation = [int(f) for f in float_allocation]
    allocated_total = sum(allocation)
    
    
    difference = total_fake_nodes - allocated_total
    
    if difference > 0:
        
        sorted_indices = sorted(range(time_steps), key=lambda i: float_allocation[i] - allocation[i], reverse=True)
        for i in range(difference):
            allocation[sorted_indices[i]] += 1
    elif difference < 0:
        
        sorted_indices = sorted(range(time_steps), key=lambda i: float_allocation[i] - allocation[i])
        for i in range(-difference):
            if allocation[sorted_indices[i]] > 0:
                allocation[sorted_indices[i]] -= 1
    
    return allocation

def get_strategy_name(strategy):
    

    strategy_names = {
        0: "Even distribution",
        1: "Random distribution", 
        2: "Decay-function distribution (theta=0.6)"
    }
    return strategy_names.get(strategy, "Unknown strategy")

if __name__ == "__main__":
    
    total_nodes = 100
    time_steps = 5
    
    print(f"Total fake nodes: {total_nodes}, number of time steps: {time_steps}")
    print()
    
    for strategy in [0, 1, 2]:
        allocation = allocate_fake_nodes(total_nodes, time_steps, strategy)
        print(f"Strategy {strategy} ({get_strategy_name(strategy)}):")
        print(f"Allocation result: {allocation}")
        print(f"Sum verification: {sum(allocation)} (should equal {total_nodes})")
        print()
