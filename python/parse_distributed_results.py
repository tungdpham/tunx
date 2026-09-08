import re
import math
import sys

def parse_log(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Find all config headers
    # Each header looks like: Config: ../configs/distributed_v1.json | Policy: compute_bandwidth
    sections_raw = re.split(r'Config: \.\./configs/distributed_(v[1-4])\.json \| Policy: (\w+)', content)
    # sections_raw[0] is everything before the first config
    # sections_raw[1] is 'v1', sections_raw[2] is 'compute_bandwidth'
    # sections_raw[3] is the text for this run
    
    results = {}
    
    for i in range(1, len(sections_raw), 3):
        workload = sections_raw[i].upper()
        policy = sections_raw[i+1]
        section = sections_raw[i+2]
        
        # Policy name mapping
        if policy == "compute_bandwidth":
            policy_name = "Compute+bandwidth"
        elif policy == "compute":
            policy_name = "Compute only"
        elif policy == "equal":
            policy_name = "Equal edge count"
        else:
            policy_name = policy
            
        # Parse partition metrics
        w1_edges_match = re.search(r'Worker 1 edges: (\d+) \(edges 0 to (\d+)\)', section)
        w2_edges_match = re.search(r'Worker 2 edges: (\d+) \(edges (\d+) to (\d+)\)', section)
        
        cut_edge = "N/A"
        edges_w1_w2 = "N/A/N/A"
        if w1_edges_match and w2_edges_match:
            w1_count = w1_edges_match.group(1)
            w1_end = w1_edges_match.group(2)
            w2_count = w2_edges_match.group(1)
            cut_edge = w1_end
            edges_w1_w2 = f"{w1_count}/{w2_count}"
            
        boundary_match = re.search(r'Boundary 1 -> 2 size \(MiB\): (\d+)', section)
        boundary = boundary_match.group(1) if boundary_match else "N/A"
        
        predicted_j_match = re.search(r'Predicted Bottleneck J \(ms\): ([\d\.]+)', section)
        predicted_j = predicted_j_match.group(1) if predicted_j_match else "N/A"
        
        # Parse throughput
        throughput_match = re.search(r'Throughput: ([\d\.]+) samples/s', section)
        if throughput_match:
            throughput_str = throughput_match.group(1)
        else:
            throughput_str = "N/A"
        
        key = (workload, policy_name)
        results[key] = {
            "cut": cut_edge,
            "edges": edges_w1_w2,
            "boundary": boundary,
            "predicted_j": predicted_j,
            "throughput": throughput_str
        }
        
    return results

def main():
    if len(sys.argv) < 2:
        filename = "../experiments/benchmark_results.log"
    else:
        filename = sys.argv[1]
    
    results = parse_log(filename)
    
    workloads = ["V1", "V2", "V3", "V4"]
    policies = ["Equal edge count", "Compute only", "Compute+bandwidth"]
    
    print("Workload | Policy | Cut after edge | Edges W1/W2 | Boundary (MiB) | Predicted J (ms) | Throughput (samples/s)")
    print("-" * 122)
    for w in workloads:
        for p in policies:
            res = results.get((w, p), {})
            print(f"{w:8} | {p:18} | {res.get('cut', 'N/A'):14} | {res.get('edges', 'N/A'):11} | {res.get('boundary', 'N/A'):14} | {res.get('predicted_j', 'N/A'):16} | {res.get('throughput', 'N/A')}")
            
if __name__ == '__main__':
    main()
