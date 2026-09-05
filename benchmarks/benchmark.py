import os
import json
import subprocess
import argparse
import time

def run_benchmark(configs, policies, build_dir, log_file, auto_run):
    with open(log_file, 'w') as f_out:
        for config_path in configs:
            for policy in policies:
                print(f"\n==================================================")
                print(f"Preparing to run {config_path} with policy: {policy}")
                print(f"==================================================")
                
                # Load the config
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update the partition policy
                config_data['partition_policy'] = policy
                
                # Write to a temporary config file in the same directory
                config_dir = os.path.dirname(config_path)
                config_name = os.path.basename(config_path)
                temp_config_path = os.path.join(config_dir, f"temp_{config_name}")
                
                with open(temp_config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                if not auto_run:
                    input(f"Please ensure tcp_worker instances are running for {config_path}.\nPress Enter to start tcp_coordinator...")
                else:
                    print(f"Waiting 5 seconds for workers to be ready... (auto-run)")
                    time.sleep(5)
                
                # Run the coordinator
                cmd = [os.path.join(build_dir, "bin/tcp_coordinator"), "--config", temp_config_path]
                print(f"Running command: {' '.join(cmd)}")
                
                f_out.write(f"\n{'='*60}\n")
                f_out.write(f"Config: {config_path} | Policy: {policy}\n")
                f_out.write(f"{'='*60}\n")
                f_out.flush()
                
                try:
                    # Run the process, streaming output to both console and the log file
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    
                    for line in process.stdout:
                        print(line, end='')
                        f_out.write(line)
                        f_out.flush()
                        
                    process.wait()
                    print(f"Process finished with return code {process.returncode}")
                except KeyboardInterrupt:
                    print("\nBenchmark interrupted by user.")
                    if os.path.exists(temp_config_path):
                        os.remove(temp_config_path)
                    return
                except Exception as e:
                    print(f"Error running coordinator: {e}")
                    f_out.write(f"Error: {e}\n")
                
                # Clean up the temporary config file
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks for multiple configs and partition policies.")
    parser.add_argument("--auto", action="store_true", help="Run automatically without prompting (make sure workers auto-restart).")
    parser.add_argument("--log", type=str, default="benchmark_results.log", help="File to log all outputs.")
    parser.add_argument("--build-dir", type=str, default="build", help="Path to the build directory.")
    
    args = parser.parse_args()
    
    configs_to_run = [
        "configs/distributed_v1.json",
        "configs/distributed_v2.json",
        "configs/distributed_v3.json",
        "configs/distributed_v4.json"
    ]
    
    policies_to_run = [
        "compute_bandwidth",
        "compute",
        "equal"
    ]
    
    # Ensure build dir exists
    if not os.path.isdir(args.build_dir):
        print(f"Error: Build directory '{args.build_dir}' not found.")
        print("Please run this script from the project root or specify --build-dir.")
        exit(1)
        
    run_benchmark(configs_to_run, policies_to_run, args.build_dir, args.log, args.auto)
    print(f"\nBenchmark finished. Full results saved to {args.log}")
