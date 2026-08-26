import argparse
import re
import csv
import sys
import statistics
import datetime
from pathlib import Path

def parse_log(file_path):
    filename = Path(file_path).name
    if "master" in filename:
        return None

    data = {
        "File": filename,
        "System/backend": None,
        "Graph version": None,
        "Run number": None,
        "Batch size": None,
        "Warm-up steps": None,
        "Measured steps": None,
        "Exit status": None,
        "Requested polling interval (ms)": 1,
        "Number of VRAM samples": None,
        "Observed median timestamp interval (ms)": None,
        "Observed maximum timestamp interval (ms)": None,
        "Throughput (samples/s)": None,
        "Elapsed Time (s)": None,
        "Forward Time (ms)": None,
        "Backward Time (ms)": None,
        "Optimizer Time (ms)": None,
        "Zero Grad Time (ms)": None,
        "Peak VRAM (MiB)": None,
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return data
        
    if "torch" in filename.lower():
        data["System/backend"] = "PyTorch"
    elif "tunx" in filename.lower():
        data["System/backend"] = "TunX"

    header_match = re.search(r"=====\s*(?:TunX\s*)?V(\d+),\s*run\s*(\d+)", content, re.IGNORECASE)
    if header_match:
        data["Graph version"] = f"V{header_match.group(1)}"
        data["Run number"] = int(header_match.group(2))

    batch_size_match = re.search(r"Batch [sS]ize:\s+(\d+)", content)
    if batch_size_match:
        data["Batch size"] = int(batch_size_match.group(1))

    benchmark_match = re.search(r"\((\d+)\s+warmup steps \+\s+(\d+)\s+measured steps\)", content)
    if benchmark_match:
        data["Warm-up steps"] = int(benchmark_match.group(1))
        data["Measured steps"] = int(benchmark_match.group(2))

    exit_match = re.search(r"exit status (\d+)", content)
    if exit_match:
        data["Exit status"] = int(exit_match.group(1))

    throughput_match = re.search(r"Throughput:\s+([\d.]+)\s+samples/s", content)
    if throughput_match:
        data["Throughput (samples/s)"] = float(throughput_match.group(1))
        
    elapsed_match = re.search(r"Elapsed time for \d+ steps:\s+([\d.]+)\s+s", content)
    if elapsed_match:
        data["Elapsed Time (s)"] = float(elapsed_match.group(1))
        
    fwd_match = re.search(r"Total Forward time:\s+([\d.]+)\s+ms", content)
    if fwd_match:
        data["Forward Time (ms)"] = float(fwd_match.group(1))
        
    bwd_match = re.search(r"Total Backward time:\s+([\d.]+)\s+ms", content)
    if bwd_match:
        data["Backward Time (ms)"] = float(bwd_match.group(1))
        
    opt_match = re.search(r"Total Optimizer time:\s+([\d.]+)\s+ms", content)
    if opt_match:
        data["Optimizer Time (ms)"] = float(opt_match.group(1))
        
    zero_match = re.search(r"Total Zero Grad time:\s+([\d.]+)\s+ms", content)
    if zero_match:
        data["Zero Grad Time (ms)"] = float(zero_match.group(1))
        
    vram_match = re.search(r"Process peak VRAM:\s+(\d+)\s+MiB", content)
    if vram_match:
        data["Peak VRAM (MiB)"] = int(vram_match.group(1))

    pid_match = re.search(r"Benchmark process PID:\s+(\d+)", content)
    target_pid = pid_match.group(1) if pid_match else None

    vram_path = Path(file_path).with_name(Path(file_path).stem + "_vram.csv")
    if vram_path.exists():
        timestamps = []
        try:
            with open(vram_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        ts_str = parts[0].strip()
                        pid = parts[1].strip()
                        if target_pid and pid != target_pid:
                            continue
                        try:
                            ts = datetime.datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f")
                            timestamps.append(ts)
                        except ValueError:
                            pass
        except Exception as e:
            pass
            
        data["Number of VRAM samples"] = len(timestamps)
        if len(timestamps) > 1:
            diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() * 1000 for i in range(1, len(timestamps))]
            data["Observed median timestamp interval (ms)"] = round(statistics.median(diffs), 2)
            data["Observed maximum timestamp interval (ms)"] = round(max(diffs), 2)

    return data

def main():
    parser = argparse.ArgumentParser(description="Gather metrics from log files")
    parser.add_argument("log_files", nargs="+", help="Path to log files")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    
    args = parser.parse_args()
    
    results = []
    for file_path in args.log_files:
        res = parse_log(file_path)
        if res is not None:
            results.append(res)
            
    if not results:
        print("No valid logs parsed.", file=sys.stderr)
        return
        
    results.sort(key=lambda x: x["File"])
    
    headers = list(results[0].keys())
    
    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
    else:
        col_widths = {h: len(h) for h in headers}
        for row in results:
            for h in headers:
                val_str = str(row[h]) if row[h] is not None else "N/A"
                col_widths[h] = max(col_widths[h], len(val_str))
                
        format_str = " | ".join(f"{{:<{col_widths[h]}}}" for h in headers)
        print(format_str.format(*headers))
        print("-+-".join("-" * col_widths[h] for h in headers))
        for row in results:
            print(format_str.format(*(str(row[h]) if row[h] is not None else "N/A" for h in headers)))

if __name__ == "__main__":
    main()

