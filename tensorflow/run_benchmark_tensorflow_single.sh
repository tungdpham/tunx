mkdir -p logs

nohup bash -c '
for run in {1..5}; do
    for version in {1..4}; do
        config="configs/tunx_v${version}.json"
        log="logs/tf_v${version}_run${run}.log"
        memory_log="logs/tf_v${version}_run${run}_vram.csv"

        echo "===== TF V${version}, run ${run}, started $(date) =====" | tee "$log"

        # Launch Python directly so bench_pid is the actual Python process ID.
        python3 -u tensorflow/tensorflow_trainer.py \
            --config "$config" --benchmark >> "$log" 2>&1 &
        bench_pid=$!

        echo "Benchmark process PID: $bench_pid" | tee -a "$log"

        # Record per-process GPU memory every 1 ms.
        nvidia-smi \
            --query-compute-apps=timestamp,pid,used_gpu_memory \
            --format=csv,noheader,nounits \
            --loop-ms=1 > "$memory_log" 2>/dev/null &
        monitor_pid=$!

        wait "$bench_pid"
        status=$?

        kill "$monitor_pid" 2>/dev/null
        wait "$monitor_pid" 2>/dev/null

        peak_mib=$(awk -F "," -v target="$bench_pid" '"'"'
        {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3)
            if ($2 == target && $3 + 0 > maximum)
                maximum = $3 + 0
        }
        END {
            if (maximum > 0)
                printf "%.0f", maximum
            else
                print "N/A"
        }
        '"'"' "$memory_log")

        if [ "$peak_mib" != "N/A" ]; then
            peak_gib=$(awk -v mib="$peak_mib" \
                '"'"'BEGIN {printf "%.3f", mib / 1024}'"'"')

            echo "Process peak VRAM: ${peak_mib} MiB (${peak_gib} GiB)" \
                | tee -a "$log"
        else
            echo "Process peak VRAM: N/A (no sample found for PID $bench_pid)" \
                | tee -a "$log"
        fi

        echo "===== Finished $(date), exit status ${status} =====" \
            | tee -a "$log"

        if [ "$status" -ne 0 ]; then
            echo "Benchmark failed: TF V${version}, run ${run}" \
                | tee -a "$log" >&2
        fi

        # Allow the CUDA context and process allocation to be released.
        sleep 2
    done
done
' > logs/tf_nohup_master.log 2>&1 &

echo "TensorFlow benchmark controller PID: $!"
