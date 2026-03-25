#!/bin/bash
# Monitor GPU metrics during training
# Targets: Memory <90%, Clock stable at 2800MHz, Temp <80C

echo "=== GPU Monitoring (Overclock: 900mV @ 2800MHz core, +200 mem) ==="
echo "Time | Mem(%) | Temp(C) | Clock(MHz) | Power(W) | Status"
echo "-------|--------|---------|-----------|----------|--------"

while true; do
    metrics=$(nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,clocks.current.graphics,power.draw --format=csv,noheader,nounits)
    
    mem_used=$(echo $metrics | awk '{print $1}')
    mem_total=$(echo $metrics | awk '{print $2}')
    temp=$(echo $metrics | awk '{print $3}')
    clock=$(echo $metrics | awk '{print $4}')
    power=$(echo $metrics | awk '{print $5}')
    
    mem_pct=$(( mem_used * 100 / mem_total ))
    
    # Status check
    status="OK"
    if [ "$mem_pct" -gt 90 ]; then
        status="MEM_HIGH"
    fi
    if [ "$temp" -gt 80 ]; then
        status="TEMP_HIGH"
    fi
    if [ "$clock" -lt 2700 ]; then
        status="CLOCK_LOW (throttling?)"
    fi
    
    printf "%s | %3d%% | %3d | %4d | %6.1f | %s\n" "$(date +%H:%M:%S)" "$mem_pct" "$temp" "$clock" "$power" "$status"
    sleep 30
done
