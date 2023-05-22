#!/bin/bash

# Function to measure startup time
measure_startup_time() {
    startup_time=$( { time -p annif --help >/dev/null; } 2>&1 | awk '/^user/{u=$2}/^sys/{s=$2} END{print u+s}' )
    echo "$startup_time"
}

startup_time1=$(measure_startup_time)
startup_time2=$(measure_startup_time)
startup_time3=$(measure_startup_time)
startup_time4=$(measure_startup_time)

# Calculate the average startup time
average_startup_time=$(echo "scale=3; ($startup_time1 + $startup_time2 + $startup_time3 + $startup_time4) / 4" | bc)

# Print the average startup time
echo "Average Startup time: $average_startup_time seconds"

# Set the threshold for acceptable startup time in seconds
threshold=0.300

# Compare the average startup time with the threshold
if (( $(echo "$average_startup_time > $threshold" | bc -l) )); then
    echo "Startup time (user + sys time) exceeds the threshold of $threshold s. Test failed."
    exit 1
fi
