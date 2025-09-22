#!/bin/bash
# Script to submit job with YAML config validation
# Usage: ./submit_job.sh [config.yaml]

CONFIG_PATH=$1

# Check if path is provided
if [[ -z "$CONFIG_PATH" ]]; then
    echo "Error: No config path provided"
    exit 1
fi

# Check if file exists
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: File not found: $CONFIG_PATH"
    exit 1
fi

# Check if file ends with .yaml
if [[ ! "$CONFIG_PATH" == *.yaml ]]; then
    echo "Error: File must end with .yaml"
    exit 1
fi

export EXP_CONFIG="$CONFIG_PATH"
qsub ./bash/crc/base.sh