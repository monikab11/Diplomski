#!/bin/bash

CONFIG_FILE="config/experimenting_sweep.yaml"
PROJECT="GNN_diplomski_novi" #<ime_projekta>

# Create the sweep and extract the sweep ID
SWEEP_ID=$(/opt/conda/bin/wandb sweep --project $PROJECT $CONFIG_FILE 2>&1 | tee wandb_sweep.out | grep 'Run' | awk '{print $8}')

# Check if the sweep ID was extracted successfully
if [ -z "$SWEEP_ID" ]; then
  echo "Failed to create sweep or extract sweep ID. Check wandb_sweep.out for more details."
  exit 1
fi

echo "Sweep created with ID: $SWEEP_ID"
rm wandb_sweep.out

# Run the wandb agent with the extracted sweep ID
/opt/conda/bin/wandb agent $SWEEP_ID
