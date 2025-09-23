
#!/bin/bash

#$ -M tma2@nd.edu    # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 16        # Specify parallel environment and legal core size
#$ -q gpu@@yye7_lab  # Run on the GPU cluster
#$ -o ~/Projects/LLaMA-Factory/logs/$JOB_NAME_$JOB_ID.log
#$ -l gpu_card=2     # Run on 1 GPU card
#$ -N LLMHalluc      # Specify job name

source ~/.bashrc

# Change to project directory
cd ~/Projects/LLaMA-Factory
source ./bash/sys/activate_env.sh


python - <<'PY'
import torch, sys
print("PyTorch CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("Device names:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.is_available(), "No GPU found; aborting."
PY


llamafactory-cli train ./configs/qwen3/qwen3_nothink_bt_squad_v2_train.yaml 
