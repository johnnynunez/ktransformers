#!/bin/bash
export CAPTURE_PLUGIN_PATH=ktransformers/util/npu_graph_so/arm
export USE_MERGE=0
export INF_NAN_MODE_FORCE_DISABLE=1
export TASK_QUEUE_ENABLE=0
#export PROF_DECODE=1
#export PROF_PREFILL=1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

torchrun --nproc_per_node 1 \
         --master_port 25565 \
         -m ktransformers.local_chat_npu \
         --cpu_infer 20 \
         --model_path /mnt/data/models/DeepSeek-R1-q4km-w8a8\
         --gguf_path /mnt/data/models/DeepSeek-R1-q4km-w8a8 \
         --optimize_config_path ./ktransformers/optimize/optimize_rules/npu/DeepSeek-V3-Chat-800IA2-npu.yaml \
         --max_new_tokens 30 \
         --tp 1

#          --use_cuda_graph True \