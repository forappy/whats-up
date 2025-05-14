#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python llama_3.2_infer.py \
    --model_name "/lpai/dataset/yhf-model/0-0-2/Llama-3.2-11B-Vision-Instruct"\
    # --finetuning_path "/mnt/pfs-mc0p4k/nlu/team/yuhaofu/llama/llama-cookbook/lora/base_viscot_3/base_viscot_3"

