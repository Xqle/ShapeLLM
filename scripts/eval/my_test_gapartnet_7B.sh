#!/bin/bash

MODEL_VERSION=ShapeLLM_7B
TAG=gapartnet_v1.0

CUDA_VISIBLE_DEVICES=6 python -m llava.eval.model_vqa \
    --model-path qizekun/$MODEL_VERSION'_'$TAG \
    --question-file ./playground/data/eval/gapartnet/question.jsonl \
    --point-folder ./playground/data/shapellm/gapartnet_pcs \
    --answers-file ./playground/data/eval/gapartnet/answers/$MODEL_VERSION'_'$TAG.jsonl \
    --conv-mode vicuna_v1 \
    --num_beams 5
    