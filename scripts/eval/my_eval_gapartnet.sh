#!/bin/bash

MODEL_VERSION=ShapeLLM_13B
TAG=gapartnet_v1.0

mkdir -p ./playground/data/eval/gapartnet/results

python -m llava.eval.eval_gapartnet \
    --answers-file ./playground/data/eval/gapartnet/answers/$MODEL_VERSION'_'$TAG.jsonl \
    --gt-file ./playground/data/eval/gapartnet/gt.jsonl \
    --output-file ./playground/data/eval/gapartnet/results/$MODEL_VERSION'_'$TAG.jsonl \
    --max_workers 16