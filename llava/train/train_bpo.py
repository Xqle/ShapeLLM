import os
import json
import copy
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers

from trl import DPOConfig
from trl.trainer import DPOTrainer
from trl.trainer.utils import DPODataCollatorWithPadding

from llava.constants import IGNORE_INDEX, DEFAULT_POINT_TOKEN, DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN
from datasets import Dataset

from llava.model import *
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_point_token, process_pts, load_pts, occlusion, rotation

# for debug
import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_path: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_pt_start_end: bool = field(default=False)
    mm_use_pt_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    prompt_token_num: int = field(default=1)
    with_ape: bool = field(default=True)
    with_local: bool = field(default=True)
    with_global: bool = field(default=True)
    with_color: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    point_folder: Optional[str] = field(default=None)
    sample_points_num: int = field(default=4096)
    occlusion: bool = field(default=False)


@dataclass
class TrainingArguments(DPOConfig):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


class CustomedDPODataCollator(DPODataCollatorWithPadding):
    def __init__(self, data_args, *args, **kwargs):
        super(CustomedDPODataCollator, self).__init__(*args, **kwargs)
        self.data_args = data_args



def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # load model
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir, 
        low_cpu_mem_usage=True
    )
    model.config.use_cache = False

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # set the default conversation template
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # load ReConV2 as point encoder
    vision_tower = model.get_vision_tower()
    vision_tower.load_model()
    
    data_args.is_multimodal = True

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_pt_start_end = data_args.mm_use_pt_start_end = model_args.mm_use_pt_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_pt_start_end = model_args.mm_use_pt_start_end
    model.config.mm_use_pt_patch_token = model_args.mm_use_pt_patch_token
    model.config.with_color = model_args.with_color
    model.config.sample_points_num = data_args.sample_points_num
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # TODO: process dataset
    with open(data_args.data_path, 'r') as file:
        dataset = json.load(file)
     
    point_list = [data['point'] for data in dataset]
    question_list = [data['question'] for data in dataset]
    response_list = [data['response'] for data in dataset]
    modified_response_list = [data['modified_response'] for data in dataset]
    dataset = Dataset.from_dict({'prompt': question_list, 'chosen': response_list, 'rejected': modified_response_list})
    # dataset = Dataset.from_dict({'point': point_list, 'question': question_list, 'chosen': response_list, 'rejected': modified_response_list})

    a = 100
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    dpo_trainer.train()

    




if __name__ == '__main__':
    train()