from llava.model import *
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live



model = LlavaLlamaForCausalLM.from_pretrained(
    'qizekun/ShapeLLM_13B_gapartnet_v1.0',
    low_cpu_mem_usage=True
)

estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=4, num_nodes=1)