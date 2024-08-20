import os
import torch
import argparse
from transformers import TextStreamer

import random

import json
from tqdm import tqdm

from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import POINT_TOKEN_INDEX, DEFAULT_POINT_TOKEN, DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN
from llava.mm_utils import load_pts, process_pts, rotation, tokenizer_point_token, get_model_name_from_path, \
    KeywordsStoppingCriteria


def main(args):
    # Model
    disable_torch_init()

    # load the supervised fine-tune dataset and do the random sample
    if args.sft_ds_path is not None:
        with open(args.sft_ds_path, 'r') as file:
            sft_ds = json.load(file)
    sample_num = args.sample_num
    if sample_num is not None:
        sft_ds = random.sample(sft_ds, sample_num)
    
    # load the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit,
                                                          args.load_4bit, device=args.device)
    
    # set the conversation mode to bpo error injection mode
    conv_mode = "llava_bpo_error_injection_v3"
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode,
                                                                                                            args.conv_mode,
                                                                                                            args.conv_mode))
    else:
        args.conv_mode = conv_mode
    
    point_list = []
    question_list = []
    response_list = []
    modified_response_list = []
    for idx in tqdm(range(len(sft_ds)), desc='Processing dataset'):
        item = sft_ds[idx]
        point_list.append(item['point'])

        conversation = item['conversations']
        question, response = conversation[0]['value'].split('<point>\n')[-1], conversation[1]['value']
        question_list.append(question)
        response_list.append(response)


        conv = conv_templates[args.conv_mode].copy()
        
        user_msg = 'Question: ' + question + '\n' + 'Response: ' + response + '\n' + 'Modified response: '
        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], None)

        # print(f"{conv.roles[0]}: ", user_msg)
        # print(f"{conv.roles[1]}: ", end="")

        prompt = conv.get_prompt()
        input_ids = tokenizer_point_token(prompt, tokenizer, POINT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                points=None,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                # streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        
        no_error_injection_count = 0
        no_error_injection_idx_list = []
        if outputs == response:
            no_error_injection_count += 1
            no_error_injection_idx_list.append(idx)
        modified_response_list.append(outputs)

    print("BPO error injection finished!")

    # save the bpo error injection data to a json file
    bpo_error_injection_data = []
    for i in range(len(question_list)):
        if i not in no_error_injection_idx_list:
            item = {
                "point": point_list[i],
                "question": question_list[i],
                "response": response_list[i],
                "modified_response": modified_response_list[i]
            }
            bpo_error_injection_data.append(item)


    with open(args.bpo_ds_path, 'w') as json_file:
        json.dump(bpo_error_injection_data, json_file, indent=4)
    
    print("BPO error injection data saved to " + args.bpo_ds_path)

    # save the no error injection data index and count to a json file
    item = {"no_error_injection_count": no_error_injection_count, "no_error_injection_idx_list": no_error_injection_idx_list}
    with open('./playground/data/shapellm/gapartnet_27k_no_error_injection.json', 'w') as json_file:
        json.dump(item, json_file, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qizekun/ShapeLLM_13B_gapartnet_v1.0")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--pts-file", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--objaverse", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # bpo_error_injection
    parser.add_argument("--bpo-error-injection", action="store_true")
    parser.add_argument("--sft-ds-path", type=str, default="./playground/data/shapellm/gapartnet_sft_27k_openai.json", help="Path of supervised finetune dataset")
    parser.add_argument("--sample_num", type=int, default=None, help="Number of samples to be sampled from the dataset")
    parser.add_argument("--bpo-ds-path", type=str, default="./playground/data/shapellm/gapartnet_bpo_27k_error_injection.json")
    args = parser.parse_args()
    main(args)