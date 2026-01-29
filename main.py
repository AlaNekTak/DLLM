# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7' # FIXME

# %%
import argparse

from tqdm.auto import tqdm

import os
import time
import random

from prompt_manager import get_prompt
from path_manager import get_main_path
import gc



# %%


def get_args(run_in_notebook=False):
    parser = argparse.ArgumentParser(description='Run the LLM')

    # Model arguments
    parser.add_argument("--model_index", type=int, default=13, help="Index of the model to use, \
                                                                    0:meta-llama/Llama-3.2-1B-Instruct, \
                                                                    1:meta-llama/Llama-3.1-8B-Instruct,  \
                                                                    2:meta-llama/Llama-3.1-70B-Instruct, \
                                                                    3:google/gemma-3-4b-it,              \
                                                                    4:google/gemma-3-12b-it,             \
                                                                    5:Qwen/Qwen3-8B,                     \
                                                                    6:google/gemma-2-2b-it,              \
                                                                    7:google/gemma-2-2b,                 \
                                                                    8:meta-llama/Llama-3.2-1B, \
                                                                    9:meta-llama/Llama-3.1-8B,  \
                                                                    10:meta-llama/Llama-3.1-70B, \
                                                                    11:google/gemma-3-4b-pt, \
                                                                    12:google/gemma-3-12b-pt, \
                                                                    13:Qwen/Qwen3-4B, \
                                                                    14:allenai/Olmo-3-7B-Think, \
                                                                    15:allenai/Olmo-3-7B-Instruct,\
                                                                    16:Qwen/Qwen3-4B-Instruct-2507, \
                                                                    17:allenai/Olmo-3-32B-Think")
    
    
    
    parser.add_argument("--use_quantization", action='store_true', help="Use quantization for the model", default=False)
    # Task arguments
    parser.add_argument('--dataset', type=str, default='rationality', help='Evaluation dataset to use')

    parser.add_argument('--result_save_path', type=str, default='results/', required=False, help='Path to save the results')
    parser.add_argument('--skip_with_file', type=str, default=None)
    # Inference arguments
    parser.add_argument('--bs', type=int, default=6, required=False, help='Batch size for inference')
    parser.add_argument('--seed', type=int, default=1, required=False, help='Random seed for reproducibility')
    parser.add_argument('--temperature', type=float, default=0.6, required=False, help='Temperature for sampling')
    parser.add_argument('--argmax', action='store_true', help='Use argmax for sampling instead of sampling from the distribution', default=True)
    parser.add_argument('--thinking', action='store_true', help='Enable thinking mode', default=False)
    parser.add_argument('--use_kv_cache', action='store_true', help='Use kv cache for faster inference', default=True)
    parser.add_argument('--eval_only', action='store_true', help='Eval only', default=False)
    
    parser.add_argument('--steer_type', type=str, default='None', choices=['None', 'Intervention', 'Prompt', 'SFT', 'DPO'], help='Type of steering to use')
    
    # General arguments
    parser.add_argument('--concept_source', type=str, default='emotions')
    parser.add_argument('--concept', type=str, default='sadness', help='Concept to steer towards')
    
    # Intervention arguments
    parser.add_argument('--intervention_type', type=str, default='add', choices=['add', 'replace'], help='Type of steering to use')
    parser.add_argument('--intervention_source', type=str, default='probeall', choices=['probe', 'probeassistant', 'meandiff', 'meandiffall', 'meandiffassistant', 'mean', 'probeall'], help='Source of the steering signal')
    parser.add_argument('--steer_coeff', type=float, default= 1.8, required=False, help='Coefficient for intervention')
    parser.add_argument('--steer_layers', type=str, default='all') #9,10
    parser.add_argument('--steer_locs', type=str, default='7')
    parser.add_argument('--steer_normalize', type=bool, default=True, help='Whether to normalize the steering vector before applying the coefficient')
    parser.add_argument('--steer_renormalize', type=bool, default=False, help='Whether to rescale the manipulated activation to the original range')
    parser.add_argument('--steer_tokens', type=str, default='new', choices=['new', 'think'], help='Which tokens to apply the steering on; new means only on the newly generated tokens; think means only on the thinking tokens and not the tokens generated after that.')    
    
    # Prompt arguments
    parser.add_argument('--prompt_strength', type=str, default='medium', choices=['high', 'low', 'medium', 'neg', 'neg_high', 'neg_low', 'very_high', 'very_low'], help='Strength of the prompt steering')
    parser.add_argument('--prompt_template', type=str, default='template_1', help='Template to use for the prompt steering')
    parser.add_argument('--prompt_method'  , type=str, default='few', choices=['zero', 'few', 'p2'], help='Strength of the prompt steering')

    # SFT and DPO arguments
    parser.add_argument('--PEFT_lr', type=float, default=1e-6)
    parser.add_argument('--PEFT_steps', type=int, default=512)
    
    if run_in_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    
    args.device = 'cuda'
    
    return args

def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

args = get_args(run_in_notebook = in_notebook())

# %%
if args.model_index in [2, 4, 10, 17]:  # Llama3-70B, Gemma-12B, Llama3-70B, Olmo-3-32B-Think
    args.use_quantization = True
    print('Automatically turning on quantization...')
    
thinking_models = [5, 13, 14, 17]  # Qwen3-8B, Qwen3-4B, Olmo-3-7B-Think, Olmo-3-32B-Think
if not args.model_index in thinking_models and args.thinking:
    args.thinking = False
    print('Thinking mode is only available for certain models. Disabling thinking mode.')

if args.model_index in [5, 13, 14, 17]:  # Qwen3-8B, Qwen3-4B, Olmo-3-7B-Think, Olmo-3-32B-Think
    stop_thinking_string = '</think>'
    start_thinking_string = '<think>'
else:
    if args.thinking:
        raise ValueError("Thinking strings are unknown for this model.")

if args.steer_type == 'Intervention' and args.steer_tokens == 'think' and not args.thinking:
    raise ValueError("Steering on thinking tokens requires thinking mode to be enabled.")
    

# %%
if args.argmax:
    print("Using argmax sampling, setting temperature to 0.6")
    args.temperature = 0.6

# %%
print(f'Dataset: {args.dataset}')
print(f'Batch size: {args.bs}')
print(f'Argmax or Sample: {"argmax" if args.argmax else "sample"}')
print(f'Temperature:', args.temperature)
print(f'Steer type: {args.steer_type}')
print(f'device: {args.device}')
if args.steer_type != 'None':
    print(f'Concept: {args.concept}')
    if args.steer_type == 'Intervention':
        print(f'Intervention source: {args.intervention_source}')
        print(f'Steer coeff: {args.steer_coeff}')
        print(f'Intervention layers: {args.steer_layers}')
        print(f'Intervention type: {args.intervention_type}')        
        print(f'Steer tokens: {args.steer_tokens}')
    elif args.steer_type == 'Prompt':
        print(f'Prompt Strength: {args.prompt_strength}')
        print(f'Prompt Template: {args.prompt_template}')
        print(f'Prompt Method: {args.prompt_method}')
    elif args.steer_type == 'SFT':
        print(f'SFT lr: {args.PEFT_lr}, steps: {args.PEFT_steps}')
    elif args.steer_type == 'DPO':
        print(f'DPO lr: {args.PEFT_lr}, steps: {args.PEFT_steps}')

    


# %%
model_names =       ['meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.1-70B-Instruct', 
                     'google/gemma-3-4b-it', 'google/gemma-3-12b-it', 
                     'Qwen/Qwen3-8B', 
                     'google/gemma-2-2b-it', 'google/gemma-2-2b',
                     'meta-llama/Llama-3.2-1B', 'meta-llama/Llama-3.1-8B', 'meta-llama/Llama-3.1-70B',
                     'google/gemma-3-4b-pt', 'google/gemma-3-12b-pt', 
                     'Qwen/Qwen3-4B',
                     'allenai/Olmo-3-7B-Think',
                     'allenai/Olmo-3-7B-Instruct',
                     'Qwen/Qwen3-4B-Instruct-2507', 
                     'allenai/Olmo-3-32B-Think'
                     ]

model_short_names = ['Llama3.2_1B', 'Llama3.1_8B', 'Llama3.1_70B', 
                     'Gemma3_4B', 'Gemma3_12B', 
                     'Qwen3_8B', 
                     'Gemma2_2B', 'Gemma2_2B_PT', 
                     'Llama3.2_1B_PT', 'Llama3.1_8B_PT', 'Llama3.1_70B_PT', 
                     'Gemma3_4B_PT', 'Gemma3_12B_PT',
                     'Qwen3_4B',
                     'Olmo3_7B_Think', 
                     'Olmo3_7B_Instruct',
                     'Qwen3_4B_Instruct', 
                     'Olmo3_32B_Think'
                     ]

model_name, model_short_name = list(zip(model_names, model_short_names))[args.model_index]  

if args.use_quantization:
    model_short_name = model_short_name + "_quantized"

model_path = model_short_name + ("_thinking" if args.thinking else "")

print('model_name:', model_name, args.model_index, 'thinking mode:', args.thinking)

save_dir = get_main_path(args.temperature, args.argmax, args.result_save_path, model_path, args.dataset, args.steer_type, args.concept_source, args.concept, 
                         args.steer_layers, args.intervention_type, args.intervention_source, args.steer_coeff, args.steer_tokens,
                         args.prompt_method, args.prompt_template, args.prompt_strength, args.PEFT_lr, args.PEFT_steps, seed = args.seed)

if args.skip_with_file:
    file_path = save_dir + '/' + args.skip_with_file
    if os.path.exists(file_path):
        print('File path exists, exiting the run:', file_path)
        assert False, 'Exit...'

# %%


# %%
import torch
import numpy as np
from unsloth import FastLanguageModel, FastModel
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from steer_manager import get_steer_fn, load_steer_vectors, apply_wieghted_sum, get_default_steer_fn
import torch.nn.functional as F
import pandas as pd
def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

# %%
set_seed_everywhere(args.seed)


from LLMs.my_llama import SteeredLlamaForCausalLM
from LLMs.my_gemma3 import SteeredGemma3ForCausalLM
from LLMs.my_qwen3 import SteeredQwen3ForCausalLM
from LLMs.my_gemma2 import SteeredGemma2ForCausalLM
from LLMs.my_olmo3 import SteeredOlmo3ForCausalLM
from transformers import BitsAndBytesConfig

model_classes =     [SteeredLlamaForCausalLM, SteeredLlamaForCausalLM, SteeredLlamaForCausalLM, 
                     SteeredGemma3ForCausalLM, SteeredGemma3ForCausalLM, 
                     SteeredQwen3ForCausalLM, 
                     SteeredGemma2ForCausalLM, SteeredGemma2ForCausalLM,
                     SteeredLlamaForCausalLM, SteeredLlamaForCausalLM, SteeredLlamaForCausalLM,
                     SteeredGemma3ForCausalLM, SteeredGemma3ForCausalLM,
                     SteeredQwen3ForCausalLM,
                     SteeredOlmo3ForCausalLM,
                     SteeredOlmo3ForCausalLM,
                     SteeredQwen3ForCausalLM, 
                     SteeredOlmo3ForCausalLM
                     ]

model_class = model_classes[args.model_index]    

print('model::', args.model_index, model_class, model_name, model_short_name)

# %%
if args.model_index in [3, 4, 11, 12]: # models that support vision
    UnslothModelClass = FastModel
else:
    UnslothModelClass = FastLanguageModel

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token    

if args.steer_type in ['SFT', 'DPO']:
    
    max_seq_length = 4096
    load_path = f"PEFT_models/{args.steer_type}/{model_short_name}/{args.PEFT_lr}_{args.PEFT_steps}/{args.concept_source}/{args.concept}"

    model, _tok = UnslothModelClass.from_pretrained(
        model_name = load_path,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = args.use_quantization,
    )
    model.eval()
    UnslothModelClass.for_inference(model) # Enable native 2x faster inference
    
else:
    if args.use_quantization:    
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
    else:
        quantization_config = None

    try:
        model.eval()
    except:
        model = model_class.from_pretrained(model_name, device_map=args.device, quantization_config=quantization_config) 
        model.eval()
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    torch.cuda.empty_cache()


# %%
model_name

# %%
default_steer_fn = get_default_steer_fn()
if args.steer_type == 'Intervention':
    steer_vecs, steer_layers, steer_locs, steer_tokens = load_steer_vectors(model_short_name, model.config.num_hidden_layers, model.config.hidden_size, [args.concept], [args.concept_source], args.steer_layers, args.steer_locs, args.intervention_source, args.steer_normalize)
    steer_vec = apply_wieghted_sum(steer_vecs, torch.tensor([args.steer_coeff]), model.device)
    steer_fn = get_steer_fn(steer_vec, steer_layers, steer_locs, steer_tokens, model.config.hidden_size, args.intervention_type, args.steer_renormalize)
    system_prompt_suffix = ""

    model.set_steer_fn(steer_fn)

elif args.steer_type == 'None':
    steer_fn = default_steer_fn
    system_prompt_suffix = ""
    model.set_steer_fn(steer_fn)
    
elif args.steer_type == 'Prompt':
    steer_fn = default_steer_fn
    model.set_steer_fn(steer_fn)
    system_prompt_suffix = get_prompt(domain_name = args.concept_source, prompt_method=args.prompt_method, template_name=args.prompt_template, concept=args.concept, prompt_strength=args.prompt_strength)

elif args.steer_type in ['SFT', 'DPO']:
    system_prompt_suffix = ""

else:
    raise ValueError(f"Not implemented steer type {args.steer_type}")

# %%


os.makedirs(save_dir, exist_ok=True)    
print('created directory:', save_dir)



# %%
import itertools

class  SimpleDataset:
    def __init__(self, eval_only = False):
        self.eval_only = eval_only
        self.results = None
        self.description = 'Pretend that you are a human. How would you feel today?' #Generate some synthetic reviews with a positive sentiment. Use the following template:\n\ntext:[review1]\ntext:[review2]\n\nand so on.
        self.done = False
    
    def get_apply_chat_template(self):
        return True
    
    def get_clean_gpu_memory_before_finalize(self):
        return False
        
    def get_system_prompt(self, preffered_batch_size):
        return [''] * 10

    def get_user_prompt(self, preffered_batch_size):
        self.done = True
        return [self.description] * 10
    
    def get_assistant_prompt(self, preffered_batch_size):
        return [''] * 10
    
    def is_finished(self):
        return self.done or self.eval_only
    
    def process_results(self, llm_generations, think_parts, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for i in range(len(llm_generations)):
            llm_generations[i] = llm_generations[i]
        self.result = llm_generations

        self.done = True
        
    def finalize(self, save_path = None, concept = None):
        for r in self.result:
            print(r)
            print('--------------------------------')
                
    def get_unique_name(self):
        return 'simple'
    
    def get_max_len(self):
        return 256

    def get_class_labels(self):
        return [' I', ' you'], 0, 5
    
    def get_progress(self):
        return 1


from my_datasets.emotion_eval import EmotionDataset

# decision
from my_datasets.a_ultimatum_game import UltimatumDataset
from my_datasets.a_blame_policy import BlamePolicyDataset
from my_datasets.a_dictator_game import DictatorDataset
from my_datasets.a_endowment import EndowmentDataset
from my_datasets.a_endowment_diagnostics import EndowmentDiagnosticsDataset
from my_datasets.a_intertemporal_choice import IntertemporalDataset
from my_datasets.a_loss_prob_weight import LossProbWeightDataset
from my_datasets.a_risk_lottery import RiskPerceptionDataset
from my_datasets.a_risk_curve import RiskCurveDataset
from my_datasets.a_risk_v_ambiguity import RiskAmbiguityDataset
from my_datasets.a_moral_severity import MoralSeverityDataset
from my_datasets.a_heuristic_stereotype import HeuristicStereotypeDataset
from functools import partial
from my_datasets.a_persuasion import EmotionMatchPersuasionDataset, GainLossPersuasionDataset
from my_datasets.a_welfare_assistance import WelfareAssistanceDataset
# from my_datasets.mmlu import MMLUEvaluator
from my_datasets.a_rationality import RationalityEvaluator

dataset_map = {
    'simple': SimpleDataset,
    
    'emotion': partial(EmotionDataset, gpt_eval_model='online'),
    'emotion_offline': partial(EmotionDataset, gpt_eval_model='offline'),
    
    "ultimatum-proposer": lambda eval_only: UltimatumDataset(role="proposer", eval_only=eval_only),
    "ultimatum-receiver": lambda eval_only: UltimatumDataset(role="receiver", eval_only=eval_only),
    "dictator": lambda eval_only: DictatorDataset(eval_only=eval_only),
    "endowment": lambda eval_only: EndowmentDataset(eval_only=eval_only),
    "endowment_diagnostics": lambda eval_only: EndowmentDiagnosticsDataset(eval_only=eval_only),
    "intertemporal": lambda eval_only: IntertemporalDataset(eval_only=eval_only),
    "loss_prob_weight": lambda eval_only: LossProbWeightDataset(eval_only=eval_only),
    "risk_lottery": lambda eval_only: RiskPerceptionDataset(eval_only=eval_only),
    "risk_curve": lambda eval_only: RiskCurveDataset(eval_only=eval_only),
    "risk_v_ambiguity": lambda eval_only: RiskAmbiguityDataset(eval_only=eval_only),
    "moral_severity": lambda eval_only: MoralSeverityDataset(eval_only=eval_only),
    "blame_policy": lambda eval_only: BlamePolicyDataset(eval_only=eval_only),
    "heuristic_stereotype": lambda eval_only: HeuristicStereotypeDataset(eval_only=eval_only),
    "emotion_matching_persuasion": lambda eval_only: EmotionMatchPersuasionDataset(eval_only=eval_only),
    "gain_loss_persuasion": lambda eval_only: GainLossPersuasionDataset(eval_only=eval_only),
    "welfare_assistance": lambda eval_only: WelfareAssistanceDataset(eval_only=eval_only),
    "rationality": lambda eval_only: RationalityEvaluator(eval_only=eval_only),
}

from dataset_loader import classes_to_labels


synthetic_data_generation_datasets = classes_to_labels.keys()

# %%

if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    print("The model has a chat template configured.")
    # print(f"Chat template: {tokenizer.chat_template}")
    chat_enabled = True
else:
    chat_enabled = False
    print("The model does not have an explicit chat template configured.")
    if args.thinking:
        raise ValueError("Thinking mode requires a chat template in the tokenizer.")


# %%
if chat_enabled:
    messages_with_system = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
        ]

    try:
        # Attempt to apply the template with a system message
        
        formatted_input = tokenizer.apply_chat_template(messages_with_system, tokenize=False)
        print("System role is likely supported.")
        system_role_supported = True
        # You can also inspect `tokenizer.chat_template` if it's explicitly defined
        # print(tokenizer.chat_template)
    except Exception as e:
        
        if "System role not supported" in str(e):
            print("System role is not supported by this model's tokenizer.")
            system_role_supported = False
        else:
            print(f"An error occurred: {e}")


# %%
def generate_thinking(model, tokenizer, prompts, args, thinking_budge = 1000):
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
        len_original_prompts = [len(x) for x in inputs.input_ids]
        
        if args.steer_type == 'Intervention' and args.steer_tokens in ['think', 'new']:
            model.set_steer_fn(steer_fn)
        
        generation = model.generate(**inputs, max_new_tokens=thinking_budge, do_sample = not args.argmax, temperature=args.temperature, 
                                    return_dict_in_generate=True, output_scores=True, use_cache=args.use_kv_cache, disable_compile=True, stop_strings=[stop_thinking_string], tokenizer=tokenizer)
        
        
        think_parts = []
        for b in range(inputs.input_ids.shape[0]):
            think_part = tokenizer.decode(generation.sequences[b, len_original_prompts[b]:], skip_special_tokens=False)
            if stop_thinking_string in think_part:
                think_part = think_part.split(stop_thinking_string)[0] + stop_thinking_string
            else: # for the case the model does not output the stop string
                think_part = think_part + ' ' + stop_thinking_string
            think_parts.append(think_part)
            # print('Generated thinking parts:', think_part)
        # assert False, "Fix thinking generation decoding"
        return think_parts


# %%

dataset = dataset_map[args.dataset](eval_only=args.eval_only)
last_progress = 0
max_tokenized_len = 0

with tqdm(total=100, unit="iteration", desc=f"Running dataset {dataset.get_unique_name()}") as pbar:
    while not dataset.is_finished():
            
        user_prompts = dataset.get_user_prompt(args.bs)
        system_prompts = dataset.get_system_prompt(args.bs)
        assistant_prompts = dataset.get_assistant_prompt(args.bs)
        
        if not isinstance(user_prompts, list):
            user_prompts = [user_prompts]
        
        if not isinstance(system_prompts, list):
            system_prompts = [system_prompts]
        
        if not isinstance(assistant_prompts, list):
            assistant_prompts = [assistant_prompts]
        
        assert len(user_prompts) == len(system_prompts) == len(assistant_prompts), f"User prompts: {len(user_prompts)}, System prompts: {len(system_prompts)}, Assistant prompts: {len(assistant_prompts)}"
            
        
        new_prompts = []
        for user_prompt, system_prompt, assistant_prompt in zip(user_prompts, system_prompts, assistant_prompts):
            if system_prompt != '':
                system = system_prompt + ' ' + system_prompt_suffix
            else:
                system = system_prompt_suffix

            if chat_enabled:
                if system_role_supported:
                    chat = [{'role': 'system', 'content': system},
                            {'role': 'user', 'content': user_prompt},
                            ]
                else:
                    system_ = f'{system}\n\n' if system != '' else ''
                    chat = [
                            {'role': 'user', 'content': system_ + user_prompt},
                            ]
                
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking = args.thinking)

                if (not args.thinking) and args.model_index in thinking_models and 'olmo' in model_name.lower():
                    # olmo does not automatically clsoe the thinking tag even with enable_thinking = False
                    chat += '\n' + stop_thinking_string + '\n\n'
            
                # removing bos token from the prompt because it is added by the tokenizer again in the future tokenize call            
                bos_token = tokenizer.bos_token
                if tokenizer.bos_token:
                    prompt = prompt.replace(tokenizer.bos_token, '')
            else:
                if system != '':
                    system += '\n\n'
                if assistant_prompt != '':
                    assistant_prompt = '\n\n' + assistant_prompt
                
                prompt = system + user_prompt + assistant_prompt
            new_prompts.append(prompt)
        
        prompts = new_prompts
        
        
        if args.thinking:
            think_parts = generate_thinking(model, tokenizer, prompts, args)
        else:
            think_parts = ['' for _ in range(len(prompts))]

        for i, assistant_prompt in enumerate(assistant_prompts):
            if args.thinking:
                assistant_prompt = think_parts[i] + '\n\n' + assistant_prompt
            prompts[i] += assistant_prompt
            # print('Final prompt:', prompts[i])
            # print('-----------------------------')
                
            
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
        len_original_prompts = [len(x) for x in inputs.input_ids]

        if args.steer_type == 'Intervention' and args.steer_tokens in ['new']:
            model.set_steer_fn(steer_fn)        
        elif args.steer_type == 'Intervention' and args.steer_tokens in ['think']:
            model.set_steer_fn(default_steer_fn)
        
        generation = model.generate(**inputs, max_new_tokens=dataset.get_max_len(), do_sample = not args.argmax, temperature=args.temperature, 
                                    return_dict_in_generate=True, output_scores=True, use_cache=args.use_kv_cache, disable_compile=True)
        
        cropped_generations = []
        full_response = []
        for b in range(inputs.input_ids.shape[0]):
            cropped_generations.append(tokenizer.decode(generation.sequences[b, len_original_prompts[b]:], skip_special_tokens=True))
            full_response.append(tokenizer.decode(generation.sequences[b, :], skip_special_tokens=False))
            
            # print('Final Cropped generation:', cropped_generations[-1])
            # print('*****************************')
        
        
        ########## Extracting logprobs and topk tokens
        topk_tokens = []
        topk_logprobs = []        
        
        class_labels, t0, k = dataset.get_class_labels()
        
        generation_scores = generation.scores[t0]
        for b in range(inputs.input_ids.shape[0]):
            logprobs = F.log_softmax(generation_scores[b], dim=-1)
            topk = torch.topk(logprobs, k, dim=-1)
            topk_tokens.append(tokenizer.batch_decode(topk.indices.cpu(), skip_special_tokens=False))
            topk_logprobs.append(topk.values.cpu().tolist())

        tokenized_class_labels = []
        for c in class_labels:
            t = tokenizer.encode(c, add_special_tokens=False, return_tensors='pt')
            if t.shape[1] > 1:
                print(f"Warning: Class label {c} is longer than 1 token. Using only the first token.")
            t = t[:, 0]
            tokenized_class_labels.append(t.cpu().tolist())

        logprobs = torch.log_softmax(generation_scores, dim=-1)
        class_label_logprobs = logprobs[:, tokenized_class_labels].cpu().tolist()
        
        think_parts_processed = [t.split(stop_thinking_string)[0].split(start_thinking_string)[-1].strip() if args.thinking else '' for t in think_parts]
        
        dataset.process_results(llm_generations=cropped_generations, think_parts=think_parts_processed, full_prompt=full_response, topk_tokens=topk_tokens, topk_logprobs=topk_logprobs, target_logprobs=class_label_logprobs)
        torch.cuda.empty_cache()
        
        pbar.update(100 * (dataset.get_progress() - last_progress))
        last_progress = dataset.get_progress()

    pbar.update((dataset.get_progress() - last_progress) == 1.0)

if dataset.get_clean_gpu_memory_before_finalize():
    try:
        del model, inputs, generation, tokenizer
        del generation_scores, logprobs, topk, topk_logprobs, class_label_logprobs
        del steer_vecs, steer_vec, steer_fn
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()

dataset.finalize(save_path=save_dir)
print('Results saved to:', save_dir)
