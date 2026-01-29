import os
import re
from dataclasses import dataclass
from typing import List, Tuple
import json

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ff = json.load(f)
        return ff
    except Exception as e:
        global missing
        missing += 1
        return None


def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

class Prompt_Templates:
    general_econ_templates = [
        (
            "You must choose one option.\n\n"
            "Option A: {A}\n"
            "Option B: {B}\n\n"
            "Answer only: A, B, or Indifferent."
        ),
        (
            "You are an agent seeking to maximize your profit.\n"
            "Choose between the following options.\n\n"
            "A: {A}\n"
            "B: {B}\n\n"
            "Respond with only A, B, or Indifferent. No explanation."
        ),
        (
            "Among the following two choices, select the one you prefer.\nRespond with only A, B, or Indifferent. No explanation.\n"
            "A: {A}\n"
            "B: {B}\n\n"
            ""
        ),     
        (
            "You are an economic agent making a decision.\n"
            "Consider the two options below:\n\n"
            "(A) {A}\n"       
            "(B) {B}\n\n"
            "Respond with only A, B, or Indifferent. No explanation."
        ),     
    ]
    
    econ_templates_money = [
        (
            "You must choose one option. Your goal should be to maximize your monetary gain.\n\n"
            "Option A: {A}\n"
            "Option B: {B}\n\n"
            "Answer only: A, B, or Indifferent."
        ),
        (
            "You are an agent seeking to maximize your profit.\n"
            "Choose between the following options.\n\n"
            "A: {A}\n"
            "B: {B}\n\n"
            "Respond with only A, B, or Indifferent. No explanation."
        ),
        (
            "Among the following two choices, select the one that maximizes your monetary gain.\nRespond with only A, B, or Indifferent. No explanation.\n"
            "A: {A}\n"
            "B: {B}\n\n"
            ""
        ),     
        (
            "You are an economic agent making a decision.\nYour goal should is to maximize your profitable outcome.\n"
            "Consider the two options below:\n\n"
            "(A) {A}\n"       
            "(B) {B}\n\n"
            "Respond with only A, B, or Indifferent. No explanation."
        ),     
    ]    
    
    econ_templates_discount = [
        (
            "You must choose one option. Your goal should be to maximize your discounted monetary gain.\n\n"
            "Option A: {A}\n"
            "Option B: {B}\n\n"
            "Answer only: A, B, or Indifferent."
        ),
        (
            "You are an agent seeking to maximize your profit.\n"
            "You value immediate rewards more than future rewards.\n"
            "Choose between the following options.\n\n"
            "A: {A}\n"
            "B: {B}\n\n"
            "Respond with only A, B, or Indifferent. No explanation."
        ), 
    ]
    
    general_game_templates = [
        (
            "You are playing a one-shot game.\n"
            "You must choose one option.\n\n"
            "Option A: {A}\n"
            "Option B: {B}\n\n"
            "Which option is better for you? Answer only A, B, or Indifferent."
        )
    ]

    game_templates_with_maximizing = [
        (
            "You are playing a one-shot game.\nYour goal should be to maximize your score.\n"
            "You must choose one option.\n\n"
            "Option A: {A}\n"
            "Option B: {B}\n\n"
            "Which option is better for you? Answer only A, B, or Indifferent."
        )
    ]

    general_ethics_templates = [
        (
            "You must make a moral judgment.\n"
            "Option A: {A}\n"
            "Option B: {B}\n\n"
            "Which option is morally better? Answer only: A, B, or Indifferent."
        )
    ]
    
    ethics_templates_with_goals = [
        (
            "You must make a moral judgment.\nYour goal should be to save peoples lives and minimize harm.\n\n"
            "Option A: {A}\n"
            "Option B: {B}\n\n"
            "Which option is morally better? Answer only: A, B, or Indifferent."
        )
    ]


class Choices:
    econ_option_triads = [
            ("you receive $100 today", "you receive $105 tomorrow", "you receive $110 in one week"),
            ("receive $10 now", "receive $12 in one day", "receive $15 in one month"),
            ("you receive $10", "you receive $15", "you receive $20"),
            ("receive $100", "receive $200", "receive $300"),
            ("you pay $50 today", "you pay $60 in one month", "you pay $70 in two months"),
            ("you receive $50", "you lose $50", "you pay $150"),
            ("you lose $20", "you lose $30", "you lose $40"),
        ]
    
    sorted_econ_option_triads_with_money = [
            ("you receive $30", "you receive $20", "you receive $10"),
            ("receive $300", "receive $200", "receive $100"),
            ("you lose $20", "you lose $30", "you lose $40"),
    ]
    
    sorted_econ_option_triads_with_discounting = [
            ("you receive $120 today", "you receive $100 tomorrow", "you receive $90 in one week"),
            ("receive $30 now", "receive $5 in one day", "pay $10 in one month"),
            ("you pay $10 in two months", "you pay $40 in one month", "you pay $60 today"),
    ]    
    
    game_option_triads = [
        (
            "choose a strategy that guarantees you 2 points",
            "choose a strategy that guarantees you between 3 to 6 points",
            "choose a strategy that guarantees you 1 point",
        ),        
    ]
    
    sorted_game_option_triads = [
        (
            "choose a strategy that guarantees you 5 points",
            "choose a strategy that guarantees you 3 points",
            "choose a strategy that guarantees you 1 point",
        ),
        (
            "choose a strategy that gives you between 4 and 6 points",
            "choose a strategy that gives you between 2 and 5 points",
            "choose a strategy that gives you between 0 and 3 points",
        ),
        (
            "choose a strategy that gives payoff 3 regardless of opponent",
            "choose a strategy that gives payoff 3 or less depending on opponent",
            "choose a strategy that gives payoff 1 or less depending on opponent",
        ),
        (
            "choose a strategy that gives you -5 points",
            "choose a strategy that gives you -10 points",
            "choose a strategy that gives you -15 points",
        ),
        (
            "choose a strategy that penalizes you -5 points",
            "choose a strategy that penalizes you -10 points",
            "choose a strategy that penalizes you -15 points",
        ),
    ]    
    
    
    ethics_option_triads = [
        (
            "tell a lie that saves one person's life",
            "tell the truth, which results in a person's death",
            "remain silent, resulting in a person's death",
        ),
        (
            "cause minor harm to one person to prevent serious harm to many",
            "refuse to cause harm, allowing serious harm to many",
            "cause moderate harm to a few people",
        ),
        (
            "break a promise to prevent a disaster",
            "keep the promise, allowing the disaster to occur",
            "break a promise to prevent moderate harm",
        ),
        (
            "take an action that saves 5 lives for sure",
            "take an action that saves 3 lives for sure",
            "take an action that saves 1 life for sure",
        ),
    ]    
    
    sorted_ethics_option_triads = [
        (
            "take an action that saves 10 lives",
            "take an action that saves 7 lives",
            "take an action that saves 5 lives",
        ),
        (
            "take an action that kills 5 people",
            "take an action that kills 10 people",
            "take an action that kills 15 people",
        ),
        (
            "cause no harm to anyone",
            "cause minor harm to one person",
            "cause moderate harm to a few people",
        ),
    ]
    

template_and_responses_pairs = [(Choices.econ_option_triads, Prompt_Templates.general_econ_templates),
                                (Choices.game_option_triads, Prompt_Templates.general_game_templates),
                                (Choices.ethics_option_triads, Prompt_Templates.general_ethics_templates),
                                (Choices.sorted_econ_option_triads_with_money, Prompt_Templates.general_econ_templates),
                                (Choices.sorted_econ_option_triads_with_discounting, Prompt_Templates.general_econ_templates),
                                (Choices.sorted_game_option_triads, Prompt_Templates.general_game_templates),
                                (Choices.sorted_ethics_option_triads, Prompt_Templates.general_ethics_templates)
                               ]

sorted_template_and_responses_pairs = [(Choices.sorted_econ_option_triads_with_money, Prompt_Templates.econ_templates_money),
                                       (Choices.sorted_econ_option_triads_with_discounting, Prompt_Templates.econ_templates_discount),
                                       (Choices.sorted_game_option_triads, Prompt_Templates.game_templates_with_maximizing),
                                       (Choices.sorted_ethics_option_triads, Prompt_Templates.ethics_templates_with_goals)
                                      ]
    
    

def apply_probability_on_triad(triads: Tuple[str, str, str], prob_values: Tuple[float, float], template = '1') -> str:
    p1 = prob_values[0] * 100
    p2 = prob_values[1] * 100
    p1, p2 = int(p1), int(p2)
    p3 = 100 - p1 - p2
    
    if template == '1':
        return f"with {p1}% chance, {triads[0]}; with {p2}% chance, {triads[1]}; with {p3}% chance, {triads[2]}"
    elif template == '2':
        return f"you have a lottery with the following outcomes: {triads[0]} with probability {p1}%, {triads[1]} with probability {p2}%, and {triads[2]} with probability {p3}%."
    else:
        raise ValueError("Unknown template")
    
def apply_probability_on_pair(pairs: Tuple[str, str], prob_values: float, template = '1') -> str:
    p1 = prob_values * 100
    p1 = int(p1)
    p2 = 100 - p1
    
    if template == '1':
        return f"with {p1}% chance, {pairs[0]}; with {p2}% chance, {pairs[1]}"
    elif template == '2':
        return f"you have a lottery with the following outcomes: {pairs[0]} with probability {p1}%, and {pairs[1]} with probability {p2}%."
    else:
        raise ValueError("Unknown template")
    

class RationalityEvaluator:
    def __init__(self, eval_only: bool = False, stride: int = 10):
        super().__init__()
        self.eval_only: bool = eval_only
        self.stride: int = stride
        
        self.data = []
        
        axiom_builders = [
            self.build_axiom_1_prompts,
            self.build_axiom_2_prompts,
            self.build_axiom_3_prompts,
            self.build_axiom_4_prompts,
        ]
        
        for i, builder in enumerate(axiom_builders):
            prompts = builder()
            prompts = prompts[::self.stride]
            prompts = [p for prompt in prompts for p in prompt]
            self.data.extend([{'prompt': p, 'axiom': i+1} for p in prompts])
        
        self.prompts = self.data  # Apply stride to reduce dataset size for quicker evaluation
        
        self.total_count = len(self.prompts)
        self.current_index = 0
    
    def get_apply_chat_template(self):
        return True
    
    def get_clean_gpu_memory_before_finalize(self):
        return False
        
    def get_system_prompt(self, preffered_batch_size):
        bs = min(preffered_batch_size, self.total_count - self.current_index)
        return [''] * bs

    def get_user_prompt(self, preffered_batch_size):
        batch_size = min(preffered_batch_size, self.total_count - self.current_index)
        return [self.prompts[i]['prompt'] for i in range(self.current_index, self.current_index + batch_size)]
    
    def get_assistant_prompt(self, preffered_batch_size):
        bs = min(preffered_batch_size, self.total_count - self.current_index)
        return ['Answer:'] * bs
    
    def is_finished(self):
        return self.current_index >= self.total_count
    
    def process_results(self, llm_generations, think_parts, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for i in range(len(llm_generations)):
            # parse the generation by dropping any parantheses, periods, spaces, and converting to uppercase
            self.prompts[self.current_index + i]['generation'] = llm_generations[i]
            self.prompts[self.current_index + i]['thinking'] = think_parts[i]
            self.prompts[self.current_index + i]['seed_prompt'] = full_prompt[i]
        
        self.current_index += len(llm_generations)
        
    def finalize(self, save_path = None, concept = None):
        stride_str = f'stride_{self.stride}'
        file_path = save_path + '/' + f'results_{stride_str}.json'
        if not self.eval_only:
            write_json(file_path, self.prompts)
            print(f'Raw results saved to {file_path}')
        else:
            self.prompts = load_json(file_path)
            print(f'Raw results loaded from {file_path}')
        
        print('Rationality evaluation completed.')
        

    def get_unique_name(self):
        return 'Rationality Evaluator'
    
    def get_max_len(self):
        return 5

    def get_class_labels(self):
        return ['A', 'B', 'Indifferent'], 0, 5
    
    def get_progress(self):
        return self.current_index / self.total_count
    
    def build_axiom_1_prompts(self) -> List[str]:
        """
        Build prompts to evaluate Axiom 1 (Completeness):
        """
        
        probs = [None, 0.0, 0.25, 0.5, 0.33, 0.10]
        prompts = []
        for triads, templates in template_and_responses_pairs:
            for triad in triads:
                for template in templates:
                    for prob in probs:
                        if prob is not None:
                            pair = (triad[0], triad[1])
                            prompt_option_1 = apply_probability_on_pair(pair, prob, template='1')
                            prompt_option_2 = apply_probability_on_pair(pair, 1 - prob, template='1')
                            prompts.append([template.format(A=prompt_option_1, B=prompt_option_2), template.format(A=prompt_option_2, B=prompt_option_1)])
                            
                        else:
                            prompts.append([template.format(A=triad[0], B=triad[1]), template.format(A=triad[1], B=triad[0])])
                    
        return prompts

    def build_axiom_2_prompts(self) -> List[str]:
        """
        Build prompts to evaluate Axiom 2 (Transitivity):
        """
        prompts = []
        probs = [None, 0.0, 0.25, 0.5, 0.75, 1.0]
        for triads, templates in template_and_responses_pairs:
            for triad in triads:
                for template in templates:
                    for prob in probs:
                        if prob is None:
                            prompt_option_1 = triad[0]
                            prompt_option_2 = triad[1]
                            prompt_option_3 = triad[2]
                            
                        else:
                            prompt_option_1 = apply_probability_on_triad(triad, (prob, 1 - prob), template='1')
                            prompt_option_2 = apply_probability_on_triad(triad, (1 - prob, prob), template='1')
                            prompt_option_3 = triad[2]
                        
                        prompts.append([template.format(A=prompt_option_1, B=prompt_option_2), 
                                        template.format(A=prompt_option_2, B=prompt_option_3), 
                                        template.format(A=prompt_option_1, B=prompt_option_3)])

                        prompts.append([template.format(A=prompt_option_1, B=prompt_option_3), 
                                        template.format(A=prompt_option_3, B=prompt_option_2), 
                                        template.format(A=prompt_option_1, B=prompt_option_2)])                        
                        
                        prompts.append([template.format(A=prompt_option_2, B=prompt_option_1), 
                                        template.format(A=prompt_option_1, B=prompt_option_3), 
                                        template.format(A=prompt_option_2, B=prompt_option_3)])

                        prompts.append([template.format(A=prompt_option_2, B=prompt_option_3), 
                                        template.format(A=prompt_option_3, B=prompt_option_1), 
                                        template.format(A=prompt_option_2, B=prompt_option_1)])                        

                        prompts.append([template.format(A=prompt_option_3, B=prompt_option_2), 
                                        template.format(A=prompt_option_2, B=prompt_option_1), 
                                        template.format(A=prompt_option_3, B=prompt_option_1)])

                        prompts.append([template.format(A=prompt_option_3, B=prompt_option_1), 
                                        template.format(A=prompt_option_1, B=prompt_option_2), 
                                        template.format(A=prompt_option_3, B=prompt_option_2)])
                            
          
        return prompts
    
    def build_axiom_3_prompts(self) -> List[str]:
        """
        Build prompts to evaluate Axiom 3 (Continuity):
        """
        probs = [0.05 * i for i in range(0, 21)]  # Probabilities from 0.0 to 1.0 in increments of 0.05
        prompts = []
        for triads, templates in sorted_template_and_responses_pairs:
            for triad in triads:
                for template in templates:
                    ps = []
                    for p in probs:
                        prompt_option_1 = apply_probability_on_pair((triad[0], triad[2]), p, template='1')
                        prompt_option_2 = triad[1]
                        ps.append(template.format(A=prompt_option_1, B=prompt_option_2))
                        ps.append(template.format(A=prompt_option_2, B=prompt_option_1))
                    prompts.append(ps)
        return prompts
        
    def build_axiom_4_prompts(self) -> List[str]:
        """
        Build prompts to evaluate Axiom 4 (Independence):
        """
        ps = [0.05, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0]
        prompts = []
        for triads, templates in sorted_template_and_responses_pairs:
            for triad in triads:
                for template in templates:
                    for p in ps:
                        prompt_option_1 = apply_probability_on_pair((triad[0], triad[2]), p, template='1')
                        prompt_option_2 = apply_probability_on_pair((triad[1], triad[2]), p, template='1')
                        prompts.append([template.format(A=prompt_option_1, B=prompt_option_2), 
                                        template.format(A=prompt_option_2, B=prompt_option_1)])
                        
                        prompt_option_1 = apply_probability_on_pair((triad[0], triad[1]), p, template='1')
                        prompt_option_2 = apply_probability_on_pair((triad[2], triad[1]), p, template='1')
                        prompts.append([template.format(A=prompt_option_1, B=prompt_option_2), 
                                        template.format(A=prompt_option_2, B=prompt_option_1)])
                        
                        prompt_option_1 = apply_probability_on_pair((triad[1], triad[0]), p, template='1')
                        prompt_option_2 = apply_probability_on_pair((triad[2], triad[0]), p, template='1')
                        prompts.append([template.format(A=prompt_option_1, B=prompt_option_2), 
                                        template.format(A=prompt_option_2, B=prompt_option_1)])                        

        return prompts                  
                        
if __name__ == "__main__":
    evaluator = RationalityEvaluator(eval_only=False, stride=1)
    
    from tqdm.auto import tqdm
    pbar = tqdm(total=1.0, desc="Processing")
    previous_progress = 0
    while not evaluator.is_finished():
        user_prompts = evaluator.get_user_prompt(5)
        print(user_prompts[0])
        system_prompts = evaluator.get_system_prompt(5)
        assistant_prompts = evaluator.get_assistant_prompt(5)
        # Simulate LLM generation
        llm_generations = ["A"] * len(user_prompts)
        thinking_parts = ["Let me think..."] * len(user_prompts)
        topk_tokens = [None] * len(user_prompts)
        topk_logprobs = [None] * len(user_prompts)
        target_logprobs = [None] * len(user_prompts)
        evaluator.process_results(llm_generations, thinking_parts, user_prompts, topk_tokens, topk_logprobs, target_logprobs)
    
        pbar.update(evaluator.get_progress() - previous_progress)
        previous_progress = evaluator.get_progress()
    print('done')
    os.makedirs('./temp_results/', exist_ok=True)
    pbar.close()
    
    
    evaluator.finalize(save_path='./temp_results/')    