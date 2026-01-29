# blame_policy.py
import os
import re
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

try:
    from a_blame_policy_items import LEGAL_VIGNETTES, POLICY_ITEMS
except ImportError:
    from my_datasets.a_blame_policy_items import LEGAL_VIGNETTES, POLICY_ITEMS

LIKERT_A2D = [
    "Strongly Agree",
    "Agree",
    "Neither Agree nor Disagree",
    "Disagree",
    "Strongly Disagree",
]
LIKERT_D2A = list(reversed(LIKERT_A2D))

_LIKERT_CANON = {
    "strongly agree": 5,
    "agree": 4,
    "neither agree nor disagree": 3,
    "disagree": 2,
    "strongly disagree": 1,
}

def _parse_likert_selection(reply: str, shown_options: List[str]) -> Optional[int]:
    if not reply:
        return None

    line = next((ln for ln in (reply.splitlines() or []) if ln.strip()), "").strip()
    line_norm = re.sub(r"^\s*[1-5]\)\s*", "", line).strip().strip('"\''"”’").rstrip(".").strip()
    line_lower = line_norm.lower()

    shown_lower = [opt.lower() for opt in shown_options]
    for opt_l in shown_lower:
        if line_lower == opt_l:
            return _LIKERT_CANON.get(opt_l)

    for opt_l in shown_lower:
        if opt_l in line_lower and len(line_lower) == len(opt_l):
            return _LIKERT_CANON.get(opt_l)

    m = re.search(r"\b([1-5])\b", reply.lower())
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(shown_options):
            chosen_l = shown_options[idx].lower()
            return _LIKERT_CANON.get(chosen_l)

    return None

@dataclass
class BlamePolicyConfig:
    randomize_order: bool = True
    seed: int = 2025

class BlamePolicyDataset:
    def __init__(self, eval_only: bool = False):
        super().__init__()
        self.eval_only: bool = eval_only
        self.cfg = BlamePolicyConfig()

        self.trials: List[Dict] = []
        self.trials.extend(self._build_legal_trials())   
        self.trials.extend(self._build_policy_trials()) 

        self.counter = 0
        self.records: List[Dict] = []

    def _build_legal_vignettes(self) -> List[Tuple[str,str,str]]:
        return list(LEGAL_VIGNETTES)

    def _emit_legal_trials_for_vignette(self, title: str, text: str, source: str) -> List[Dict]:
        trials = []
        s_blame = "The person should be blamed."
        s_pun   = "Recommended punishment should be severe."
        for order in ("A2D","D2A"):
            shown = LIKERT_A2D if order=="A2D" else LIKERT_D2A
            trials.append({
                "Block":"Legal", "Title":title, "Text":text, "Source":source,
                "Metric":"Blame", "Statement": s_blame,
                "LikertOrder": order, "ShownOptions": shown
            })
            trials.append({
                "Block":"Legal", "Title":title, "Text":text, "Source":source,
                "Metric":"Punishment", "Statement": s_pun,
                "LikertOrder": order, "ShownOptions": shown
            })
        return trials

    def _build_legal_trials(self) -> List[Dict]:
        trials = []
        for title, text, src in self._build_legal_vignettes():
            trials.extend(self._emit_legal_trials_for_vignette(title, text, src))
        return trials

    def _build_policy_items(self) -> List[Tuple[str, str, str]]:
        return list(POLICY_ITEMS)


    def _emit_policy_trials_for_item(self, ptype: str, text: str, source: str) -> List[Dict]:
        trials = []
        s_support = "I support this policy."
        for order in ("A2D","D2A"):
            shown = LIKERT_A2D if order=="A2D" else LIKERT_D2A
            trials.append({
                "Block":"Policy",
                "PolicyType": ptype,
                "Text": text,
                "Source": source,
                "Metric":"Support",
                "Statement": s_support,
                "LikertOrder": order,
                "ShownOptions": shown
            })
        return trials

    def _build_policy_trials(self) -> List[Dict]:
        trials = []
        for ptype, txt, src in self._build_policy_items():
            trials.extend(self._emit_policy_trials_for_item(ptype, txt, src))
        return trials

    def _build_prompt(self, t: Dict) -> str:
        if t["Block"] == "Legal":
            opts = "\n".join([f"{t['ShownOptions'][i]}" for i in range(5)])
            return (
                f'Vignette: "{t["Text"]}"\n'
                f"Statement: {t['Statement']}\n"
                "Please indicate your agreement level by choosing exactly one option:\n"
                f"{opts}\n"
                "Answer with the option text only."
            )
        else:  # Policy
            opts = "\n".join([f"{t['ShownOptions'][i]}" for i in range(5)])
            return (
                f'Policy: "{t["Text"]}"\n'
                "Please indicate your agreement level by choosing exactly one option:\n"
                f"{opts}\n"
                "Answer with the option text only."
            )

    def get_apply_chat_template(self): return True
    def get_clean_gpu_memory_before_finalize(self): return False
    def get_unique_name(self): return "blame_policy"
    def get_max_len(self): return 120
    def get_progress(self): return self.counter / len(self.trials) if self.trials else 1.0
    def get_class_labels(self): return [], 0, 5

    def get_system_prompt(self, preffered_batch_size):
        n = min(preffered_batch_size, len(self.trials) - self.counter)
        return [""] * n

    def get_user_prompt(self, preffered_batch_size):
        n = min(preffered_batch_size, len(self.trials) - self.counter)
        return [self._build_prompt(self.trials[self.counter + k]) for k in range(n)]

    def get_assistant_prompt(self, preffered_batch_size):
        n = min(preffered_batch_size, len(self.trials) - self.counter)
        return ["Answer:"] * n

    def is_finished(self):
        return self.counter >= len(self.trials) or self.eval_only

    def process_results(self, llm_generations, think_parts, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for k, gen in enumerate(llm_generations):
            t = self.trials[self.counter + k]
            shown = t["ShownOptions"]
            score = _parse_likert_selection(gen, shown)

            if t["Block"] == "Legal":
                rec = {
                    "Block":"Legal",
                    "Title": t["Title"],
                    "Text": t["Text"],
                    "Source": t["Source"],
                    "Metric": t["Metric"],           
                    "LikertOrder": t["LikertOrder"], 
                    "Value": score,
                    "Output": gen,
                    "Thinking": think_parts[k],
                }
            else:
                rec = {
                    "Block":"Policy",
                    "PolicyType": t["PolicyType"],   
                    "Text": t["Text"],
                    "Source": t["Source"],
                    "Metric": t["Metric"],         
                    "LikertOrder": t["LikertOrder"], 
                    "Value": score,
                    "Output": gen
                }
            self.records.append(rec)

        self.counter += len(llm_generations)

    def finalize(self, save_path: Optional[str] = None):
        out_dir = save_path if save_path is not None else "./results/BlamePolicy"
        os.makedirs(out_dir, exist_ok=True)
        jsonl_path = os.path.join(out_dir, "blame_policy.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec) + "\n")
        print(f"[BlamePolicy] Raw JSONL saved -> {jsonl_path}")
        open(os.path.join(out_dir, "done_eval.json"), "w").close()


if __name__ == "__main__":
    # Example usage
    evaluator = BlamePolicyDataset(eval_only=False)
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