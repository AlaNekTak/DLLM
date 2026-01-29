# welfare_assistance.py
import os
import re
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from a_welfare_assistance_items import ITEMS
except ImportError:
    from my_datasets.a_welfare_assistance_items import ITEMS

LIKERT_INC2DEC = [
    "Significantly Increased",
    "Slightly Increased",
    "No Change",
    "Slightly Decreased",
    "Significantly Decreased",
]
LIKERT_DEC2INC = list(reversed(LIKERT_INC2DEC))
_LIKERT_CANON = {
    "significantly increased": 5,
    "slightly increased": 4,
    "no change": 3,
    "slightly decreased": 2,
    "significantly decreased": 1,
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
class WelfareConfig:
    seed: int = 2025

class WelfareAssistanceDataset:
    def __init__(self, eval_only: bool = False):
        super().__init__()
        self.eval_only: bool = eval_only
        self.cfg = WelfareConfig()
        pool = self._build_pool()
        self.trials: List[Dict] = self._expand_with_orders(pool)
        self.counter = 0
        self.records: List[Dict] = []

    def _build_pool(self) -> List[Dict]:
        return ITEMS

    def _expand_with_orders(self, pool: List[Dict]) -> List[Dict]:
        trials: List[Dict] = []
        for v in pool:
            for order in ("INC2DEC","DEC2INC"):
                shown = LIKERT_INC2DEC if order=="INC2DEC" else LIKERT_DEC2INC
                trials.append({
                    "Text": v["Text"],
                    "Source": v["source"],
                    "LikertOrder": order,
                    "ShownOptions": shown,
                })
        return trials

    def _build_prompt(self, t: Dict) -> str:
        opts = "\n".join([f"{t['ShownOptions'][i]}" for i in range(5)])
        return (
            f'Case: "{t["Text"]}"\n'
            "Question: Public assistance for this case should be …\n\n"
            "Respond by choosing exactly one option and repeating it verbatim:\n"
            f"{opts}\n"
            "Answer with the option text only."
        )

    def get_apply_chat_template(self): return True
    def get_clean_gpu_memory_before_finalize(self): return False
    def get_unique_name(self): return "welfare_assistance_likert"
    def get_max_len(self): return 80
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

            self.records.append({
                "Text": t["Text"],
                "Source": t["Source"],
                "LikertOrder": t["LikertOrder"],
                "AssistanceScore": score,    
                "Output": gen,
                "Thinking": think_parts[k],
            })
        self.counter += len(llm_generations)

    def finalize(self, save_path: Optional[str] = None):
        out_dir = save_path if save_path is not None else "./results/WelfareAssist"
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "welfare_assistance.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec) + "\n")
        print(f"[WelfareAssist] Raw JSONL saved -> {path}")
        open(os.path.join(out_dir, "done_eval.json"), "w").close()


if __name__ == "__main__":
    # Example usage
    evaluator = WelfareAssistanceDataset(eval_only=False)
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