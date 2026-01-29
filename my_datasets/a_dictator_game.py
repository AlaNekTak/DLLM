import os
import re
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

_MONEY_RX = re.compile(r"\$?\s*([0-9]+(?:\.[0-9]+)?)")

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _closest_allowed(x: float, allowed: List[float]) -> float:
    return min(allowed, key=lambda a: abs(a - x)) if allowed else x

def parse_amount_from_text(text: str, allowed_amounts: List[float]) -> Optional[float]:
    if not text:
        return None
    t = text.strip().lower()

    if "all" in t or "everything" in t:
        return max(allowed_amounts) if allowed_amounts else None
    if "nothing" in t or "zero" in t:
        return min(allowed_amounts) if allowed_amounts else None
    if "half" in t:
        half = 0.5 * (max(allowed_amounts) if allowed_amounts else 0.0)
        return _closest_allowed(half, allowed_amounts)

    m = _MONEY_RX.search(t)
    if m:
        val = float(m.group(1))
        return _closest_allowed(val, allowed_amounts)

    return None

def dollars(vals: List[float]) -> List[str]:
    return [f"${int(v) if v.is_integer() else v:.0f}" for v in vals]

def to_percent(v: float) -> float:
    return 100.0 * v


class DictatorDataset:
    def __init__(self, eval_only: bool = False):
        super().__init__()
        self.role: str = "dictator"
        self.funds: Tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.eval_only: bool = eval_only

        self.data: List[dict] = []
        self._build_prompts()

        self.counter = 0
        self.detailed_results: List[dict] = []

    def _build_prompts(self):
        templates = [
            (
                "You have ${total}.\n"
                "You may give any integer dollar amount to the other player; whatever you don't give, you keep.\n"
                "How much will you give?\n"
                "Pick one amount from: [{opts_str}].\n"
            ),
            (
                "You control ${total} for a one-shot allocation.\n"
                "Any dollars you GIVE go to the other player; you KEEP the remainder.\n"
                'Answer with the dollar amount you GIVE only.'
            ),
            (
                "You have a total of ${total}.\n"
                "You may give any integer dollar amount to the other player;\n"
                "Choose an integer GIVE amount for the other player; you KEEP the remainder.\n"
                "Respond with just the dollar amount you GIVE.\n"
            ),
            (
                "Dictator Game (one-shot).\n"
                "You hold ${total} to allocate between yourself and the other player.\n"
                "Choose an integer GIVE amount for the other player from: [{opts_str}].\n"
                "Reply with the dollar amount you GIVE only."
            ),
            (
                "Single-round allocation: you have ${total}.\n"
                "You can transfer any whole-dollar amount to the other player (you keep the rest).\n"
                "Enter only the amount you GIVE in dollars."
            ),
        ]

        for total in self.funds:
            allowed = list(range(0, total + 1)) 

            pair_strs = [f"${g}" for g in allowed]
            opts_str = ", ".join(pair_strs)
            for vidx, tmpl in enumerate(templates, start=1):
                prompt = tmpl.format(total=total, opts_str=opts_str)

                self.data.append({
                    "role": "dictator",
                    "total": total,
                    "allowed_gives": allowed,  
                    "prompt": prompt,
                    "prompt_variant": vidx,  
                })

    def get_apply_chat_template(self): return True
    def get_clean_gpu_memory_before_finalize(self): return False
    def get_unique_name(self): return "dictator"
    def get_max_len(self): return 200
    def get_progress(self): return self.counter / len(self.data) if self.data else 1.0
    def get_class_labels(self):
        return [], 0, 5

    def get_system_prompt(self, preffered_batch_size):
        n = min(preffered_batch_size, len(self.data) - self.counter)
        return [""] * n

    def get_user_prompt(self, preffered_batch_size):
        n = min(preffered_batch_size, len(self.data) - self.counter)
        return [d["prompt"] for d in self.data[self.counter:self.counter + n]]

    def get_assistant_prompt(self, preffered_batch_size):
        n = min(preffered_batch_size, len(self.data) - self.counter)
        return ["Answer:"] * n

    def is_finished(self):
        return self.counter >= len(self.data) or self.eval_only

    def process_results(self, llm_generations, think_parts, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for i, gen in enumerate(llm_generations):
            idx = self.counter + i
            item = self.data[idx]
            allowed = item["allowed_gives"]
            amt = parse_amount_from_text(gen, allowed)
            pct = to_percent(amt / item["total"]) if (amt is not None and item["total"] > 0) else None

            record = {
                "Role": "dictator",
                "Total_sum": item["total"],
                "Allowed_gives": allowed,
                "Chosen_give_amount": amt,
                "Chosen_give_pct": pct,
                "Raw_output": gen,
                "Thinking": think_parts[i],
                "Prompt": item["prompt"],
            }
            self.detailed_results.append(record)

        self.counter += len(llm_generations)

    def finalize(self, save_path=None):
        out_dir = save_path if save_path is not None else "./results/Dictator"
        _ensure_dir(out_dir)

        jsonl_name = f"role-{self.role}.jsonl"
        jsonl_path = os.path.join(out_dir, jsonl_name)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in self.detailed_results:
                f.write(json.dumps(rec) + "\n")
        print(f"[Dictator] Raw JSONL saved -> {jsonl_path}")
        open(os.path.join(out_dir, "done_eval.json"), "w").close()

if __name__ == "__main__":
    # Example usage
    evaluator = DictatorDataset(eval_only=False)
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