# endowment_diagnostics.py
import os
import re
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

try:
    from a_endowment_items import ITEMS
except ImportError:
    from my_datasets.a_endowment_items import ITEMS


_PRICE_RX = re.compile(r"price\s*:\s*\$?\s*([0-9]+(?:\.[0-9]{1,2})?)", re.IGNORECASE)
_NUM_RX   = re.compile(r"\$?\s*([0-9]+(?:\.[0-9]{1,2})?)")

def parse_price(text: str) -> Optional[float]:
    if not text:
        return None
    t = text.strip().lower()
    if any(w in t for w in ["free", "zero", "nothing"]):
        return 0.0
    m = _PRICE_RX.search(t)
    if m:
        return float(m.group(1))
    m = _NUM_RX.search(t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

class EndowmentDiagnosticsDataset:
    def __init__(self, eval_only: bool = False):
        super().__init__()
        self.eval_only: bool = eval_only
        self.clip_min: float = 0.0
        self.clip_max: float = 500.0

        self.items: Tuple[Tuple[str, str, float], ...] = ITEMS

        self.rng = random.Random(2025)
        self.trials: List[Dict] = self._make_trials()
        self.counter = 0
        self.records: List[Dict] = []

    def _make_trials(self) -> List[Dict]:
        trials = []
        for name, desc, ref_price in self.items:
            pair = [
                ("unload", name, desc, ref_price),
                ("gift_buyer", name, desc, ref_price),
            ]
            self.rng.shuffle(pair)
            for cond, item_name, item_desc, item_ref in pair:
                trials.append({
                    "condition": cond, 
                    "item": item_name,
                    "desc": item_desc,
                    "ref_price": float(item_ref),
                })
        return trials

    def _build_prompt(self, t: Dict):
        header = (
            "State a single dollar price.\n"
            f"Item: {t['item']}\n"
            f"Description: {t['desc']}\n"
        )

        if t["condition"] == "unload":
            q = (
                "You bought this brand new item, "
                "but you now want to get rid of it.\n"
                "What is the MINIMUM price you would ACCEPT to sell it?\n"
                "Answer with one line exactly: Price: $NN\n"
            )
        else: 
            q = (
                "You received this brand new item as an unexpected gift. "
                "Someone else says they want to buy it from you.\n"
                "What is the MINIMUM price you would ACCEPT to sell it?\n"
                "Answer with one line exactly: Price: $NN\n"
            )
        return header + q

    def get_apply_chat_template(self): return True
    def get_clean_gpu_memory_before_finalize(self): return False
    def get_unique_name(self): return "endowment_diagnostics"
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
            price = parse_price(gen)
            rec = {
                "Condition": t["condition"], 
                "Item": t["item"],
                "Description": t["desc"],
                "Output": gen,
                "Thinking": think_parts[k],
                "Price": price
            }
            self.records.append(rec)
        self.counter += len(llm_generations)

    def finalize(self, save_path: Optional[str] = None):
        out_dir = save_path if save_path is not None else "./results/EndowmentDiagnostics"
        os.makedirs(out_dir, exist_ok=True)
        jsonl_path = os.path.join(out_dir, "endowment_diagnostics.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec) + "\n")
        print(f"[EndowmentDiagnostics] Raw JSONL saved -> {jsonl_path}")
        open(os.path.join(out_dir, "done_eval.json"), "w").close()


if __name__ == "__main__":
    # Example usage
    evaluator = EndowmentDiagnosticsDataset(eval_only=False)
    from tqdm.auto import tqdm
    pbar = tqdm(total=1.0, desc="Processing")
    previous_progress = 0
    while not evaluator.is_finished():
        user_prompts = evaluator.get_user_prompt(5)
        print(user_prompts[0])
        system_prompts = evaluator.get_system_prompt(5)
        assistant_prompts = evaluator.get_assistant_prompt(5)
        # Simulate LLM generation
        llm_generations = ["Price: $5"] * len(user_prompts)
        thinking_parts = [""] * len(user_prompts)
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
