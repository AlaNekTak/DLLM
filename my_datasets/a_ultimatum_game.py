import os
import re
import json
import random
from collections import defaultdict
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

def parse_accept_reject(text: str) -> Optional[bool]:
    if not text:
        return None
    pos = {"accept", "yes", "approve", "agree", "ok", "okay", "sure", "affirm", "approved", "agreed"}
    neg = {"reject", "no", "decline", "disagree", "refuse", "deny", "denied"}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        first = lines[0].lower()
        first = re.sub(r'^[^a-z0-9]+', '', first)
        m = re.match(r'^([a-z]+)\b', first)
        if m:
            tok = m.group(1)
            if tok in pos:
                return True
            if tok in neg:
                return False

    t = text.lower()
    hits = []
    for w in pos:
        for m in re.finditer(rf'\b{re.escape(w)}\b', t):
            hits.append((m.start(), True))
    for w in neg:
        for m in re.finditer(rf'\b{re.escape(w)}\b', t):
            hits.append((m.start(), False))

    if hits:
        hits.sort(key=lambda x: x[0])
        return hits[0][1]

    return None

def dollars(vals: List[float]) -> List[str]:
    return [f"${int(v) if v.is_integer() else v:.0f}" for v in vals]

def to_percent(v: float) -> float:
    return 100.0 * v

def money_str(x: float) -> str:
    s = f"${x:.2f}"
    return s.rstrip("0").rstrip(".")

class UltimatumDataset:
    def __init__(self, role: str = "proposer", eval_only: bool = False):
        super().__init__()
        self.role: str = role
        assert self.role in ("proposer", "receiver"), "role must be 'proposer' or 'receiver'"
        self.funds: Tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.eval_only: bool = eval_only
        self.data: List[dict] = []
        self._build_prompts()
        self.counter = 0
        self.detailed_results: List[dict] = []

    def _build_prompts(self):

        for total in self.funds:
            if self.role == "proposer":
                allowed = [total * i / 5.0 for i in range(6)]
                opts_str = ", ".join(dollars([float(x) for x in allowed]))
                prompt = (
                    f"You have {dollars([float(total)])[0]} to split between yourself and the other player.\n"
                    f"If the other player agrees, you both get the proposed split; otherwise both get $0.\n"
                    f"How much will you offer to the other player?\n"
                    f"Pick one amount from: [{opts_str}].\n"
                )
                self.data.append({
                    "role": "proposer",
                    "total": total,
                    "allowed_offers": allowed,
                    "prompt": prompt
                })

            else:  
                y_max = (total - 1) // 2
                for you_amt in range(1, y_max + 1):
                    for counterpart in [""]:  
                        self_amt = total - you_amt  
                        you_pct = round(100.0 * you_amt / total, 2)
                        prompts = [
                            (
                                f"The {counterpart}proposer has {money_str(total)} to split between you and them.\n"
                                f"They propose {money_str(you_amt)} for you and {money_str(self_amt)} for themselves.\n"
                                f"If you REJECT, both receive $0.\n"
                                f"Do you accept this split?\n"
                                f'Answer with one word: "ACCEPT" or "REJECT".'
                            ),
                            (
                                f"Your {counterpart}coplayer has a fixed amount of money — {money_str(total)} — to split between themself and you.\n"
                                "If you agree with the split, you both receive the proposed amounts; otherwise, both of you receive $0.\n\n"
                                f"The proposed split is {100 - you_pct}% for the coplayer and {you_pct}% for you.\n"
                                "Do you accept this split?\n\n"
                                'Answer with 1 word: "ACCEPT" or "REJECT".'
                            ),
                            (
                                f"Game: one-shot take-it-or-leave-it split. A {counterpart}proposer divides a fixed pot of {money_str(total)} "
                                f"between you and themselves. If you ACCEPT, you receive {money_str(you_amt)} and they receive {money_str(self_amt)}; "
                                f"if you REJECT, both receive $0.\n\n"
                                f"Your sole objective is to maximize your own monetary payoff in this single round.\n\n"
                                f"Do you accept this split?\n"
                                f'Answer with exactly one word: "ACCEPT" or "REJECT".'
                            ),
                            (
                                f"This is a one-round Ultimatum game. You are a rational agent who seeks to maximize your own utility. A {counterpart}coplayer proposes how to split a fixed pot worth {money_str(total)}. "
                                f"If you ACCEPT, the split is implemented; if you REJECT, both get $0.\n\n"
                                f"Proposed split: you get {you_pct}% ({money_str(you_amt)}) and the coplayer gets {100 - you_pct}% ({money_str(self_amt)}).\n\n"
                                f"Decision rule: maximize your own monetary payoff in this round.\n"
                                f"Do you accept?\n"
                                f'Reply with one word: "ACCEPT" or "REJECT".'
                            ),                            
                        ]

                        for prompt in prompts:
                            self.data.append({
                                "role": "receiver",
                                "total": total,
                                "you_amount": you_amt,          
                                "self_amount": self_amt,        
                                "you_pct": you_pct,             
                                "counterpart": counterpart,
                                "prompt": prompt,
                            })

    def get_apply_chat_template(self): return True
    def get_clean_gpu_memory_before_finalize(self): return False
    def get_unique_name(self): return "ultimatum"
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

            if item["role"] == "proposer":
                allowed = item["allowed_offers"]
                amt = parse_amount_from_text(gen, allowed)
                pct = to_percent(amt / item["total"]) if (amt is not None and item["total"] > 0) else None
                record = {
                    "Role": "proposer",
                    "Total_sum": item["total"],
                    "Allowed_offers": allowed,
                    "Chosen_amount": amt,
                    "Chosen_pct": pct,
                    "Raw_output": gen,
                    "Thinking": think_parts[i],
                    "Prompt": item["prompt"],
                }
            else:
                accept = parse_accept_reject(gen)
                you_amt = item["you_amount"]
                self_amt = item["self_amount"]
                you_pct = round(100.0 * you_amt / item["total"], 2) if item["total"] else None
                record = {
                    "Role": "receiver",
                    "Total_sum": item["total"],
                    "Proposed_amount_you": you_amt,
                    "Proposed_amount_self": self_amt,
                    "Proposer_pct_you": you_pct,             
                    "Accept": accept,
                    "Raw_output": gen,
                    "Prompt": item["prompt"],
                }

            self.detailed_results.append(record)

        self.counter += len(llm_generations)

    def finalize(self, save_path=None):
        out_dir = save_path if save_path is not None else "./results/Ultimatum"
        _ensure_dir(out_dir)
        jsonl_name = f"ultimatum-{self.role}.jsonl"
        jsonl_path = os.path.join(out_dir, jsonl_name)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in self.detailed_results:
                f.write(json.dumps(rec) + "\n")
        print(f"[Ultimatum] Raw JSONL saved -> {jsonl_path}")
        open(os.path.join(out_dir, "done_eval.json"), "w").close()


if __name__ == "__main__":
    # Example usage
    evaluator = UltimatumDataset(role="receiver", eval_only=False)
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