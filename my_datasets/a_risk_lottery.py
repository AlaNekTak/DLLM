# risk_lottery.py
import os
import re
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def parse_choice_token(text: str) -> Optional[str]:
    if not text: return None
    t = text.strip().lower()
    if "for certain" in t:             
        return "SURE"
    if "%" in t or "chance" in t:    
        return "GAMBLE"
    if "sure" in t or "certain" in t or "safe" in t:
        return "SURE"
    if "lottery" in t or "bet" in t:
        return "GAMBLE"
    return None

def parse_estimate(text: str) -> Optional[int]:
    if not text:
        return None
    s = text.strip()
    def next_nonspace_char(idx: int) -> Optional[str]:
        j = idx
        while j < len(s) and s[j].isspace():
            j += 1
        return s[j] if j < len(s) else None
    def prev_nonspace_char(idx: int) -> Optional[str]:
        j = idx - 1
        while j >= 0 and s[j].isspace():
            j -= 1
        return s[j] if j >= 0 else None

    plain_rx = re.compile(r'([-+]?\d+(?:\.\d+)?)')
    for m in plain_rx.finditer(s):
        token = m.group(1)
        end = m.end()
        nxt = next_nonspace_char(end)
        prv = prev_nonspace_char(m.start())

        if nxt == ')':
            continue

        if nxt in {'%', '/'}:
            continue

        if prv == '$':
            continue

        try:
            x = float(token)
        except Exception:
            continue

        if x >= 1.0:
            val = x
            return int(round(clamp(val, 0, 100)))

    pct_rx = re.compile(r'([-+]?\d+(?:\.\d+)?)\s*%')
    m = pct_rx.search(s)
    if m:
        val = float(m.group(1))
        return int(round(clamp(val, 0, 100)))

    frac_rx = re.compile(r'([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)')
    m = frac_rx.search(s)
    if m:
        try:
            num = float(m.group(1))
            den = float(m.group(2))
            if den == 0:
                return None
            val = (num / den) * 100.0
            return int(round(clamp(val, 0, 100)))
        except Exception:
            return None

    for m in plain_rx.finditer(s):
        token = m.group(1)
        end = m.end()
        nxt = next_nonspace_char(end)
        prv = prev_nonspace_char(m.start())

        if nxt == ')':
            continue

        if nxt in {'%', '/'}:
            continue

        if prv == '$':
            continue

        try:
            x = float(token)
        except Exception:
            continue

        if 0.0 <= x <= 1.0:
            val = x * 100.0
            return int(round(clamp(val, 0, 100)))
    return None

@dataclass
class RiskPerceptionConfig:
    n_trials_min: int = 20
    n_trials_max: int = 30
    n_estimate_trials: int = 5 
    seed: int = 42

    choice_templates: Tuple[str, ...] = (
        "Choose exactly ONE option by repeating the option text verbatim.\n1) {opt1}\n2) {opt2}\n",
        "Pick one option.\n1) {opt1}\n2) {opt2}\n",
        "Select a single option. Respond with the option text only.\n1) {opt1}\n2) {opt2}\n",
    )
    estimate_templates: Tuple[str, ...] = (
        "Consider this gamble: {p_pct}% chance to receive ${G}; otherwise $0.\n"
        "On a 0-100 scale, what is the chance this gamble pays off? Answer with just the integer.",
        "Imagine a lottery with a {p_pct}% chance at ${G} (else $0).\n"
        "Report a single number 0-100 for how likely it feels to succeed.",
        "There is a {p_pct}% chance to win ${G}. Write an integer from 0 to 100 representing the chance it pays out.",
    )

class RiskPerceptionDataset:
    def __init__(self, eval_only: bool = False):
        super().__init__()
        self.eval_only: bool = eval_only
        self.cfg = RiskPerceptionConfig()
        self.rng = random.Random(self.cfg.seed)
        self._uid = 0 
        self.trials: List[Dict] = self._make_trials()
        self.counter = 0
        self.records: List[Dict] = []
        
    def _make_trials(self) -> List[Dict]:
        rng = self.rng
        sure_grid = [10, 20, 50, 100]
        p_grid = [0.30, 0.35, 0.40, 0.45, 0.55, 0.60, 0.65, 0.70]
        deltas = [-0.15, -0.125, -0.10, -0.075, -0.05, +0.05, +0.075, +0.10, +0.125, +0.15]
        trials: List[Dict] = []

        for S in sure_grid:
            for p in p_grid:
                for d in deltas:
                    target_ev = S * (1.0 + d)
                    g_raw = target_ev / p
                    step = 5 if S >= 50 else 1
                    G = max(step, int(round(g_raw / step) * step))
                    if G < step or G > 10_000:
                        continue

                    p_pct = int(round(100 * p))
                    sure_text   = f"Receive ${S} for certain."
                    gamble_text = f"{p_pct}% chance to receive ${G}; otherwise $0."

                    if rng.random() < 0.5:
                        opt1, opt2 = gamble_text, sure_text
                        order_map = {"OPT1": "GAMBLE", "OPT2": "SURE"}
                    else:
                        opt1, opt2 = sure_text, gamble_text
                        order_map = {"OPT1": "SURE", "OPT2": "GAMBLE"}

                    pair_key = f"{p_pct}-{S}-{G}"

                    tmpl_idx = rng.randrange(len(self.cfg.choice_templates))
                    self._uid += 1
                    trials.append({
                        "uid": f"C-{pair_key}-T{tmpl_idx}",
                        "pair_key": pair_key,
                        "prompt_kind": "choice",
                        "tmpl_idx": tmpl_idx,
                        "S": S, "p": p, "p_pct": p_pct, "G": G,
                        "opt1": opt1, "opt2": opt2, "order_map": order_map,
                        "EV_gamble": p * G,
                        "EV_diff":   p * G - S,
                    })

                    et_idx = rng.randrange(len(self.cfg.estimate_templates))
                    self._uid += 1
                    trials.append({
                        "uid": f"E-{pair_key}-T{et_idx}",
                        "pair_key": pair_key,
                        "prompt_kind": "estimate",
                        "tmpl_idx": et_idx,
                        "S": S, "p": p, "p_pct": p_pct, "G": G,
                        "EV_gamble": p * G,
                        "EV_diff":   p * G - S,
                    })

        return trials
    
    def _build_prompt(self, t: Dict) -> str:
        kind = t.get("prompt_kind", "choice")
        if kind == "choice":
            templ = self.cfg.choice_templates[t.get("tmpl_idx", 0) % len(self.cfg.choice_templates)]
            return templ.format(opt1=t["opt1"], opt2=t["opt2"])
        elif kind == "estimate":
            templ = self.cfg.estimate_templates[t.get("tmpl_idx", 0) % len(self.cfg.estimate_templates)]
            return templ.format(p_pct=int(round(t["p"]*100)), G=t["G"])
        else:
            return f"1) {t.get('opt1','')}\n2) {t.get('opt2','')}\n"

    def get_apply_chat_template(self): return True
    def get_clean_gpu_memory_before_finalize(self): return False
    def get_unique_name(self): return "risk_perception"
    def get_max_len(self): return 80
    def get_progress(self): return self.counter / len(self.trials) if self.trials else 1.0
    def get_class_labels(self): return [], 0, 5 

    def get_system_prompt(self, preffered_batch_size):
        n = min(preffered_batch_size, len(self.trials) - self.counter)
        return [""] * n

    def get_user_prompt(self, preffered_batch_size):
        n = min(preffered_batch_size, len(self.trials) - self.counter)
        return [self._build_prompt(self.trials[self.counter + i]) for i in range(n)]

    def get_assistant_prompt(self, preffered_batch_size):
        n = min(preffered_batch_size, len(self.trials) - self.counter)
        return ["Answer:"] * n

    def is_finished(self):
        return self.counter >= len(self.trials) or self.eval_only

    def process_results(self, llm_generations, think_parts, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for i, gen in enumerate(llm_generations):
            t = self.trials[self.counter + i]
            kind = t.get("prompt_kind", "choice")

            if kind == "choice":
                choice = None
                if t["opt1"] in gen:
                    choice = t["order_map"]["OPT1"]
                elif t["opt2"] in gen:
                    choice = t["order_map"]["OPT2"]
                else:
                    parsed = parse_choice_token(gen)
                    if parsed in ("SURE", "GAMBLE"):
                        choice = parsed

                rec = {
                    "UID": t["uid"],
                    "PairKey": t["pair_key"],
                    "PromptKind": "choice",
                    "TemplateIdx": t.get("tmpl_idx", 0),
                    "Sure_amount": t["S"],
                    "Prob_success": t["p"],
                    "Prob_success_pct": t["p_pct"],
                    "Gamble_payoff": t["G"],
                    "Option1": t.get("opt1"),
                    "Option2": t.get("opt2"),
                    "Choice": choice,
                    "Estimate": None,
                    "EV_gamble": t["EV_gamble"],
                    "EV_diff": t["EV_diff"],
                    "Output": gen,
                    "Thinking": think_parts[i],
                }

            elif kind == "estimate":
                est = parse_estimate(gen)
                rec = {
                    "UID": t["uid"],
                    "PairKey": t["pair_key"],
                    "PromptKind": "estimate",
                    "TemplateIdx": t.get("tmpl_idx", 0),
                    "Sure_amount": t["S"],
                    "Prob_success": t["p"],
                    "Prob_success_pct": t["p_pct"],
                    "Gamble_payoff": t["G"],
                    "Option1": None,
                    "Option2": None,
                    "Choice": None,
                    "Estimate": est,
                    "EV_gamble": t["EV_gamble"],
                    "EV_diff": t["EV_diff"],
                    "Output": gen,
                }

            else:
                rec = {"UID": t.get("uid"), "PairKey": t.get("pair_key"),
                    "PromptKind": str(kind), "TemplateIdx": t.get("tmpl_idx", 0),
                    "Output": gen}

            self.records.append(rec)
        self.counter += len(llm_generations)

    def finalize(self, save_path: Optional[str] = None):
        out_dir = save_path if save_path is not None else "./results/RiskLottery"
        os.makedirs(out_dir, exist_ok=True)
        jsonl_path = os.path.join(out_dir, "risk_perception.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec) + "\n")
        print(f"[RiskPerception] Raw JSONL saved -> {jsonl_path}")
        open(os.path.join(out_dir, "done_eval.json"), "w").close()
        
if __name__ == "__main__":
    # Example usage
    evaluator = RiskPerceptionDataset(eval_only=False)
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