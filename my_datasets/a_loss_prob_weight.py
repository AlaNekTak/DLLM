# loss_prob_weight.py
import os, re, json, math, random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def parse_prob_value(text: str) -> Optional[float]:
    if not text:
        return None
    s = text.strip().lower()
    m = re.search(r"([-+]?\d*\.?\d+)", s)
    if not m:
        return None
    val = float(m.group(1))
    if "%" in s:
        val = val / 100.0
    return max(0.0, min(1.0, val))

def parse_accept_reject(text: str) -> Optional[bool]:
    if not text:
        return None
    t = text.strip().lower()
    pos = ["accept", "yes", "take gamble", "i accept", "go for it"]
    neg = ["reject", "no", "decline", "skip", "i reject"]
    if any(w in t for w in pos): return True
    if any(w in t for w in neg): return False
    return None

def parse_choice_token(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().lower()
    if "for certain" in t:            
        return "SURE"
    if "%" in t or "chance" in t:     
        return "GAMBLE"
    if "sure" in t or "certain" in t or "safe" in t:
        return "SURE"
    if "lottery" in t or "bet" in t or "gamble" in t:
        return "GAMBLE"
    return None

def _resolve_by_index(order_map: Dict[str, str], gen: str):
    if not gen:
        return None
    g = gen.lower()
    m = re.search(r"^\s*\(?\s*([12])\s*[\)\.\-:]\s*", g)
    if m:
        return order_map["OPT1" if m.group(1) == "1" else "OPT2"]
    i1 = g.find("1)")
    i2 = g.find("2)")
    if i1 == -1 and i2 == -1:
        return None
    if i1 == -1:
        chosen = "OPT2"
    elif i2 == -1:
        chosen = "OPT1"
    else:
        chosen = "OPT1" if i1 < i2 else "OPT2"
    return order_map.get(chosen)

def fmt_money(x: float | int) -> str:
    return f"${int(x):,}"

LOSS_ACCEPT_TEMPLATES = (
    "ACCEPT the coin-flip gamble: 50% chance to win {G} and 50% chance to lose {L}.",
    "ACCEPT the 50/50 gamble: win {G} with a 50% chance; otherwise lose {L}.",
    "ACCEPT the gamble: an even chance — win {G} or lose {L}.",
)
LOSS_REJECT_TEMPLATES = (
    "REJECT.",
    "REJECT; take $0 for certain.",
    "REJECT: win/lose $0 for sure.",
)

WEIGHT_SURE_TEMPLATES = (
    "Receive {S} for certain.",
    "Take {S} for sure.",
    "A sure {S}.",
)
WEIGHT_GAMBLE_TEMPLATES = (
    "{p_pct}% chance to receive {G}; otherwise $0.",
    "{p_pct}% chance at {G}; else $0.",
    "A {p_pct}% chance of winning {G}; otherwise nothing.",
)

@dataclass
class LAPWConfig:
    loss_grid: Tuple[int, ...] = (10, 20, 30, 40, 50)   
    ratio_grid: Tuple[float, ...] = tuple(np.round(np.linspace(1.2, 2.5, 10), 2))  
    ev_band: Tuple[float, float] = (0.8, 1.2)     
    p_list_pct: Tuple[int, ...] = (1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 93, 95, 97, 99)
    payoff_grid: Tuple[int, ...] = (5, 10, 20, 50, 100) 
    prelec_pilot_alpha: float = 0.7
    prelec_pilot_beta: float = 1.0
    p_mid_lo: float = 0.10
    p_mid_hi: float = 0.90
    mid_multipliers: Tuple[float, ...] = (0.90, 0.95, 1.00, 1.05, 1.10)
    ext_multipliers: Tuple[float, ...] = (0.80, 0.90, 1.00, 1.10, 1.20)

    loss_choice_templates: Tuple[str, ...] = (
        "Choose exactly ONE option by repeating the option text verbatim.\n1) {opt1}\n2) {opt2}.",
        "Pick one option:\n1) {opt1}\n2) {opt2}\n",
        "Select a single option. Respond with the option text only.\n1) {opt1}\n2) {opt2}\n",
    )

    weight_choice_templates: Tuple[str, ...] = (
        "Choose exactly ONE option by repeating the option text verbatim.\n1) {opt1}\n2) {opt2}\n",
        "Pick one option:\n1) {opt1}\n2) {opt2}\n",
        "Select a single option. Respond with the option text only.\n1) {opt1}\n2) {opt2}\n",
    )

class LossProbWeightDataset:
    def __init__(self, eval_only: bool = False):
        super().__init__()
        self.eval_only: bool = eval_only
        self.cfg = LAPWConfig()
        self.rng = random.Random(123)
        self._uid = 0

        self.trials: List[Dict] = []
        self.trials.extend(self._make_loss_trials())
        self.trials.extend(self._make_weight_trials())
        self.counter = 0
        self.records: List[Dict] = []

    def _make_loss_trials(self) -> List[Dict]:
        gains  = list(range(5, 15)) 
        losses = list(range(5, 15))  
        full = []
        for G in gains:
            for L in losses:
                for tmpl_idx in range(len(self.cfg.loss_choice_templates)):
                    EV  = 0.5*G - 0.5*L
                    Var = 0.5*(G**2) + 0.5*(L**2) - EV**2
                    Gf, Lf = fmt_money(G), fmt_money(L)
                    opt_accept = self.rng.choice(LOSS_ACCEPT_TEMPLATES).format(G=Gf, L=Lf)
                    opt_reject = self.rng.choice(LOSS_REJECT_TEMPLATES)
                    if self.rng.random() < 0.5:
                        opt1, opt2 = opt_accept, opt_reject
                        order_map = {"OPT1": True, "OPT2": False}
                    else:
                        opt1, opt2 = opt_reject, opt_accept
                        order_map = {"OPT1": False, "OPT2": True}
                    full.append({
                        "Block": "Loss",
                        "gain": G,
                        "loss": L,
                        "ratio": G / L,     
                        "EV": EV,
                        "Var": Var,
                        "opt1": opt1,
                        "opt2": opt2,
                        "order_map": order_map,
                        "tmpl_idx": tmpl_idx,
                    })
        return full

    def _make_weight_trials(self) -> List[Dict]:
        p_list = [pp / 100.0 for pp in self.cfg.p_list_pct]
        trials = []
        def _w0(p: float) -> float:
            eps = 1e-12
            p_clip = max(min(p, 1.0 - eps), eps)
            a0, b0 = self.cfg.prelec_pilot_alpha, self.cfg.prelec_pilot_beta
            return math.exp(-b0 * ((-math.log(p_clip)) ** a0))
        for p in p_list:
            for G in self.cfg.payoff_grid:
                w0 = _w0(p)
                if self.cfg.p_mid_lo <= p <= self.cfg.p_mid_hi:
                    mults = self.cfg.mid_multipliers
                else:
                    mults = self.cfg.ext_multipliers
                for m in mults:
                    S = int(round(m * w0 * G))
                    p_pct = int(round(p * 100))
                    Sf, Gf = fmt_money(S), fmt_money(G)
                    sure = self.rng.choice(WEIGHT_SURE_TEMPLATES).format(S=Sf)
                    gamble = self.rng.choice(WEIGHT_GAMBLE_TEMPLATES).format(p_pct=p_pct, G=Gf)
                    if self.rng.random() < 0.5:
                        opt1, opt2 = gamble, sure
                        order_map = {"OPT1": "GAMBLE", "OPT2": "SURE"}
                    else:
                        opt1, opt2 = sure, gamble
                        order_map = {"OPT1": "SURE", "OPT2": "GAMBLE"}
                    tmpl_idx = self.rng.randrange(len(self.cfg.weight_choice_templates))
                    trials.append({
                        "Block": "Weight",
                        "p": p, "p_pct": p_pct, "G": G, "S": S,
                        "opt1": opt1, "opt2": opt2,
                        "order_map": order_map,
                        "tmpl_idx": tmpl_idx,
                    })
        return trials
    
    def _build_prompt(self, t: Dict) -> str:
        if t.get("Block") == "Loss":
            templ = self.cfg.loss_choice_templates[t.get("tmpl_idx", 0) % len(self.cfg.loss_choice_templates)]
        else:  
            templ = self.cfg.weight_choice_templates[t.get("tmpl_idx", 0) % len(self.cfg.weight_choice_templates)]
        return templ.format(opt1=t["opt1"], opt2=t["opt2"])

    def get_apply_chat_template(self): return True
    def get_clean_gpu_memory_before_finalize(self): return False
    def get_unique_name(self): return "loss_prob_weight"
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
            if t["Block"] == "Loss":
                if t["opt1"] in gen:
                    choice = t["order_map"]["OPT1"]
                elif t["opt2"] in gen:
                    choice = t["order_map"]["OPT2"]
                else:
                    pr = parse_accept_reject(gen)
                    if pr is not None:
                        choice = pr
                    else:
                        choice = _resolve_by_index(t["order_map"], gen)

                rec = {
                    "Block": "Loss",
                    "TemplateIdx": t["tmpl_idx"],
                    "EV": t["EV"],
                    "Gain": t["gain"],
                    "Loss": t["loss"],
                    "Ratio": t["ratio"],
                    "Option1": t["opt1"],
                    "Option2": t["opt2"],
                    "Accept": choice,   
                    "Output": gen,
                    "Thinking": think_parts[k],
                }

            elif t["Block"] == "Weight":
                if t["opt1"] in gen:
                    choice = t["order_map"]["OPT1"]
                elif t["opt2"] in gen:
                    choice = t["order_map"]["OPT2"]
                else:
                    parsed = parse_choice_token(gen)
                    if parsed in ("SURE", "GAMBLE"):
                        choice = parsed
                    else:
                        choice = _resolve_by_index(t["order_map"], gen)
                rec = {
                    "Block": "Weight",
                    "TemplateIdx": t["tmpl_idx"],
                    "p": t["p"],
                    "p_pct": t["p_pct"],
                    "G": t["G"],
                    "S": t["S"],
                    "Option1": t["opt1"],
                    "Option2": t["opt2"],
                    "Choice": choice,   
                    "Output": gen
                }

            self.records.append(rec)
        self.counter += len(llm_generations)

    def finalize(self, save_path: Optional[str] = None):
        out_dir = save_path if save_path is not None else "./results/LossProbWeight"
        os.makedirs(out_dir, exist_ok=True)
        jsonl_path = os.path.join(out_dir, "loss_prob_weight.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec) + "\n")
        print(f"[LAPW] Raw JSONL saved -> {jsonl_path}")
        open(os.path.join(out_dir, "done_eval.json"), "w").close()


if __name__ == "__main__":
    # Example usage
    evaluator = LossProbWeightDataset(eval_only=False)
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