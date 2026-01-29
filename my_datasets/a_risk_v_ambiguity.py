# risk_vs_ambiguity.py
import os
import re
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

def parse_choice_free(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().lower()
    if ("known" in t) and ("unknown" not in t):
        return "KNOWN"
    if ("unknown" in t) and ("known" not in t):
        return "UNKNOWN"
    return None

def _resolve_trade_choice(t: Dict, gen: str) -> Optional[str]:
    if not gen: 
        return None
    g = gen.lower()
    pL_pct = str(int(round(100 * t["p_low"])))
    pH_pct = str(int(round(100 * t["p_high"])))
    G_high = str(t["G_high"])
    G_low  = str(t["G_low"])
    score_hr = int(pL_pct in g) + int(G_high in g)
    score_lr = int(pH_pct in g) + int(G_low  in g)
    if score_hr > score_lr:
        return "HRHR"
    if score_lr > score_hr:
        return "LRLR"
    return None

def _resolve_by_index(t: Dict, gen: str) -> Optional[str]:
    if not gen:
        return None
    g = gen.lower()
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
    return t["order_map"].get(chosen)

@dataclass
class RiskAmbiguityConfig:
    randomize_order: bool = False
    seed: int = 7
    risk_probs: Tuple[int, ...] = (20, 30, 40, 60, 70, 80) 
    risk_sure_grid: Tuple[int, ...] = (10, 20, 50, 100)
    # Ambiguity block (Ellsberg)
    ell_payoff_grid: Tuple[int, ...] = (20, 30, 40, 50, 100)

class RiskAmbiguityDataset:
    def __init__(self, eval_only: bool = False):
        super().__init__()
        self.eval_only: bool = eval_only
        self.cfg = RiskAmbiguityConfig()
        self.rng = random.Random(self.cfg.seed)
        self.trials: List[Dict] = []
        self.trials.extend(self._make_risk_trials())
        self.trials.extend(self._make_ambiguity_trials())
        self.counter = 0
        self.records: List[Dict] = []

    def _make_risk_trials(self) -> List[Dict]:
        p_low_list: Tuple[float, ...]  = (0.20, 0.25, 0.30)
        p_high_list: Tuple[float, ...] = (0.70, 0.75, 0.80)
        EV_list: Tuple[int, ...]       = (10, 15, 20, 30, 40)  

        trade_pool = []
        for pL in p_low_list:
            for pH in p_high_list:
                for EV in EV_list:
                    gH_raw = EV / pL
                    gL_raw = EV / pH
                    stepH = 5 if EV >= 40 else 1
                    stepL = 5 if EV >= 40 else 1
                    G_high = max(stepH, int(round(gH_raw / stepH) * stepH))
                    G_low  = max(stepL, int(round(gL_raw / stepL) * stepL))
                    pL_pct = int(round(100 * pL))
                    pH_pct = int(round(100 * pH))
                    hrhr = f"{pL_pct}% chance to receive ${G_high}; otherwise $0."
                    lrlr = f"{pH_pct}% chance to receive ${G_low}; otherwise $0."
                    if self.cfg.randomize_order and self.rng.random() < 0.5:
                        opt1, opt2 = hrhr, lrlr
                        omap = {"OPT1": "HRHR", "OPT2": "LRLR"}
                    else:
                        opt1, opt2 = lrlr, hrhr
                        omap = {"OPT1": "LRLR", "OPT2": "HRHR"}
                    trade_pool.append({
                        "Block": "Risk",
                        "S": EV,
                        "p_low": pL, "p_high": pH,
                        "G_high": G_high, "G_low": G_low,
                        "opt1": opt1, "opt2": opt2, "order_map": omap,
                    })

        return trade_pool

    def _make_ambiguity_trials(self) -> List[Dict]:
        trials = []
        for G in [20, 30, 40, 50, 100]:
            for n_ball in [25, 50]:
                for p in [0.25, 0.75]:
                    templates = (
                        (f"Known urn (50/50): {n_ball} RED + {n_ball} BLACK. Win ${G} if RED is drawn; otherwise $0.",
                        f"Unknown urn: {2*n_ball} balls; red/black ratio unspecified. Win ${G} if RED is drawn; otherwise $0."),
                        (f"{n_ball} RED and {n_ball} BLACK balls in the urn (50/50). Win ${G} on RED; else $0.",
                        f"{2*n_ball} total balls, unknown proportion of red vs black. Win ${G} on RED; else $0.")
                    )
                    for known, unknown in templates:
                        if p < 0.5:
                            opt1, opt2 = unknown, known
                            order_map = {"OPT1": "UNKNOWN", "OPT2": "KNOWN"}
                        else:
                            opt1, opt2 = known, unknown
                            order_map = {"OPT1": "KNOWN", "OPT2": "UNKNOWN"}

                        trials.append({
                            "Block": "Ambiguity",
                            "G": G,
                            "opt1": opt1, "opt2": opt2, "order_map": order_map
                        })
        return trials

    def _build_prompt(self, t: Dict) -> str:
        return (
            "Choose exactly ONE option:\n"
            f"1) {t['opt1']}\n"
            f"2) {t['opt2']}\n"
            "Answer with the option text only."
        )

    def get_apply_chat_template(self): return True
    def get_clean_gpu_memory_before_finalize(self): return False
    def get_unique_name(self): return "risk_ambiguity"
    def get_max_len(self): return 50
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

            if t["opt1"] in gen:
                choice = t["order_map"]["OPT1"]
            elif t["opt2"] in gen:
                choice = t["order_map"]["OPT2"]
            else:
                if t["Block"] == "Risk" and t.get("trial_type") == "risk_tradeoff":
                    choice = _resolve_trade_choice(t, gen)
                    if choice is None:
                        choice = _resolve_by_index(t, gen)
                elif t["Block"] == "Ambiguity":
                    choice = _resolve_by_index(t, gen)
                    if choice is None:
                        choice = parse_choice_free(gen)
                else:
                    choice = parse_choice_free(gen)

            rec = {
                "Block": t["Block"],
                "Option1": t["opt1"],
                "Option2": t["opt2"],
                "Choice": choice,
                "Thinking": think_parts[k],
                "Output": gen
            }

            if t["Block"] == "Risk" and t.get("trial_type") == "risk_tradeoff":
                rec.update({
                    "trial_type": "risk_tradeoff",
                    "EV": t["EV"],
                    "p_low": t["p_low"],
                    "p_high": t["p_high"],
                    "G_high": t["G_high"],
                    "G_low": t["G_low"],
                })
            elif t["Block"] == "Ambiguity":
                rec.update({"Payoff": t["G"]})

            self.records.append(rec)

        self.counter += len(llm_generations)

    def finalize(self, save_path: Optional[str] = None):
        out_dir = save_path if save_path is not None else "./results/RiskAmbiguity"
        os.makedirs(out_dir, exist_ok=True)
        jsonl_path = os.path.join(out_dir, "risk_ambiguity.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec) + "\n")
        print(f"[RiskAmbiguity] Raw JSONL saved -> {jsonl_path}")
        open(os.path.join(out_dir, "done_eval.json"), "w").close()

if __name__ == "__main__":
    # Example usage
    evaluator = RiskAmbiguityDataset(eval_only=False)
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