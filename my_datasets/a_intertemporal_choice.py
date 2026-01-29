# intertemporal_choice.py
import os, re, json, random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

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

def parse_choice_token(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().lower()
    m = re.search(r"\b([12])\b", t)
    if m:
        return f"OPT{m.group(1)}"
    if any(w in t for w in ["today", "now", "immediately", "right away", "sooner"]):
        return "SOONER"
    if any(w in t for w in ["later", "week", "weeks", "month", "months", "wait", "in ", "after "]):
        return "LATER"
    return None

@dataclass
class IntertemporalConfig:
    mcq_delays_days: Tuple[int, ...] = (7, 14, 30, 60, 90, 120, 180) 
    mcq_bands: Tuple[Tuple[str, int, Tuple[int, ...]], ...] = (
        ("MCQ_Small", 25, (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)),
        ("MCQ_Med",   55, (25, 27, 30, 32, 35, 37, 40, 42, 45, 47, 50, 52, 53, 54)),
        ("MCQ_Large", 85, (35, 37, 40, 42, 45, 50, 52, 55, 60, 62, 65, 70, 75, 80)),
    )
    set2_amount_pairs: Tuple[Tuple[int, int], ...] = (
        (6, 7), (10, 12), (14, 17), (20, 25), (28, 35), (34, 45), (40, 57)
    )
    set2_delay_pairs: Tuple[Tuple[int, int], ...] = (
        (0, 14), (0, 28), (0, 42), (14, 28), (14, 42), (28, 42)
    )

class IntertemporalDataset:
    def __init__(self, eval_only: bool = False):
        super().__init__()
        self.eval_only: bool = eval_only
        self.cfg = IntertemporalConfig()
        self.rng = random.Random(2025)
        self.prompt_templates = [
            "Choose exactly ONE option, weighing the payoff of waiting against the time you must wait.",
            "Choose exactly ONE option, considering both the delay and how much additional money it offers.",
        ]
        base_trials: List[Dict] = []
        base_trials.extend(self._make_set_mcq27_trials())
        base_trials.extend(self._make_set2_trials())
        self.trials: List[Dict] = []
        for tmpl_id, tmpl in enumerate(self.prompt_templates):
            for t in base_trials:
                t2 = dict(t)                  
                t2["prompt_template_id"] = tmpl_id
                self.trials.append(t2)
        self.counter = 0
        self.records: List[Dict] = []

    def _make_set_mcq27_trials(self) -> List[Dict]:
        rng = self.rng
        trials = []
        delays = list(self.cfg.mcq_delays_days)
        for band_name, A_l, ss_list in self.cfg.mcq_bands:
            for i, A_s in enumerate(ss_list):      
                for p in [0.25, 0.75]:
                    t_l = delays[i % len(delays)]       
                    sooner = f"${A_s} today."
                    later  = f"${A_l} in {t_l} days."
                    if p < 0.5:
                        opt1, opt2 = later, sooner
                        order_map = {"OPT1": "LATER", "OPT2": "SOONER"}
                    else:
                        opt1, opt2 = sooner, later
                        order_map = {"OPT1": "SOONER", "OPT2": "LATER"}
                    trials.append({
                        "Set": "SetMCQ27",
                        "Band": band_name,
                        "A_s": A_s, "t_s": 0,
                        "A_l": A_l, "t_l": t_l,
                        "opt1": opt1, "opt2": opt2, "order_map": order_map
                    })
        return trials

    def _make_set2_trials(self) -> List[Dict]:
        trials = []
        pairs  = list(self.cfg.set2_amount_pairs)
        delays = list(self.cfg.set2_delay_pairs)
        for (A_s, A_l) in pairs:
            for (t_s, t_l) in delays:
                for p in [0.25, 0.75]:
                    sooner = f"${A_s} today." if t_s == 0 else f"${A_s} in {t_s} days."
                    if t_l in (14, 28, 42):
                        w = t_l // 7
                        later = f"${A_l} in {w} weeks."
                    else:
                        later = f"${A_l} in {t_l} days."
                    if p < 0.5:
                        opt1, opt2 = later, sooner
                        order_map = {"OPT1": "LATER", "OPT2": "SOONER"}
                    else:
                        opt1, opt2 = sooner, later
                        order_map = {"OPT1": "SOONER", "OPT2": "LATER"}
                    trials.append({
                        "Set": "Set2",
                        "A_s": A_s, "t_s": t_s,
                        "A_l": A_l, "t_l": t_l,
                        "opt1": opt1, "opt2": opt2, "order_map": order_map
                    })
        return trials

    def _build_prompt(self, t: Dict) -> str:
        tmpl_id = t.get("prompt_template_id", 0)
        header = self.prompt_templates[tmpl_id]
        return (
            f"{header}\n"
            f"1) {t['opt1']}\n"
            f"2) {t['opt2']}\n"
            "Answer with the option text only."
        )

    def get_apply_chat_template(self): return True
    def get_clean_gpu_memory_before_finalize(self): return False
    def get_unique_name(self): return "intertemporal"
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
            choice = None
            if t["opt1"] in gen:
                choice = t["order_map"]["OPT1"]
            elif t["opt2"] in gen:
                choice = t["order_map"]["OPT2"]
            else:
                parsed = parse_choice_token(gen)
                if parsed in ("OPT1", "OPT2"):
                    choice = t["order_map"][parsed]
                elif parsed in ("SOONER", "LATER"):
                    choice = parsed

            rec = {
                "Set": t["Set"],
                "A_sooner": t["A_s"], "t_sooner_days": t["t_s"],
                "A_later": t["A_l"], "t_later_days": t["t_l"],
                "Option1": t["opt1"], "Option2": t["opt2"],
                "Choice": choice, "Output": gen, "Thinking": think_parts[k],
                "prompt_template_id": t.get("prompt_template_id", 0),
                "prompt_template_text": self.prompt_templates[t.get("prompt_template_id", 0)],
            }
            # keep MCQ band name if present
            if "Band" in t:
                rec["Band"] = t["Band"]
            self.records.append(rec)
        self.counter += len(llm_generations)

    def finalize(self, save_path: Optional[str] = None):
        out_dir = save_path if save_path is not None else "./results/Intertemporal"
        os.makedirs(out_dir, exist_ok=True)
        jsonl_path = os.path.join(out_dir, "intertemporal.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec) + "\n")
        print(f"[Intertemporal] Raw JSONL saved -> {jsonl_path}")
        open(os.path.join(out_dir, "done_eval.json"), "w").close()
 
 
if __name__ == "__main__":
    # Example usage
    evaluator = IntertemporalDataset(eval_only=False)
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