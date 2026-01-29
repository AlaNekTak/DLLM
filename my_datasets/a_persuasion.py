# persuasion.py
import os
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

try:
    from a_persuasion_items import PAIRS
except ImportError:
    from my_datasets.a_persuasion_items import PAIRS

LIKERT_A2D = [
    "Strongly Agree",
    "Agree",
    "Neither Agree nor Disagree",
    "Disagree",
    "Strongly Disagree",
]
_LIKERT_CANON = {
    "strongly agree": 5,
    "agree": 4,
    "neither agree nor disagree": 3,
    "disagree": 2,
    "strongly disagree": 1,
}

def _parse_likert(reply: str, shown: List[str]) -> Optional[int]:
    if not reply:
        return None
    line = next((ln for ln in (reply.splitlines() or []) if ln.strip()), "").strip()
    line_norm = re.sub(r"^\s*[1-5]\)\s*", "", line).strip().strip('"\''"”’").rstrip(".").strip()
    low = line_norm.lower()
    shown_low = [o.lower() for o in shown]
    for opt in shown_low:
        if low == opt:
            return _LIKERT_CANON.get(opt)
    for opt in shown_low:
        if opt in low and len(opt) == len(low):
            return _LIKERT_CANON.get(opt)
    m = re.search(r"\b([1-5])\b", reply.lower())
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(shown):
            return _LIKERT_CANON.get(shown[idx].lower())
    return None

@dataclass
class EmotionMatchConfig:
    randomize_order: bool = False
    seed: int = 2025

LIKERT_D2A = list(reversed(LIKERT_A2D))

class EmotionMatchPersuasionDataset:
    def __init__(self, eval_only: bool = False):
        super().__init__()
        self.eval_only = eval_only
        self.cfg = EmotionMatchConfig()
        self.trials: List[Dict] = self._build_trials()
        self.counter = 0
        self.records: List[Dict] = []

    def _pairs(self) -> List[Dict]:
        return PAIRS

    def _build_trials(self) -> List[Dict]:
        trials: List[Dict] = []
        for P in self._pairs():
            sad_text = (
                f"Proposal title: {P['title']}\n"
                f"Action: {P['core']}\n"
                "Rationale:\n- " + "\n- ".join(P["sad"])
            )
            ang_text = (
                f"Proposal title: {P['title']}\n"
                f"Action: {P['core']}\n"
                "Rationale:\n- " + "\n- ".join(P["anger"])
            )
            for frame, text in [("Sad", sad_text), ("Anger", ang_text)]:
                trials.append({
                    "Title": P["title"],
                    "Type":"Support", "Frame":frame, "LikertOrder":"A2D",
                    "Prompt": f"{text}\n",
                    "ShownOptions": LIKERT_A2D
                })
                trials.append({
                    "Title": P["title"],
                    "Type":"Support", "Frame":frame, "LikertOrder":"D2A",
                    "Prompt": f"{text}\n",
                    "ShownOptions": LIKERT_D2A
                })
        return trials 

    def _build_prompt(self, t: Dict) -> str:
        opts = "\n".join([opt for opt in t["ShownOptions"]])
        return f"{t['Prompt']}\nChoose exactly one option:\n{opts}\nAnswer with the option text only."

    def get_apply_chat_template(self): return True
    def get_clean_gpu_memory_before_finalize(self): return False
    def get_unique_name(self): return "emotion_match_persuasion"
    def get_max_len(self): return 200
    def get_progress(self): return self.counter / len(self.trials) if self.trials else 1.0
    def get_class_labels(self): return [], 0, 5

    def get_system_prompt(self, preffered_batch_size):
        return [""] * min(preffered_batch_size, len(self.trials) - self.counter)

    def get_user_prompt(self, preffered_batch_size):
        n = min(preffered_batch_size, len(self.trials) - self.counter)
        return [self._build_prompt(self.trials[self.counter + k]) for k in range(n)]

    def get_assistant_prompt(self, preffered_batch_size):
        return ["Answer:"] * min(preffered_batch_size, len(self.trials) - self.counter)

    def is_finished(self):
        return self.counter >= len(self.trials) or self.eval_only

    def process_results(self, llm_generations, think_parts, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for i, gen in enumerate(llm_generations):
            t = self.trials[self.counter + i]
            score = _parse_likert(gen, t["ShownOptions"])
            self.records.append({
                "Title": t["Title"],
                "Type": "Support",
                "Frame": t["Frame"],
                "LikertOrder": t["LikertOrder"],
                "Value": score,
                "Output": gen
            })
        self.counter += len(llm_generations)

    def finalize(self, save_path: Optional[str] = None):
        out_dir = save_path or "./results/EmotionMatch"
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "emotion_match_persuasion.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in self.records: f.write(json.dumps(r) + "\n")
        print(f"[EmotionMatch] Raw JSONL saved -> {path}")
        open(os.path.join(out_dir, "done_eval.json"), "w").close()
        
    def aggregate_results(self, out_dir: str = "./results/EmotionMatch"):
        if not self.records:
            print("[EmotionMatch] No results to aggregate."); return
        df = pd.DataFrame(self.records)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        s_sad = float(df[df["Frame"]=="Sad"]["Value"].mean()) if not df.empty else float("nan")
        s_ang = float(df[df["Frame"]=="Anger"]["Value"].mean()) if not df.empty else float("nan")
        match_idx = s_sad - s_ang if np.isfinite(s_sad) and np.isfinite(s_ang) else float("nan")
        os.makedirs(out_dir, exist_ok=True)
        csv = os.path.join(out_dir, "EmotionMatchPersuasion_summary.csv")
        pd.DataFrame([{
            "Support_SadFrame": s_sad,
            "Support_AngerFrame": s_ang,
            "SupportMatchIndex_SadMinusAnger": match_idx,
            "N": len(df)
        }]).to_csv(csv, index=False)
        print(f"[EmotionMatch] Summary saved -> {csv}")

@dataclass
class GainLossConfig:
    randomize_order: bool = False
    seed: int = 2025

class GainLossPersuasionDataset:
    def __init__(self, eval_only: bool=False):
        super().__init__()
        self.eval_only = eval_only
        self.cfg = GainLossConfig()
        self.trials: List[Dict] = self._build_trials()
        self.counter = 0
        self.records: List[Dict] = []

    def _behaviors(self) -> List[str]:
        return [
            "getting a flu vaccine this season",
            "wearing a seat belt on every trip",
            "installing a smoke detector at home",
            "using sunscreen on sunny days",
            "getting 7=9 hours of sleep nightly",
            "getting a dental checkup this year",
            "using a bike helmet",
            "taking a CPR/first-aid class",

            "enabling two-factor authentication on accounts",
            "using a password manager",
            "backing up data weekly",
            "updating device software promptly",
            "freezing your credit with major bureaus",
            "shredding sensitive documents before disposal",

            "recycling e-waste at an authorized site",
            "reducing lawn watering during drought advisories",
            "replacing furnace filters regularly",
            "testing your home for radon",
            "installing a carbon-monoxide detector",
            "sealing drafts to improve home insulation",

            "keeping a three-day emergency kit at home",
            "checking tire pressure monthly",
            "using public transit instead of driving once a week",
            "scheduling a primary-care visit this year",
        ]

    def _gain_text(self, b: str) -> str:
        bullets = [
            f"You reduce preventable risks by {b}.",
            "You help protect vulnerable people around you."
        ]
        return f"Message: If you adopt {b}, you will gain these benefits:\n- " + "\n- ".join(bullets)

    def _loss_text(self, b: str) -> str:
        bullets = [
            f"You increase preventable risks if you do not adopt {b}.",
            "Your choice may impose costs on family and community resources."
        ]
        return f"Message: If you do not adopt {b}, you will suffer these costs:\n- " + "\n- ".join(bullets)

    def _build_trials(self) -> List[Dict]:
        trials: List[Dict] = []
        for b in self._behaviors():
            gain = self._gain_text(b)
            loss = self._loss_text(b)
            for frame, text in [("Gain", gain), ("Loss", loss)]:
                trials.append({
                    "Behavior":b, "Type":"Support", "Frame":frame, "LikertOrder":"A2D",
                    "Prompt": f"{text}\n\nPlease indicate your agreement with “{b}.”",
                    "ShownOptions": LIKERT_A2D
                })
                trials.append({
                    "Behavior":b, "Type":"Support", "Frame":frame, "LikertOrder":"D2A",
                    "Prompt": f"{text}\n\nPlease indicate your agreement with “{b}.”",
                    "ShownOptions": LIKERT_D2A
                })
        return trials

    def _build_prompt(self, t: Dict) -> str:
        opts = "\n".join([opt for opt in t["ShownOptions"]])
        return f"{t['Prompt']}\n\nChoose exactly one option:\n{opts}\nAnswer with the option text only."

    def get_apply_chat_template(self): return True
    def get_clean_gpu_memory_before_finalize(self): return False
    def get_unique_name(self): return "gain_loss_persuasion"
    def get_max_len(self): return 200
    def get_progress(self): return self.counter / len(self.trials) if self.trials else 1.0
    def get_class_labels(self): return [], 0, 5

    def get_system_prompt(self, preffered_batch_size):
        return [""] * min(preffered_batch_size, len(self.trials) - self.counter)

    def get_user_prompt(self, preffered_batch_size):
        n = min(preffered_batch_size, len(self.trials) - self.counter)
        return [self._build_prompt(self.trials[self.counter + k]) for k in range(n)]

    def get_assistant_prompt(self, preffered_batch_size):
        return ["Answer:"] * min(preffered_batch_size, len(self.trials) - self.counter)

    def is_finished(self):
        return self.counter >= len(self.trials) or self.eval_only
   
    def process_results(self, llm_generations, think_parts, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for i, gen in enumerate(llm_generations):
            t = self.trials[self.counter + i]
            score = _parse_likert(gen, t["ShownOptions"])
            self.records.append({
                "Behavior": t["Behavior"],
                "Type": "Support",
                "Frame": t["Frame"],
                "LikertOrder": t["LikertOrder"],
                "Value": score,
                "Output": gen,
                "Thinking": think_parts[i],
            })
        self.counter += len(llm_generations)

    def finalize(self, save_path: Optional[str]=None):
        out_dir = save_path or "./results/GainLoss"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "gain_loss_persuasion.jsonl"), "w", encoding="utf-8") as f:
            for r in self.records: f.write(json.dumps(r)+"\n")
        print(f"[GainLoss] Raw JSONL saved -> {os.path.join(out_dir, 'gain_loss_persuasion.jsonl')}")
        open(os.path.join(out_dir, "done_eval.json"), "w").close()

if __name__ == "__main__":
    # Example usage
    evaluator = EmotionMatchPersuasionDataset(eval_only=False)
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