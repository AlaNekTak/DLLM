import argparse, json, re, sys
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "Qwen/Qwen3-8B"  # small, instruct-tuned Qwen3
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a precise writing-quality evaluator. "
    "Evaluate the provided text and return ONLY a valid JSON object that follows this schema:\n"
    "{\n"
    '  "fluency": <integer 1-5>,\n'
    '  "grammar": <integer 1-5>,\n'
    '  "coherence": <integer 1-5>,\n'
    # '  "readability": <integer 1-5>,\n'
    '  "overall": <integer 1-10>,\n'
    # '  "rationale": "<=60 words, concise justification>"\n'
    "}\n"
    "Rules: 1) Output pure JSON (no markdown, no extra text). "
    "2) Do not show any intermediate steps. 3) Use integers only for scores."
)

EVAL_INSTRUCTIONS = (
    "Evaluate the following synthetic text using G-Eval-style criteria:\n"
    "- Fluency (1–5): smoothness and flow; natural phrasing, no awkwardness.\n"
    "- Grammar (1–5): correctness of syntax, tense, agreement, punctuation.\n"
    "- Coherence (1–5): logical organization; ideas connect and progress sensibly.\n"
    # "- Readability (1–5): clarity and ease for a general audience; appropriate vocabulary.\n"
    "- Overall (1–10): holistic quality as writing (not factual accuracy).\n"
)

def _maybe_from_file(maybe_path: str) -> str:
    if maybe_path.startswith("@"):
        path = maybe_path[1:]
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return maybe_path

def build_message(text: str) -> List[Dict[str, str]]:
    user_content = (
        f"{EVAL_INSTRUCTIONS}\n"
        "Here is the text to evaluate delimited by triple fences:\n"
        "```text\n"
        f"{text}\n"
        "```\n"
        "Return ONLY the JSON object."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

def extract_json(s: str) -> Dict[str, Any]:
    """
    Robustly extract the first JSON object from a string.
    """
    # If the model behaved, the whole output is JSON already:
    try:
        return json.loads(s)
    except Exception:
        pass

    # Strip code fences if any:
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE | re.DOTALL)

    # Try first/last brace heuristic:
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            # As a last resort, try to remove trailing commas
            candidate2 = re.sub(r",\s*([\]}])", r"\1", candidate)
            return json.loads(candidate2)
        except:
            pass
    
    return {'fluency': 0, 'grammar': 0, 'coherence': 0,  'overall': 0} #'readability': 0,, 'rationale': "Failed to parse the input."}


def load_model_and_tokenizer(model_name: str, load_in_8bit: bool, load_in_4bit: bool):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    kwargs = dict(torch_dtype="auto", device_map="auto")
    if load_in_8bit:
        kwargs.update(dict(load_in_8bit=True))
    if load_in_4bit:
        kwargs.update(dict(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tok



def g_eval(texts: str, model, tok, temperature: float = 0.0,
           top_p: float = 1.0, seed: int | None = 7,
           load_in_8bit: bool = False, load_in_4bit: bool = False, batch_size : int = 4) -> Dict[str, Any]:

    prompts = []
    for text in texts:
        message = build_message(text)
        prompt = tok.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)

    gen = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=temperature > 0.0,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )

    with torch.no_grad():
        prompt_lens = (inputs.input_ids != tok.pad_token_id).sum(dim=1)    
    
    results = []
    for i in range(len(texts)):
        out_ids = gen[i, prompt_lens[i]:]
        raw = tok.decode(out_ids, skip_special_tokens=True).strip()
        data = extract_json(raw)

        # Optional: light validation & clipping
        for k in ["fluency", "grammar", "coherence"]:#, "readability"]:
            if k in data:
                data[k] = int(max(1, min(5, int(data[k]))))
        if "overall" in data:
            data["overall"] = int(max(1, min(10, int(data["overall"]))))
        # if "rationale" in data and isinstance(data["rationale"], str):
        #     data["rationale"] = data["rationale"].strip()
        results.append(data)
    return results

class QEvaluator:
    def __init__(self):
        self.model, self.tok = load_model_and_tokenizer('Qwen/Qwen3-8B', False, True)
    
    def geval(self, texts):
        return g_eval(
            texts=texts,
            model=self.model, tok=self.tok,
            temperature=0,
            top_p=1.0,
            seed=100,
        )        


if __name__ == '__main__':
    qe = QEvaluator()
    res = qe.geval(['Hey, hellllllo.', 'Good morning! How can I help you in this beautiful day?'])
    print(res)


    # %%



