
import os
import json
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

# ---------------- Config ----------------
ROOT      = Path(__file__).resolve().parents[1]
# Use TF-native tiny model by default; override with env var if  want "gpt2"
MODEL_ID  = os.getenv("DPO_BASE_MODEL", "sshleifer/tiny-gpt2")
SFT_DIR   = ROOT / "out" / "sft"
DATA_FILE = ROOT / "D:\Important\Projects\shopping-choice-helper\data\prefs.jsonl"
OUT_DIR   = ROOT / "out" / "dpo"

MAX_LEN = 512
LR      = 5e-6
EPOCHS  = 1
BETA    = 0.1

print(" [DPO] Direct Preference Optimization (TF-only).")
print(f" Prefs file: {DATA_FILE}")
print(f" SFT dir:    {SFT_DIR}")
print(f" Base:       {MODEL_ID}")

if not DATA_FILE.exists():
    sys.exit("❌ data/prefs.jsonl missing. Format per line:\n"
             '{"prompt":"...","chosen":"...","rejected":"..."}')

# -------------- Loaders --------------
def load_tf_or_convert(path_or_id: str | Path) -> TFAutoModelForCausalLM:
    """Prefer TF weights; if missing, convert from PyTorch with safetensors fallbacks."""
    path_or_id = str(path_or_id)
    try:
        return TFAutoModelForCausalLM.from_pretrained(path_or_id, from_pt=False)
    except Exception as e1:
        print(f"[DPO] TF load failed for '{path_or_id}': {e1}")
        print("[DPO] Trying PT->TF conversion (safetensors=True)...")
        try:
            return TFAutoModelForCausalLM.from_pretrained(
                path_or_id, from_pt=True, use_safetensors=True, low_cpu_mem_usage=True
            )
        except Exception as e2:
            print(f"[DPO] safetensors=True failed: {e2}")
            print("[DPO] Retrying with safetensors=False...")
            return TFAutoModelForCausalLM.from_pretrained(
                path_or_id, from_pt=True, use_safetensors=False, low_cpu_mem_usage=True
            )

# Tokenizer from SFT if available, else from base
tok_source = SFT_DIR if SFT_DIR.exists() else MODEL_ID
tok = AutoTokenizer.from_pretrained(str(tok_source))
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print("📝 Tokenizer ready.")

# Policy = SFT if exists, else base; Reference = frozen base
policy    = load_tf_or_convert(SFT_DIR if SFT_DIR.exists() else MODEL_ID)
reference = load_tf_or_convert(MODEL_ID)
for v in reference.trainable_variables:
    v._trainable = False

# -------------- Utilities --------------
def encode_pair(prompt: str, completion: str):
    """Return concatenated ids and prompt length to compute completion logprob."""
    p_ids = tok(prompt.rstrip() + "\n", return_tensors="tf",
                truncation=True, max_length=MAX_LEN // 2)["input_ids"][0]
    c_ids = tok(completion.lstrip(), return_tensors="tf",
                truncation=True, max_length=MAX_LEN // 2)["input_ids"][0]
    ids = tf.concat([p_ids, c_ids], axis=0)
    return ids, tf.shape(p_ids)[0]

def logprob(model, ids, prompt_len):
    """Total log-prob of completion tokens under `model`."""
    out = model(ids[None, :], training=False)
    logits = out.logits[0]                 # [seq, vocab]
    comp_logits = logits[prompt_len-1:-1]  # align next-token
    comp_ids = ids[prompt_len:]
    ce = tf.keras.losses.sparse_categorical_crossentropy(
        comp_ids, comp_logits, from_logits=True
    )
    return -tf.reduce_sum(ce)

# -------------- Data --------------
pairs = []
with DATA_FILE.open("r", encoding="utf-8") as f:
    for ln, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            pairs.append(json.loads(line))
        except Exception as e:
            print(f"[DPO] Skipping bad JSON at line {ln}: {e}")
if not pairs:
    sys.exit("❌ No valid preference pairs found in prefs.jsonl")

# -------------- Train --------------
opt = tf.keras.optimizers.Adam(learning_rate=LR)

print("🚀 Training DPO...")
for epoch in range(EPOCHS):
    for i, ex in enumerate(pairs, 1):
        prompt   = ex["prompt"]
        chosen   = ex["chosen"]
        rejected = ex["rejected"]

        ch_ids, ch_p = encode_pair(prompt, chosen)
        rj_ids, rj_p = encode_pair(prompt, rejected)

        with tf.GradientTape() as tape:
            lp_ch  = logprob(policy, ch_ids, ch_p)
            lp_rj  = logprob(policy, rj_ids, rj_p)
            lpr_ch = logprob(reference, ch_ids, ch_p)
            lpr_rj = logprob(reference, rj_ids, rj_p)

            margin = (lp_ch - lp_rj) - (lpr_ch - lpr_rj)
            loss   = -tf.math.log_sigmoid(BETA * margin)  # DPO objective

        grads = tape.gradient(loss, policy.trainable_variables)
        opt.apply_gradients(zip(grads, policy.trainable_variables))

        if i % 10 == 0:
            print(f"  Step {i}/{len(pairs)} | DPO Loss = {float(loss):.4f}")

# -------------- Save --------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
policy.save_pretrained(OUT_DIR.as_posix())  # TF weights
tok.save_pretrained(OUT_DIR.as_posix())
print(f"✅ Saved DPO TF model to {OUT_DIR}")
print(f"   Base model: {MODEL_ID} | Used SFT: {SFT_DIR.exists()}")
