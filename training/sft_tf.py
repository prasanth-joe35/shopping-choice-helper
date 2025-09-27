
import os
import json
from pathlib import Path

# Keep TF logs quieter
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

# ------------------- Config -------------------
ROOT = Path(__file__).resolve().parents[1]     # project root
# Change to "gpt2" if  insist on GPT-2; tiny model is faster & TF-native.
MODEL_ID = os.getenv("SFT_BASE_MODEL", "sshleifer/tiny-gpt2")

DATA_FILE = ROOT /"D:\Important\Projects\shopping-choice-helper\data\sft_pairs.jsonl"
OUT_DIR   = ROOT / "out" / "sft"

MAX_LEN   = 512
LR        = 5e-5
EPOCHS    = 2
BATCH_SIZE = 1

# ------------------- Guards -------------------
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"sft_pairs.jsonl not found at: {DATA_FILE}\n"
        "Create it with one JSON per line, e.g.\n"
        '{"prompt":"cheap + big battery", "response":"Choose A: lower price and 6000mAh."}'
    )

# ------------------- Tokenizer ----------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ------------------- Model Loader ----------------
def load_tf_or_convert(model_id: str) -> TFAutoModelForCausalLM:
    """
    Try to load native TF weights; if unavailable, convert from PyTorch.
    Handles common safetensors issues gracefully.
    """
    try:
        # Prefer TF weights (no conversion path -> most stable)
        return TFAutoModelForCausalLM.from_pretrained(model_id, from_pt=False)
    except Exception as e1:
        print(f"[SFT] TF weights not found or failed for '{model_id}': {e1}")
        print("[SFT] Trying PyTorch -> TF conversion with safetensors=True ...")
        try:
            return TFAutoModelForCausalLM.from_pretrained(
                model_id, from_pt=True, use_safetensors=True, low_cpu_mem_usage=True
            )
        except Exception as e2:
            print(f"[SFT] safetensors=True failed: {e2}")
            print("[SFT] Retrying conversion with safetensors=False ...")
            return TFAutoModelForCausalLM.from_pretrained(
                model_id, from_pt=True, use_safetensors=False, low_cpu_mem_usage=True
            )

model = load_tf_or_convert(MODEL_ID)

# ------------------- Data Pipeline ----------------
def iter_examples():
    """Yield tokenized sequences (prompt + newline + response)."""
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception as e:
                print(f"[SFT] Skipping bad JSON on line {i}: {e}")
                continue
            prompt = (ex.get("prompt") or "").rstrip()
            resp   = (ex.get("response") or "").lstrip()
            text = prompt + "\n" + resp
            ids = tok(
                text,
                return_tensors="tf",
                truncation=True,
                max_length=MAX_LEN
            )["input_ids"][0]
            yield ids

def tf_dataset():
    ds = tf.data.Dataset.from_generator(
        iter_examples,
        output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
    ds = ds.padded_batch(
        BATCH_SIZE,
        padded_shapes=[MAX_LEN],
        padding_values=tok.pad_token_id,
    ).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = tf_dataset()

# ------------------- Optimizer/Train Step ----------------
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

@tf.function
def train_step(input_ids):
    with tf.GradientTape() as tape:
        out = model(input_ids, labels=input_ids, training=True)
        loss = out.loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# ------------------- Train ----------------
for epoch in range(EPOCHS):
    for step, batch in enumerate(train_ds, 1):
        loss = train_step(batch)
        if step % 20 == 0:
            print(f"[SFT] Epoch {epoch+1} | Step {step} | Loss {float(loss):.4f}")

# ------------------- Save ----------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT_DIR.as_posix())   # saves TF weights (no PT files needed afterward)
tok.save_pretrained(OUT_DIR.as_posix())
print(f" Saved SFT TF model to {OUT_DIR}")
print(f"   Base model: {MODEL_ID}")
