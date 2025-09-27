import os, re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

# Check if transformers is installed
try:
    import transformers
    print("transformers version:", transformers.__version__)
except ImportError:
    print("transformers is not installed.")

BASE_MODEL_ID = "gpt2"  # small, TF-native

_NUM = r"(?:\d+(?:\.\d+)?)"

def _extract_number(s):
    try:
        return float(str(s).replace(",", ""))
    except Exception:
        return None

def parse_specs(text: str) -> dict:
    t = (text or "").lower()

    price = None
    m = re.search(r"(?:₹|rs\.?\s*|inr\s*|\$)\s*([0-9][0-9,\.]*)", t)
    if m: price = _extract_number(m.group(1))
    if price is None:  # bare number with 'k'
        m = re.search(rf"({_NUM})\s*k\b", t)
        if m: price = 1000 * _extract_number(m.group(1))

    # RAM before generic GB to avoid misread
    ram = None
    m = re.search(rf"({_NUM})\s*gb\s*ram\b", t)
    if m: ram = _extract_number(m.group(1))

    storage = None
    m = re.search(rf"({_NUM})\s*gb(?!\s*ram)", t)
    if m: storage = _extract_number(m.group(1))

    battery = None
    m = re.search(rf"({_NUM})\s*mah", t)
    if m: battery = _extract_number(m.group(1))

    camera_mp = None
    m = re.search(rf"({_NUM})\s*mp", t)
    if m: camera_mp = _extract_number(m.group(1))
    has_ois = bool(re.search(r"\bois\b", t))

    hz = None
    m = re.search(rf"({_NUM})\s*hz", t)
    if m: hz = _extract_number(m.group(1))
    panel = "AMOLED" if "amoled" in t else ("OLED" if "oled" in t else ("LCD" if "lcd" in t else None))

    return {
        "price": price,
        "battery_mAh": battery,
        "camera_mp": camera_mp,
        "camera_ois": has_ois,
        "display_hz": hz,
        "panel": panel,
        "storage_gb": storage,
        "ram_gb": ram,
    }

def _fmt(v, unit=""):
    if v is None: return "Unknown"
    if isinstance(v, float) and v.is_integer(): v = int(v)
    return f"{v}{unit}"

def build_table(a: dict, b: dict) -> str:
    rows = []

    def add(name, aval, bval, note):
        rows.append((name, aval, bval, note))

    if a["price"] or b["price"]:
        add(
            "Price",
            "₹" + _fmt(a["price"]) if a["price"] else "Unknown",
            "₹" + _fmt(b["price"]) if b["price"] else "Unknown",
            "A cheaper" if (a["price"] and b["price"] and a["price"] < b["price"]) else
            ("B cheaper" if (a["price"] and b["price"] and b["price"] < a["price"]) else "—")
        )
    if a["battery_mAh"] or b["battery_mAh"]:
        add(
            "Battery",
            _fmt(a["battery_mAh"], "mAh"),
            _fmt(b["battery_mAh"], "mAh"),
            "A lasts longer" if (a["battery_mAh"] and b["battery_mAh"] and a["battery_mAh"] > b["battery_mAh"]) else
            ("B lasts longer" if (a["battery_mAh"] and b["battery_mAh"] and b["battery_mAh"] > a["battery_mAh"]) else "—")
        )
    if any([a["camera_mp"], b["camera_mp"], a["camera_ois"], b["camera_ois"]]):
        a_cam = f"{_fmt(a['camera_mp'],'MP')}" + (" OIS" if a["camera_ois"] else "")
        b_cam = f"{_fmt(b['camera_mp'],'MP')}" + (" OIS" if b["camera_ois"] else "")
        note = "A higher MP" if (a["camera_mp"] and b["camera_mp"] and a["camera_mp"] > b["camera_mp"]) else \
               ("B higher MP" if (a["camera_mp"] and b["camera_mp"] and b["camera_mp"] > a["camera_mp"]) else "—")
        if b["camera_ois"] and not a["camera_ois"]: note = (note + "; " if note!="—" else "") + "B has OIS"
        if a["camera_ois"] and not b["camera_ois"]: note = (note + "; " if note!="—" else "") + "A has OIS"
        add("Camera", a_cam, b_cam, note)
    if any([a["display_hz"], b["display_hz"], a["panel"], b["panel"]]):
        a_disp = (f"{_fmt(a['display_hz'],'Hz')} " if a["display_hz"] else "") + (a["panel"] or "Unknown")
        b_disp = (f"{_fmt(b['display_hz'],'Hz')} " if b["display_hz"] else "") + (b["panel"] or "Unknown")
        note = "A smoother" if (a["display_hz"] and b["display_hz"] and a["display_hz"] > b["display_hz"]) else \
               ("B smoother" if (a["display_hz"] and b["display_hz"] and b["display_hz"] > a["display_hz"]) else "—")
        add("Display", a_disp.strip(), b_disp.strip(), note)
    if a["storage_gb"] or b["storage_gb"]:
        add(
            "Storage",
            _fmt(a["storage_gb"], "GB"),
            _fmt(b["storage_gb"], "GB"),
            "A more storage" if (a["storage_gb"] and b["storage_gb"] and a["storage_gb"] > b["storage_gb"]) else
            ("B more storage" if (a["storage_gb"] and b["storage_gb"] and b["storage_gb"] > a["storage_gb"]) else "—")
        )
    if a["ram_gb"] or b["ram_gb"]:
        add(
            "RAM",
            _fmt(a["ram_gb"], "GB"),
            _fmt(b["ram_gb"], "GB"),
            "A more RAM" if (a["ram_gb"] and b["ram_gb"] and a["ram_gb"] > b["ram_gb"]) else
            ("B more RAM" if (a["ram_gb"] and b["ram_gb"] and b["ram_gb"] > a["ram_gb"]) else "—")
        )

    if not rows:
        return "_No comparable attributes found._"
    md = ["| Attribute | Product A | Product B | Notes |", "|---|---|---|---|"]
    md += [f"| {n} | {av} | {bv} | {nt} |" for (n,av,bv,nt) in rows]
    return "\n".join(md)

WEIGHTS = {"price": 1.0, "battery": 1.0, "display": 0.8, "camera": 0.8, "storage": 0.6}

def score_products(a: dict, b: dict, priorities: str):
    prefs = [p.strip().lower() for p in (priorities or "").split(",") if p.strip()]
    w = WEIGHTS.copy()
    for idx, p in enumerate(prefs[:3]):
        if p in w: w[p] *= (1.4 - 0.2 * idx)

    def s_one(x, y):
        s = 0.0
        if x["price"] and y["price"] and x["price"] < y["price"]: s += w["price"]
        if x["battery_mAh"] and y["battery_mAh"] and x["battery_mAh"] > y["battery_mAh"]: s += w["battery"]
        if x["display_hz"] and y["display_hz"] and x["display_hz"] > y["display_hz"]: s += 0.7 * w["display"]
        if x["panel"] in ("AMOLED","OLED") and y["panel"] not in ("AMOLED","OLED"): s += 0.3 * w["display"]
        if x["camera_mp"] and y["camera_mp"] and x["camera_mp"] > y["camera_mp"]: s += 0.5 * w["camera"]
        if x["camera_ois"] and not y["camera_ois"]: s += 0.5 * w["camera"]
        if x["storage_gb"] and y["storage_gb"] and x["storage_gb"] > y["storage_gb"]: s += w["storage"]
        return s
    return s_one(a, b), s_one(b, a)

def _load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok

def _load_tf_model(model_id: str):
    try:
        return TFAutoModelForCausalLM.from_pretrained(model_id, from_pt=False)
    except Exception:
        return TFAutoModelForCausalLM.from_pretrained(model_id, from_pt=True)

class TFGenerator:
    def __init__(self, model_id: str = BASE_MODEL_ID):
        self.model_id = model_id
        self.tok = _load_tokenizer(model_id)
        self.model = _load_tf_model(model_id)

    def _proscons_prompt(self, table_md: str, priorities: str) -> str:
        return (
            "You are a shopping advisor. Only use the table below; do NOT invent specs.\n"
            f"User priorities (highest first): {priorities}\n\n"
            "TABLE (source of truth):\n"
            f"{table_md}\n\n"
            "Write exactly:\n"
            "1) **Pros A:** <≤3 items>; **Cons A:** <≤3 items>\n"
            "2) **Pros B:** <≤3 items>; **Cons B:** <≤3 items>\n"
            "3) Best fit: A or B — because <one sentence citing the top priorities>.\n"
            "No URLs, emails, or external claims.\n"
            "Answer:\n"
        )

    def _gen_text(self, prompt: str, deterministic: bool, temperature: float, top_p: float, max_tokens: int):
        inputs = self.tok(prompt, return_tensors="tf")
        ids = self.model.generate(
            **inputs,
            do_sample=not deterministic,
            temperature=temperature if not deterministic else None,
            top_p=top_p if not deterministic else None,
            max_new_tokens=max_tokens,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
        )
        text = self.tok.decode(ids[0], skip_special_tokens=True)
        return text[len(prompt):].strip()

    def generate(self, a: str, b: str, priorities: str, mode: str = "Factual",
                 temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 220) -> str:

        a_specs = parse_specs(a); b_specs = parse_specs(b)
        table_md = build_table(a_specs, b_specs)

        # 1) Deterministic first for anti-hallucination (Factual)
        deterministic = (mode == "Factual")
        prompt = self._proscons_prompt(table_md, priorities)
        commentary = self._gen_text(prompt, deterministic, temperature, top_p, max_tokens)

        # 2) Validate: must reference 'Best fit:' and not add URLs/emails
        bad = re.search(r"https?://|www\.|@[A-Za-z0-9_.-]+", commentary)
        if ("Best fit:" not in commentary) or bad:
            a_s, b_s = score_products(a_specs, b_specs, priorities)
            best = "A" if a_s >= b_s else "B"
            reasons = []
            if a_specs["price"] and b_specs["price"]:
                cheaper = "A" if a_specs["price"] < b_specs["price"] else "B"
                if cheaper == best: reasons.append("price")
            if a_specs["battery_mAh"] and b_specs["battery_mAh"]:
                batt = "A" if a_specs["battery_mAh"] > b_specs["battery_mAh"] else "B"
                if batt == best: reasons.append("battery")
            reason = ", ".join(reasons) or "overall balance vs your priorities"
            commentary = (
                "**Pros/Cons derived strictly from the table above.**\n\n"
                f"Best fit: {best} — because {reason}."
            )

        return f"{table_md}\n\n{commentary}\n\n_Citations: Based only on your provided specs table._"
