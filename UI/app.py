import os, sys
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.guard import sanitize, redact_pii
from core.infer_tf import TFGenerator, BASE_MODEL_ID

st.set_page_config(page_title="🛍️ Shopping Choice Helper", layout="centered")
st.markdown("<h1 style='text-align:center;color:#2E86C1'>🛒 Shopping Choice Helper</h1>", unsafe_allow_html=True)
st.caption("Paste two short product specs, pick your priorities, and get a grounded recommendation.")

choices = [("Base (GPT-2)", BASE_MODEL_ID)]
sft_dir = os.path.join(PROJECT_ROOT, "out", "sft")
dpo_dir = os.path.join(PROJECT_ROOT, "out", "dpo")
if os.path.isdir(sft_dir): choices.append(("SFT (out/sft)", sft_dir))
if os.path.isdir(dpo_dir): choices.append(("DPO (out/dpo)", dpo_dir))

with st.sidebar:
    st.subheader("⚙️ Controls")
    mode = st.radio("Style / Tone", ["Factual", "Balanced", "Persuasive"], index=0)
    temp_map = {"Factual": 0.15, "Balanced": 0.35, "Persuasive": 0.6}
    temperature = temp_map[mode]
    top_p = 0.9
    priorities = st.multiselect(
        "Your priorities (highest first)",
        ["price", "battery", "camera", "display", "storage"],
        default=["price", "battery"]
    )
    model_label = st.selectbox("Model", options=[lbl for (lbl, _) in choices], index=0)
    model_id = dict(choices)[model_label]
    st.caption("Factual = deterministic; Balanced/Persuasive allow mild creativity.")

cols = st.columns(2)
with cols[0]:
    prod_a = st.text_area("📦 Product A", height=140, placeholder="₹14999, 5000mAh, 50MP, 90Hz, 128GB, AMOLED")
with cols[1]:
    prod_b = st.text_area("📦 Product B", height=140, placeholder="₹15999, 6000mAh, 50MP OIS, 120Hz, 128GB, AMOLED")

@st.cache_resource(show_spinner=False)
def get_generator(mid: str) -> TFGenerator:
    return TFGenerator(model_id=mid)

if "gen_model_id" not in st.session_state or st.session_state.gen_model_id != model_id:
    with st.spinner(f"Loading model: {model_label} ..."):
        st.session_state.gen = get_generator(model_id)
        st.session_state.gen_model_id = model_id

if st.button("🔍 Compare Now", type="primary"):
    if not prod_a.strip() or not prod_b.strip():
        st.warning("Please paste both product descriptions.")
    else:
        a = sanitize(prod_a); b = sanitize(prod_b)
        prio_str = ", ".join(priorities) if priorities else "price"
        with st.spinner("Analyzing..."):
            raw = st.session_state.gen.generate(
                a=a, b=b, priorities=prio_str, mode=mode,
                temperature=temperature, top_p=top_p, max_tokens=220
            )
        st.markdown(redact_pii(raw))
        st.success(f"✅ Generated with **{model_label}** | Mode: **{mode}**")
