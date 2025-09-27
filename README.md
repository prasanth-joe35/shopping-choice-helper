🛒 Shopping Choice Helper

A Generative AI mini-project showcasing LLM fine-tuning (SFT + DPO) with a clean Streamlit UI.

Overview:

Shopping Choice Helper is a Generative AI app that helps users compare two product descriptions and get a grounded recommendation.

It demonstrates practical skills from the Generative AI with LLMs (DeepLearning.AI + AWS) course:

LLM inference with Transformers

Prompt engineering & grounding

Supervised Fine-Tuning (SFT)

RLHF-lite via Direct Preference Optimization (DPO)

Security & reliability (sanitization, PII redaction, fallback logic)

Scalable UI with Streamlit

✅ No hallucinations → all outputs are based on user-provided specs
✅ Multiple models (Base GPT-2, SFT, DPO)
✅ Configurable style (Factual, Balanced, Persuasive)

Features:

Structured Comparison: Converts free-text specs into a clear markdown table

Pros & Cons Generator: Short, grounded bullet points for each product

Best Fit Selection: Always ends with “Best fit: A or B — because …”

Security Layer:

sanitize() removes prompt injection attempts & URLs

redact_pii() masks emails, phone numbers, IPs in outputs

Fallback Logic: Rule-based scorer ensures valid output if LLM drifts

🔧 Modes:

Factual → deterministic, grounded

Balanced → mild creativity

Persuasive → expressive, user-facing tone

Training Modules:

sft_tf.py → Supervised Fine-Tuning (teach structured output)

dpo_tf.py → Direct Preference Optimization (align with human preferences)

Project Structure:
shopping-choice-helper/
│
├── ui/                     # Streamlit UI layer
│   └── app.py
│
├── core/                   # Core logic
│   ├── infer_tf.py         # Model loading, parsing, fallback logic
│   ├── guard.py            # Sanitization & PII redaction
│   └── __init__.py
│
├── training/               # Training scripts
│   ├── sft_tf.py           # Supervised Fine-Tuning
│   ├── dpo_tf.py           # Direct Preference Optimization
│   └── __init__.py
│
├── data/                   # Datasets
│   ├── sft_pairs.jsonl     # Prompt → Response pairs for SFT
│   └── prefs.jsonl         # Preference pairs for DPO
│
├── out/                    # Saved models
│   ├── sft/                # Fine-tuned SFT model
│   └── dpo/                # DPO-aligned model
│
├── requirements.txt
├── README.md
└── .gitignore

graph TD
    %% Define Nodes
    A[User Interface (Streamlit)]
    B{core/guard.py\\n\\n- sanitize()\\n- redact_pii()}
    C[core/infer_tf.py (TFGenerator)\\n\\n- parse\_specs()\\n- build\_table()\\n- LLM Invocation + Fallback]
    D[Large Language Model (LLM)\\n\\n- Base Model (e.g., GPT-2)\\n- Fine-tuned (SFT)\\n- Preference (DPO)]
    E[Output Generation\\n\\n- Markdown Table\\n- Pros/Cons A & B\\n- Best Fit: A/B]
    F[User Interface (Streamlit)]

    ```mermaid
    
    %% Define Flow
    A -- User Input --> B
    B -- Clean Text --> C
    C -- Prompt & Specs --> D
    D -- Structured Response --> E
    E -- Final Output --> F

    %% Styling (Optional but recommended for clarity)
    style A fill:#e6f3ff,stroke:#3399ff,stroke-width:2px
    style B fill:#fff0e6,stroke:#ff9933,stroke-width:2px
    style C fill:#fff0e6,stroke:#ff9933,stroke-width:2px
    style D fill:#f3e6ff,stroke:#9933ff,stroke-width:2px
    style E fill:#e6ffe6,stroke:#33cc33,stroke-width:2px
    style F fill:#e6f3ff,stroke:#3399ff,stroke-width:2px
    ```
⚙️ Installation

Clone the repo:

git clone https://github.com/prasanth_joe35/shopping-choice-helper.git
cd shopping-choice-helper


Install dependencies:

pip install -r requirements.txt




Usage:
Run the UI
streamlit run ui/app.py


Paste two product descriptions (A & B)

Pick your top priorities (e.g., price, battery, display)

Choose mode (Factual / Balanced / Persuasive)

Click 🔍 Compare Now

You’ll get:

Markdown comparison table

Pros & Cons for A and B

Best fit: A or B — because …

 Training:
Supervised Fine-Tuning (SFT)

Format: data/sft_pairs.jsonl

{"prompt":"[Inputs]\nProduct A: ₹14999, 5000mAh, 50MP, 90Hz, 128GB\nProduct B: ₹15999, 6000mAh, 50MP OIS, 120Hz, 128GB\nUser priorities: battery, display\n[Output]\n","response":"| Attribute | Product A | Product B | Notes |\n|---|---|---|---|\nBattery | 5000mAh | 6000mAh | B lasts longer\nCamera | 50MP | 50MP OIS | B has OIS\nDisplay | 90Hz | 120Hz | B smoother\nPrice | ₹14999 | ₹15999 | A cheaper\nStorage | 128GB | 128GB | Tie\n\n**Pros A:** Cheaper; Light; Simple\n**Cons A:** Smaller battery; No OIS; 90Hz\n**Pros B:** Bigger battery; OIS; 120Hz\n**Cons B:** Pricier; Heavier; Overkill\n\nBest fit: B — because battery and display are top priorities."}


Run:

python training/sft_tf.py

Direct Preference Optimization (DPO)

Format: data/prefs.jsonl

{"prompt":"Compare phones for battery and display with prices ₹14999 vs ₹15999.","chosen":"Best fit: B — bigger battery and 120Hz display match priorities.","rejected":"Both are fine; depends on taste."}


Run:

python training/dpo_tf.py

Security & Reliability

Sanitization: Strips injections like “ignore previous instructions”

Redaction: Removes PII (phones, emails, IPs)

Determinism: Factual mode disables randomness (temperature = 0.15)

Fallback: If LLM output invalid → rule-based decision ensures “Best fit”

Why This Project?

✅ Showcases end-to-end GenAI lifecycle (inference, SFT, RLHF-lite)

✅ Demonstrates security-aware AI app design

✅ Provides a real-world use case (shopping assistant)

✅ Built with scalable design (UI ↔ core ↔ models ↔ training pipeline)

 Author:

Developed as a mini-project to apply concepts from
 Generative AI with LLMs – DeepLearning.AI & AWS
