🛒 Shopping Choice Helper

A Generative AI mini-project showcasing LLM fine-tuning (SFT + DPO) with a clean Streamlit UI.

📌 Overview

Shopping Choice Helper is a Generative AI app that helps users compare two product descriptions and get a grounded recommendation.

This project applies concepts from the Generative AI with LLMs (DeepLearning.AI + AWS) course:

🔹 LLM inference with Transformers

🔹 Prompt engineering & grounding

🔹 Supervised Fine-Tuning (SFT)

🔹 RLHF-lite via Direct Preference Optimization (DPO)

🔹 Security & reliability (sanitization, PII redaction, fallback logic)

🔹 Scalable UI with Streamlit

✅ No hallucinations → all outputs are grounded in user-provided specs
✅ Multiple models: Base GPT-2, SFT, DPO
✅ Configurable tones: Factual, Balanced, Persuasive

✨ Features

Structured Comparison → converts free-text specs into a clean markdown table

Pros & Cons Generator → short, grounded bullet points for each product

Best Fit Selection → always ends with “Best fit: A or B — because …”

🔒 Security Layer

sanitize() → removes prompt injection attempts & URLs

redact_pii() → masks emails, phone numbers, IPs in outputs

Fallback Logic → ensures valid output if LLM drifts

🎭 Modes

Factual → deterministic, grounded

Balanced → mild creativity

Persuasive → expressive, user-facing tone

🧑‍💻 Training Modules
Supervised Fine-Tuning (SFT)

File: training/sft_tf.py

Dataset: data/sft_pairs.jsonl

Example:

{"prompt":"[Inputs]\nProduct A: ₹14999, 5000mAh, 50MP, 90Hz, 128GB\nProduct B: ₹15999, 6000mAh, 50MP OIS, 120Hz, 128GB\nUser priorities: battery, display\n[Output]\n",
 "response":"| Attribute | Product A | Product B | Notes | ... Best fit: B — because battery and display are top priorities."}


Run:

python training/sft_tf.py

Direct Preference Optimization (DPO)

File: training/dpo_tf.py

Dataset: data/prefs.jsonl

Example:

{"prompt":"Compare phones for battery and display with prices ₹14999 vs ₹15999.",
 "chosen":"Best fit: B — bigger battery and 120Hz display match priorities.",
 "rejected":"Both are fine; depends on taste."}


Run:

python training/dpo_tf.py

🏗️ Project Structure
shopping-choice-helper/
│── ui/                # Streamlit UI layer
│   └── app.py
│── core/              # Core logic
│   ├── infer_tf.py    # Model loading, parsing, fallback logic
│   ├── guard.py       # Sanitization & PII redaction
│── training/          # Training scripts
│   ├── sft_tf.py      # Supervised Fine-Tuning
│   ├── dpo_tf.py      # Direct Preference Optimization
│── data/              # Datasets
│   ├── sft_pairs.jsonl
│   └── prefs.jsonl
│── out/               # Saved models
│   ├── sft/           # Fine-tuned SFT model
│   └── dpo/           # DPO-aligned model
│── requirements.txt
│── README.md
└── .gitignore

🔄 Architecture
Mermaid Flow
flowchart LR
    A[User Input] --> B[Sanitize & Redact PII]
    B --> C[Specs Parsing & Table Building]
    C --> D[LLM (Base/SFT/DPO)]
    D --> E[Structured Response (Pros/Cons + Best Fit)]
    E --> F[Final Output to UI]

    style A fill:#e6f3ff,stroke:#3399ff,stroke-width:2px
    style B fill:#fff0e6,stroke:#ff9933,stroke-width:2px
    style C fill:#fff0e6,stroke:#ff9933,stroke-width:2px
    style D fill:#f3e6ff,stroke:#9933ff,stroke-width:2px
    style E fill:#e6ffe6,stroke:#33cc33,stroke-width:2px
    style F fill:#e6f3ff,stroke:#3399ff,stroke-width:2px

⚙️ Installation
# Clone repo
git clone https://github.com/prasanth-joe35/shopping-choice-helper.git
cd shopping-choice-helper

# Install dependencies
pip install -r requirements.txt

🚀 Usage

Run the UI:

streamlit run ui/app.py


Then:

Paste two product descriptions (A & B)

Pick your top priorities (e.g., price, battery, display)

Choose mode (Factual / Balanced / Persuasive)

Click 🔍 Compare Now

You’ll get:

📊 Markdown comparison table

✅ Pros & Cons for A and B

⭐ Best fit: A or B — with grounded reasoning

🔐 Security & Reliability

Sanitization → strips injections like “ignore previous instructions”

Redaction → removes PII (phones, emails, IPs)

Determinism → Factual mode disables randomness (temperature=0.15)

Fallback → Rule-based scorer guarantees valid Best Fit

🎯 Why This Project?

✅ Demonstrates end-to-end GenAI lifecycle (inference → SFT → RLHF-lite)

✅ Security-aware AI design with fallback logic

✅ Real-world use case: shopping assistant

✅ Built with modular, scalable architecture (UI ↔ core ↔ models ↔ training pipeline)

👤 Author

Developed as a mini-project to apply concepts from Generative AI with LLMs – DeepLearning.AI & AWS. by Prasanth.A
