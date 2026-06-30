"""
AI Copilot — rule-based, project-scoped knowledge assistant.
Answers ONLY questions about this project / model / metrics. Never invents facts:
unknown questions get an explicit "out of scope" answer.
"""
from __future__ import annotations

from core import engine

_KB = {
    "cnn-lstm": (
        "**CNN-LSTM** is the hybrid deep-learning model this system uses. The 1-D "
        "**CNN** layers extract local patterns from the electricity-consumption "
        "sequence; the **LSTM** layers then model the temporal dependencies across "
        "time. The network has two inputs — the scaled reading sequence and 59 "
        "engineered statistical features — and outputs a single sigmoid probability "
        "of electricity theft."
    ),
    "accuracy": (
        "**Accuracy** = (correct predictions) / (total predictions). It tells you the "
        "overall fraction the model got right, but on imbalanced theft data it can be "
        "misleading — always read it alongside precision, recall and ROC-AUC."
    ),
    "precision": (
        "**Precision** = TP / (TP + FP). Of all customers flagged as theft, how many "
        "really were. High precision means few false alarms — important to avoid "
        "wrongly accusing honest customers."
    ),
    "recall": (
        "**Recall** (sensitivity) = TP / (TP + FN). Of all actual theft cases, how "
        "many the model caught. High recall means few thieves slip through."
    ),
    "f1": (
        "**F1-score** is the harmonic mean of precision and recall: "
        "2·P·R/(P+R). A single balanced number, useful when you care about both "
        "false alarms and missed thefts."
    ),
    "roc": (
        "**ROC-AUC** is the area under the ROC curve — the probability that the model "
        "ranks a random theft case above a random normal one. 0.5 = random, 1.0 = "
        "perfect separation."
    ),
    "risk score": (
        "The **Risk Score** is the model's theft probability expressed as 0–100. "
        "≥75 = High, 40–74 = Medium, <40 = Low. It drives the red/green status badge."
    ),
    "threshold": (
        "The **decision threshold** (default 0.5) is the probability cut-off above "
        "which a customer is labelled Theft. Lower it to catch more theft (higher "
        "recall, more false alarms); raise it for fewer false alarms (higher precision)."
    ),
    "preprocessing": (
        "Before prediction each reading sequence is **per-row min-max scaled to [0,1]** "
        "(exactly as in training) and 59 statistical features are extracted and "
        "StandardScaler-normalised. Skipping this scaling makes the model predict "
        "everything as Normal."
    ),
    "strategy": (
        "When an uploaded sequence length differs from what the model expects, a "
        "**preprocessing strategy** maps it: last_n (keep most recent), truncate "
        "(keep first), pad (zero-fill), interpolate (resample), or sliding_window "
        "(scan and aggregate)."
    ),
    "shap": (
        "Explanations here use **integrated gradients** over the model's own sequence "
        "input — real gradients, not a surrogate — plus rule-based risk factors "
        "derived from the statistical features."
    ),
    "theft": (
        "A customer is classified **Theft (Class 1)** when the model's probability is "
        "≥ the decision threshold. Typical theft signatures: long zero-consumption "
        "runs, sudden sustained drops, abnormally low or erratic usage."
    ),
    "normal": (
        "A **Normal (Class 0)** customer shows stable, regular consumption with no "
        "theft signature; the model's probability is below the threshold."
    ),
}

_GREET = {"hi", "hello", "hey", "salam", "مرحبا", "السلام عليكم", "اهلا"}


def answer(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "Ask me about the model, a prediction, or any metric (accuracy, recall, ROC-AUC, risk score…)."
    if q in _GREET:
        return "Hello! I'm the ETD-XAI Copilot. Ask me about the CNN-LSTM model, a prediction result, or any evaluation metric."

    # model-status questions
    if any(k in q for k in ("which model", "what model", "active model", "model loaded")):
        if engine.is_model_loaded():
            i = engine.get_model_info()
            return (f"The active model is **{i['model_name']}** ({i['architecture']}), "
                    f"loaded via `tensorflow.keras.models.load_model()`. Input {i['input_shape']}, "
                    f"output {i['output_shape']}, {i['total_params_fmt']} parameters. "
                    f"Every prediction comes from this model only — no fallbacks.")
        return engine.NO_MODEL_MSG

    # knowledge base
    keys = [
        ("cnn-lstm", ("cnn", "lstm", "architecture", "neural", "network", "deep learning")),
        ("accuracy", ("accuracy",)),
        ("precision", ("precision",)),
        ("recall", ("recall", "sensitivity")),
        ("f1", ("f1", "f-1", "f score")),
        ("roc", ("roc", "auc")),
        ("risk score", ("risk score", "risk")),
        ("threshold", ("threshold", "cut-off", "cutoff")),
        ("preprocessing", ("preprocess", "scal", "normaliz", "feature engineering")),
        ("strategy", ("strategy", "strategies", "length", "resize")),
        ("shap", ("shap", "explain", "xai", "interpret", "gradient")),
        ("theft", ("why theft", "theft", "class 1", "steal")),
        ("normal", ("normal", "class 0")),
    ]
    for kb_key, triggers in keys:
        if any(t in q for t in triggers):
            return _KB[kb_key]

    return (
        "I can only answer questions about **this project** — the CNN-LSTM theft-"
        "detection model, its predictions, preprocessing, and the evaluation metrics "
        "(accuracy, precision, recall, F1, ROC-AUC, risk score). Could you rephrase "
        "your question around one of those topics?"
    )


SUGGESTIONS = [
    "Explain the CNN-LSTM model",
    "What is the Risk Score?",
    "Explain Recall vs Precision",
    "How does the decision threshold work?",
    "Why is a customer classified as theft?",
    "What preprocessing is applied?",
]
