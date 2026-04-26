import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Folders
GOLD_DIR = Path("puzzles/annotations_json")
MODEL_DIR = Path("puzzles/gemma-4-31b-it/annotations_json")
RANDOM_DIR = Path("puzzles/random/annotations_json")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ---------- NORMALIZATION ----------

def get_gold_arrows(data):
    return set((a["from_square"], a["to_square"]) for a in data["arrows"])

def get_gold_candidate_arrows(data):
    return set((a["from_square"], a["to_square"]) 
               for a in data["arrows"] if a["kind"] == "candidate_move")

def get_gold_threat_arrows(data):
    return set((a["from_square"], a["to_square"]) 
               for a in data["arrows"] if a["kind"] == "threat")

def get_gold_key_squares(data):
    return set(x["square"] for x in data["highlighted_squares"])


def get_model_arrows(data):
    return set((a["from"], a["to"]) for a in data["candidate_move_arrows"]) | \
           set((a["from"], a["to"]) for a in data["threat_arrows"])

def get_model_candidate_arrows(data):
    return set((a["from"], a["to"]) for a in data["candidate_move_arrows"])

def get_model_threat_arrows(data):
    return set((a["from"], a["to"]) for a in data["threat_arrows"])

def get_model_key_squares(data):
    return set(data["key_squares"])


def get_random_arrows(data):
    return set((a["from"], a["to"]) for a in data["candidate_move_arrows"]) | \
           set((a["from"], a["to"]) for a in data["threat_arrows"])

def get_random_candidate_arrows(data):
    return set((a["from"], a["to"]) for a in data["candidate_move_arrows"])

def get_random_threat_arrows(data):
    return set((a["from"], a["to"]) for a in data["threat_arrows"])

def get_random_key_squares(data):
    return set(data["key_squares"])


# ---------- METRICS ----------

def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0

def recall(a, b):  # b relative to a
    return len(a & b) / len(a) if a else 0.0

def precision(a, b):
    return len(a & b) / len(b) if b else 0.0


# ---------- MAIN ----------

results = []

for gold_path in GOLD_DIR.glob("*.json"):
    pid = gold_path.stem

    model_path = MODEL_DIR / f"{pid}.json"
    random_path = RANDOM_DIR / f"{pid}.json"

    if not model_path.exists() or not random_path.exists():
        continue

    gold = load_json(gold_path)
    model = load_json(model_path)
    rand = load_json(random_path)

    gold_ar = get_gold_arrows(gold)
    model_ar = get_model_arrows(model)
    rand_ar = get_random_arrows(rand)

    gold_key = get_gold_key_squares(gold)
    model_key = get_model_key_squares(model)
    rand_key = get_random_key_squares(rand)

    # Pairwise metrics
    results.append({
        "id": pid,

        # --- GOLD vs MODEL ---
        "gm_arrow_jaccard": jaccard(gold_ar, model_ar),
        "gm_arrow_recall": recall(gold_ar, model_ar),
        "gm_arrow_precision": precision(gold_ar, model_ar),

        "gm_key_jaccard": jaccard(gold_key, model_key),
        "gm_key_recall": recall(gold_key, model_key),
        "gm_key_precision": precision(gold_key, model_key),

        # --- GOLD vs RANDOM ---
        "gr_arrow_jaccard": jaccard(gold_ar, rand_ar),
        "gr_arrow_recall": recall(gold_ar, rand_ar),
        "gr_arrow_precision": precision(gold_ar, rand_ar),

        "gr_key_jaccard": jaccard(gold_key, rand_key),
        "gr_key_recall": recall(gold_key, rand_key),
        "gr_key_precision": precision(gold_key, rand_key),

        # --- MODEL vs RANDOM ---
        "mr_arrow_jaccard": jaccard(model_ar, rand_ar),
        "mr_arrow_recall": recall(model_ar, rand_ar),
        "mr_arrow_precision": precision(model_ar, rand_ar),

        "mr_key_jaccard": jaccard(model_key, rand_key),
        "mr_key_recall": recall(model_key, rand_key),
        "mr_key_precision": precision(model_key, rand_key),
    })


df = pd.DataFrame(results)

# Aggregate
summary = df.mean(numeric_only=True)

print("\n=== SUMMARY ===")
print(summary.sort_values(ascending=False))

# Save
df.to_csv("annotation_alignment_per_puzzle.csv", index=False)
summary.to_csv("annotation_alignment_summary.csv")