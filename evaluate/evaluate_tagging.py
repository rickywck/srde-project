#!/usr/bin/env python3
"""
Tagging Evaluation Script (Section 8.2)

Loads a fixed tagging test dataset (datasets/tagging_test.jsonl), calls the Tagging Agent LLM
for each record (unless --offline is specified), and computes precision/recall/F1 per tag and macro average.

Usage:
    python evaluate_tagging.py [--config config.poc.yaml] [--offline] [--model gpt-4o]

Outputs:
    eval/tagging_f1.json           -> aggregate metrics
    eval/tagging_predictions.jsonl -> per-record prediction details

Offline Mode:
    If --offline is provided (or no OPENAI_API_KEY is set), a simple heuristic tagger is used:
      - If any existing story title shares a significant keyword with generated story title -> gap
      - If generated story implies replacement or stronger policy vs existing story -> conflict
      - Else new
    This allows running evaluation without external API calls.
"""
import os
import re
import json
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass

try:
    from openai import OpenAI  # OpenAI SDK (already used elsewhere in repo)
except ImportError:  # Graceful failure; offline mode still works
    OpenAI = None  # type: ignore

DATASET_PATH = os.path.join("datasets", "tagging_test.json")
EVAL_DIR = os.path.join("eval")
PREDICTIONS_JSONL = os.path.join(EVAL_DIR, "tagging_predictions.json")
METRICS_JSON = os.path.join(EVAL_DIR, "tagging_f1.json")

DEFAULT_MODEL = "gpt-4o"

TAG_VALUES = ["new", "gap", "conflict"]

@dataclass
class Record:
    story_title: str
    story_description: str
    story_acceptance_criteria: List[str]
    existing_stories: List[Dict[str, Any]]
    gold_tag: str
    gold_related_ids: List[str]


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        import yaml
        return yaml.safe_load(f)


def read_dataset(path: str) -> List[Record]:
    records: List[Record] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(Record(
                story_title=obj["story_title"],
                story_description=obj["story_description"],
                story_acceptance_criteria=obj["story_acceptance_criteria"],
                existing_stories=obj["existing_stories"],
                gold_tag=obj["gold_tag"],
                gold_related_ids=obj.get("gold_related_ids", [])
            ))
    return records


def ensure_eval_dir():
    os.makedirs(EVAL_DIR, exist_ok=True)


def build_prompt(record: Record) -> str:
    existing_fmt = []
    for i, es in enumerate(record.existing_stories, start=1):
        ac = " | ".join(es.get("acceptance_criteria", []))
        existing_fmt.append(f"[{i}] Title: {es['title']}\nDescription: {es['description']}\nAC: {ac}")
    existing_block = "\n\n".join(existing_fmt) if existing_fmt else "(None)"
    story_ac = " | ".join(record.story_acceptance_criteria)
    instructions = (
        "You are a tagging agent. Classify the generated story relative to existing stories.\n"
        "Exactly one tag: new, gap, conflict.\n"
        "Definitions:\n"
        "- new: No sufficiently similar existing story.\n"
        "- gap: Extends or complements functionality in one or more existing stories.\n"
        "- conflict: Overlaps or contradicts an existing story's scope or intent (eg replacement, stronger policy).\n"
        "Return strict JSON with keys: decision_tag, related_ids (list of existing story titles referenced), reason (short)."
    )
    prompt = (
        f"{instructions}\n\nGenerated Story:\nTitle: {record.story_title}\nDescription: {record.story_description}\nAcceptance Criteria: {story_ac}\n\nExisting Stories:\n{existing_block}\n\nJSON:" 
    )
    return prompt


def call_openai(prompt: str, model: str) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not available; install the openai package or use --offline")
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a precise JSON-producing assistant."}, {"role": "user", "content": prompt}],
        temperature=0.0,
    )
    content = response.choices[0].message.content.strip()
    # Attempt JSON parse; if fails, try to extract JSON substring
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}\Z", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Model output not valid JSON: {content[:200]}")


def heuristic_tagger(record: Record) -> Dict[str, Any]:
    title = record.story_title.lower()
    existing_titles = [es['title'].lower() for es in record.existing_stories]

    # Conflict indicators: replacement, stronger policy keywords
    conflict_keywords = ["replace", "switch", "strong", "enforce", "migrate"]
    if any(kw in title for kw in conflict_keywords):
        related = [es['title'] for es in record.existing_stories if any(kw in es['title'].lower() for kw in conflict_keywords) or any(word in title for word in es['title'].lower().split())]
        return {"decision_tag": "conflict", "related_ids": related[:2], "reason": "Heuristic conflict keyword match"}

    # Gap: share at least one significant keyword (len>5) with existing title
    title_tokens = [t for t in re.split(r"\W+", title) if len(t) > 5]
    for et, es in zip(existing_titles, record.existing_stories):
        et_tokens = [t for t in re.split(r"\W+", et) if len(t) > 5]
        if set(title_tokens) & set(et_tokens):
            return {"decision_tag": "gap", "related_ids": [es['title']], "reason": "Shared significant keyword"}

    return {"decision_tag": "new", "related_ids": [], "reason": "No keyword overlap"}


def evaluate(records: List[Record], model: str, offline: bool) -> Dict[str, Any]:
    metrics_counts = {tag: {"TP": 0, "FP": 0, "FN": 0} for tag in TAG_VALUES}
    ensure_eval_dir()
    with open(PREDICTIONS_JSONL, "w") as pred_f:
        for idx, record in enumerate(records, start=1):
            if offline:
                prediction = heuristic_tagger(record)
            else:
                prompt = build_prompt(record)
                try:
                    prediction = call_openai(prompt, model=model)
                except Exception as e:
                    prediction = {"decision_tag": "new", "related_ids": [], "reason": f"Fallback due to error: {e}"}
            decision_tag = prediction.get("decision_tag", "new").lower()
            if decision_tag not in TAG_VALUES:
                decision_tag = "new"

            gold = record.gold_tag.lower()
            # Update counts
            for tag in TAG_VALUES:
                if gold == tag and decision_tag == tag:
                    metrics_counts[tag]["TP"] += 1
                elif gold != tag and decision_tag == tag:
                    metrics_counts[tag]["FP"] += 1
                elif gold == tag and decision_tag != tag:
                    metrics_counts[tag]["FN"] += 1

            out_obj = {
                "index": idx,
                "story_title": record.story_title,
                "gold_tag": gold,
                "predicted_tag": decision_tag,
                "prediction": prediction,
                "gold_related_ids": record.gold_related_ids,
            }
            pred_f.write(json.dumps(out_obj) + "\n")

    # Compute precision/recall/F1
    metrics = {}
    f1_values = []
    for tag, counts in metrics_counts.items():
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        metrics[tag] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        f1_values.append(f1)
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    metrics["macro_f1"] = round(macro_f1, 4)

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate tagging agent on fixed dataset")
    parser.add_argument("--config", default="config.poc.yaml", help="Path to config file")
    parser.add_argument("--offline", action="store_true", help="Use heuristic tagger instead of LLM")
    parser.add_argument("--model", default=None, help="Override chat model name")
    args = parser.parse_args()

    config = load_config(args.config)
    model = args.model or config.get("openai", {}).get("chat_model", DEFAULT_MODEL)

    offline = args.offline or (os.getenv(config.get("openai", {}).get("api_key_env_var", "OPENAI_API_KEY")) is None)
    if offline:
        print("Running in OFFLINE heuristic mode (no OpenAI API key detected or --offline specified)")
    else:
        if OpenAI is None:
            print("OpenAI SDK not installed; falling back to offline mode")
            offline = True

    records = read_dataset(DATASET_PATH)
    print(f"Loaded {len(records)} evaluation records")
    metrics = evaluate(records, model=model, offline=offline)
    print("Evaluation complete. Metrics written to eval/tagging_f1.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
