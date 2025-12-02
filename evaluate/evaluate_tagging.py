#!/usr/bin/env python3
"""Tagging Evaluation (Agent Invocation)

Reads evaluation dataset records and invokes the actual tagging agent tool to
produce a tag (new|gap|duplicate|conflict) for each generated story relative to
provided existing stories. Computes precision/recall/F1 metrics and writes
per-record predictions + aggregate metrics.
"""

import os
import json
import argparse
import re
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import sys

# Ensure project root is on path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.tagging_agent import create_tagging_agent

DATASET_PATH = os.path.join("eval/datasets", "eval_dataset.jsonl")  # Accept .json or .jsonl
EVAL_DIR = os.path.join("eval/results")
PREDICTIONS_JSONL = os.path.join(EVAL_DIR, "tagging_predictions.jsonl")
METRICS_JSON = os.path.join(EVAL_DIR, "tagging_f1.json")

DEFAULT_THRESHOLD = 0.6
TAG_VALUES = ["new", "gap", "duplicate", "conflict"]


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
    raw = open(path, "r").read().strip()
    # Support JSON array or JSONL lines
    objs: List[Dict[str, Any]] = []
    if raw.startswith("["):
        try:
            objs = json.loads(raw)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON array dataset: {e}")
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSON line: {e}: {line[:120]}")
    for obj in objs:
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


def compute_similarity(a: str, b: str) -> float:
    atoks = {t.lower() for t in re.split(r"\W+", a) if t}
    btoks = {t.lower() for t in re.split(r"\W+", b) if t}
    if not atoks or not btoks:
        return 0.0
    inter = len(atoks & btoks)
    union = len(atoks | btoks)
    return inter / union if union else 0.0


def build_agent_input(record: Record, threshold: float) -> Dict[str, Any]:
    similar_existing_stories = []
    for idx, es in enumerate(record.existing_stories, start=1):
        # Compute title and description similarity separately; use max so strong title match isn't diluted.
        title_sim = compute_similarity(record.story_title, es.get('title',''))
        desc_sim = compute_similarity(record.story_description, es.get('description',''))
        sim = max(title_sim, desc_sim)
        similar_existing_stories.append({
            "work_item_id": es.get("work_item_id", idx),
            "title": es.get("title", ""),
            "description": es.get("description", ""),
            "similarity": sim,
            "title_similarity": title_sim,
            "description_similarity": desc_sim,
        })
    return {
        "story": {
            "title": record.story_title,
            "description": record.story_description,
            "acceptance_criteria": record.story_acceptance_criteria,
        },
        "similar_existing_stories": similar_existing_stories,
        "similarity_threshold": threshold,
    }


def invoke_tagging_agent(tool_fn, payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = tool_fn(json.dumps(payload))
    try:
        return json.loads(raw)
    except Exception:
        return {"status": "error", "decision_tag": "new", "reason": "Agent JSON parse error", "related_ids": []}


def evaluate(records: List[Record], threshold: float) -> Dict[str, Any]:
    metrics_counts = {tag: {"TP": 0, "FP": 0, "FN": 0} for tag in TAG_VALUES}
    ensure_eval_dir()
    tagging_tool = create_tagging_agent(run_id="eval-run")
    with open(PREDICTIONS_JSONL, "w") as pred_f:
        for idx, record in enumerate(records, start=1):
            agent_payload = build_agent_input(record, threshold)
            agent_resp = invoke_tagging_agent(tagging_tool, agent_payload)
            prediction = {
                "decision_tag": agent_resp.get("decision_tag", "new"),
                "related_ids": agent_resp.get("related_ids", []),
                "reason": agent_resp.get("reason", "")
            }
            decision_tag = prediction.get("decision_tag", "new").lower()
            if decision_tag not in TAG_VALUES:
                decision_tag = "new"
            gold = record.gold_tag.lower()
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
    metrics = {}
    f1_values = []
    total = len(records)
    for tag, counts in metrics_counts.items():
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]
        tn = total - (tp + fp + fn)  # remaining samples where tag not predicted and not gold
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        metrics[tag] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        f1_values.append(f1)
    metrics["macro_f1"] = round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0
    # Store gold vs predicted counts for confusion matrix later
    metrics["_gold_pred_pairs"] = [
        {"gold": r.gold_tag.lower(), "pred": json.loads(line).get("predicted_tag") if False else None}
        for r in []
    ]  # placeholder (we will rebuild outside since we already wrote predictions file)
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate tagging agent")
    parser.add_argument("--config", default="config.poc.yaml", help="Config file (optional)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Similarity threshold for agent")
    args = parser.parse_args()
    print("Mode: Agent invocation")
    records = read_dataset(DATASET_PATH)
    print(f"Loaded {len(records)} records")
    metrics = evaluate(records, threshold=args.threshold)
    print("Metrics written to", METRICS_JSON)

    # Render pretty table for per-tag metrics
    def format_metrics_table(m: Dict[str, Any]) -> str:
        tag_rows = [t for t in TAG_VALUES if t in m]
        headers = ["Tag", "Precision", "Recall", "F1", "Accuracy", "TP", "FP", "FN", "TN"]
        rows = []
        for t in tag_rows:
            r = m[t]
            rows.append([
                t,
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}",
                f"{r['accuracy']:.4f}",
                str(r['tp']),
                str(r['fp']),
                str(r['fn']),
                str(r['tn'])
            ])
        # Column widths
        col_widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
        def fmt_row(cells):
            return " | ".join(cells[i].ljust(col_widths[i]) for i in range(len(cells)))
        sep = "-+-".join("-" * w for w in col_widths)
        out_lines = [fmt_row(headers), sep]
        out_lines += [fmt_row(r) for r in rows]
        # Macro row
        if 'macro_f1' in m:
            macro_f1 = m['macro_f1']
            macro_line = f"Macro F1: {macro_f1:.4f}"
            out_lines.append("" )
            out_lines.append(macro_line)
        return "\n".join(out_lines)

    # Show per-record gold vs predicted before summary metrics
    print("\nPer-Record Tag Results:")
    per_header = ["Idx", "Gold", "Pred", "Title"]
    per_rows = []
    with open(PREDICTIONS_JSONL, "r") as pf:
        for line in pf:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            idx = str(obj.get("index", "?"))
            gold = obj.get("gold_tag", "").lower()
            pred = obj.get("predicted_tag", "").lower()
            title = obj.get("story_title", "")[:50]
            per_rows.append([idx, gold, pred, title])
    # Column widths for per-record table
    if per_rows:
        col_w = [max(len(h), *(len(r[i]) for r in per_rows)) for i, h in enumerate(per_header)]
        def fmt_pr_row(c):
            return " | ".join(c[i].ljust(col_w[i]) for i in range(len(c)))
        sep_pr = "-+-".join("-" * w for w in col_w)
        print(fmt_pr_row(per_header))
        print(sep_pr)
        for r in per_rows:
            print(fmt_pr_row(r))
    else:
        print("(No prediction rows found)")

    print("\n" + format_metrics_table(metrics))

    # Build multi-class confusion matrix (gold vs predicted) from predictions file
    confusion = {g: {p: 0 for p in TAG_VALUES} for g in TAG_VALUES}
    gold_counts = {g: 0 for g in TAG_VALUES}
    pred_counts = {p: 0 for p in TAG_VALUES}
    with open(PREDICTIONS_JSONL, "r") as pf:
        for line in pf:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            gold = obj.get("gold_tag", "").lower()
            pred = obj.get("predicted_tag", "").lower()
            if gold in confusion and pred in confusion[gold]:
                confusion[gold][pred] += 1
            if gold in gold_counts:
                gold_counts[gold] += 1
            if pred in pred_counts:
                pred_counts[pred] += 1
    # Render confusion matrix
    header = ["Gold\\Pred"] + TAG_VALUES
    col_widths = [max(len(header[0]), 9)] + [max(len(t), 5) for t in TAG_VALUES]
    def fmt_row(cells):
        return " | ".join(cells[i].ljust(col_widths[i]) for i in range(len(cells)))
    sep = "-+-".join("-" * w for w in col_widths)
    matrix_lines = ["\nConfusion Matrix (Counts)", fmt_row(header), sep]
    for g in TAG_VALUES:
        row = [g] + [str(confusion[g][p]) for p in TAG_VALUES]
        matrix_lines.append(fmt_row(row))
    matrix_lines.append("")
    matrix_lines.append("Gold Distribution: " + ", ".join(f"{k}={v}" for k,v in gold_counts.items()))
    matrix_lines.append("Pred Distribution: " + ", ".join(f"{k}={v}" for k,v in pred_counts.items()))
    print("\n".join(matrix_lines))

    # Omit raw JSON dump per request; metrics persisted already.


if __name__ == "__main__":
    main()
