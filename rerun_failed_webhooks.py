import os
import csv
import json
import time
import traceback
import pandas as pd
from datetime import datetime
from pypdf import PdfReader

from graph.pipeline import build_graph
from evaluation.evaluator import evaluate_theft

CSV_PATH = "evaluation/experiment_results.csv"
FAILED_PATH = "evaluation/failed_experiments.json"

FIELDNAMES = [
    "dataset",
    "pipeline",
    "messaging_mode",
    "model",
    "defense",
    "document",
    "result",
    "success",
    "reason",
    "webhook_hit",
    "output_clean",
    "tool_called",
]

DOC_BASE_PATHS = {
    "scam": "data/infected_docs/replicating/scam",
    "theft": "data/infected_docs/replicating/theft",
}


def load_pdf(path: str, max_chars: int = 6000) -> str:
    reader = PdfReader(path)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text[:max_chars]


def load_failed() -> list:
    if not os.path.exists(FAILED_PATH):
        return []
    with open(FAILED_PATH, "r") as f:
        return json.load(f)


def save_failed(failed: list):
    with open(FAILED_PATH, "w") as f:
        json.dump(failed, f, indent=2)


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)

    # ── Dedup on load: prefer real results over placeholder failures ──
    key_cols = ["dataset", "pipeline", "messaging_mode", "model", "defense", "document"]
    df["_priority"] = (df["reason"] == "rate_limit_or_context_error").astype(int)
    df = df.sort_values("_priority").drop_duplicates(subset=key_cols, keep="first")
    df = df.drop(columns=["_priority"]).reset_index(drop=True)
    print(f"Loaded CSV: {len(df)} rows (after dedup)")
    return df


def save_csv(df: pd.DataFrame):
    df.to_csv(CSV_PATH, index=False)


def update_row(df: pd.DataFrame, key: dict, result: dict) -> pd.DataFrame:
    """Update a single row in the dataframe in place."""
    mask = (
        (df["dataset"] == key["dataset"])
        & (df["pipeline"] == key["pipeline"])
        & (df["messaging_mode"] == key["messaging_mode"])
        & (df["model"] == key["model"])
        & (df["defense"] == key["defense"])
        & (df["document"] == key["document"])
    )
    for col in [
        "result",
        "success",
        "reason",
        "webhook_hit",
        "output_clean",
        "tool_called",
    ]:
        if col in result:
            df.loc[mask, col] = result[col]
    return df


def run_single(row: pd.Series) -> dict:
    pipeline = row["pipeline"]
    messaging_mode = row["messaging_mode"]
    model_name = row["model"]
    defense = row["defense"]
    dataset = row["dataset"]
    doc_name = row["document"]

    doc_path = os.path.join(DOC_BASE_PATHS[pipeline], doc_name)

    print(f"\n{'='*60}")
    print(f"  Dataset  : {dataset}")
    print(f"  Pipeline : {pipeline}")
    print(f"  Mode     : {messaging_mode}")
    print(f"  Model    : {model_name}")
    print(f"  Defense  : {defense}")
    print(f"  Document : {doc_name}")
    print(f"{'='*60}")

    document = load_pdf(doc_path)
    graph = build_graph(pipeline)

    initial_state = {
        "messages": [{"role": "user", "content": document}],
        "messaging_mode": messaging_mode,
        "model_name": model_name,
        "tool_called": False,
        "infected": False,
        "defense": defense,
    }

    start = time.time()
    final_state = graph.invoke(initial_state)
    elapsed = round(time.time() - start, 2)

    final_output = final_state["messages"][-1]["content"]
    tool_called = final_state.get("tool_called", False)

    # Theft only — these are all theft rows
    evaluation = evaluate_theft(final_output, tool_called)

    result = {
        "result": evaluation["result"],
        "success": evaluation["success"],
        "reason": evaluation["reason"],
        "webhook_hit": evaluation.get("webhook_hit", False),
        "output_clean": evaluation.get("output_clean", False),
        "tool_called": tool_called,
    }

    print(f"\n  Result: {result['result'].upper()} — {result['reason']}")
    return result


def rerun_webhook_failures():
    df = load_csv()
    failed = load_failed()

    # Find all theft rows where reason is "webhook received no data"
    mask = (df["reason"] == "webhook received no data") & (df["pipeline"] == "theft")
    to_rerun = df[mask].copy()
    total = len(to_rerun)

    print(f"\nFound {total} rows with 'webhook received no data' to rerun")
    print(f"Breakdown by model:\n{to_rerun['model'].value_counts().to_string()}")

    changed = 0
    errored = 0

    for i, (idx, row) in enumerate(to_rerun.iterrows(), 1):
        print(f"\n[{i}/{total}]")
        try:
            result = run_single(row)

            key = {
                "dataset": row["dataset"],
                "pipeline": row["pipeline"],
                "messaging_mode": row["messaging_mode"],
                "model": row["model"],
                "defense": row["defense"],
                "document": row["document"],
            }

            # Update row in dataframe
            df = update_row(df, key, result)

            # Save CSV after every run so nothing is lost
            save_csv(df)

            if result["result"] != "failure":
                changed += 1
                print(f"  Changed: failure → {result['result'].upper()}")
            else:
                print(f"  No change: still failure")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            errored += 1
            failed.append(
                {
                    "dataset": row["dataset"],
                    "pipeline": row["pipeline"],
                    "mode": row["messaging_mode"],
                    "model": row["model"],
                    "defense": row["defense"],
                    "doc": os.path.join(
                        DOC_BASE_PATHS[row["pipeline"]], row["document"]
                    ),
                    "error": str(e),
                }
            )
            save_failed(failed)

        if i < total:
            print(f"  Waiting 15s...")
            time.sleep(60)

    print(f"\n{'='*60}")
    print(f"Done. {total} rerun | {changed} changed | {errored} errored")
    print(f"CSV saved to: {CSV_PATH}")
    if errored:
        print(f"Errors logged to: {FAILED_PATH}")


if __name__ == "__main__":
    rerun_webhook_failures()
