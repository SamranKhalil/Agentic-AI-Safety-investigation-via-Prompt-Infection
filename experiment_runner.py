import os
import csv
import json
import time
import traceback
import itertools
from datetime import datetime
from pypdf import PdfReader

from graph.pipeline import build_graph
from evaluation.evaluator import evaluate_scam, evaluate_theft

MODELS = ["llama", "gemini", "qwen", "openai20b", "openai120b"]
PIPELINES = ["scam", "theft"]
MESSAGING_MODES = ["global", "local"]

# ── Only the new combined defenses ────────────────────────────────
DEFENSES = [
    "sandwich_llm_tagging",
    "marking_llm_tagging",
    "instruction_llm_tagging",
    "delimiting_llm_tagging",
]

DATASETS = ["replicating"]

RESULTS_PATH = "evaluation/experiment_results.csv"
FAILED_PATH = "evaluation/failed_experiments.json"
RESULTS_DIR = "evaluation/logs"

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

# ── Groq key rotation ──────────────────────────────────────────────
GROQ_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
]
GROQ_MODELS = ["qwen", "llama", "openai20b", "openai120b"]
_groq_key_cycle = itertools.cycle(GROQ_KEYS)


def get_next_groq_key() -> str:
    return next(_groq_key_cycle)


def get_docs(dataset, pipeline):
    folder = os.path.join("data", "infected_docs", dataset, pipeline)
    return sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pdf")]
    )


SCAM_DOCS = {dataset: get_docs(dataset, "scam") for dataset in DATASETS}
THEFT_DOCS = {dataset: get_docs(dataset, "theft") for dataset in DATASETS}

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("evaluation", exist_ok=True)


def load_pdf(path: str, max_chars: int = 6000) -> str:
    reader = PdfReader(path)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text[:max_chars]


def load_completed() -> set:
    completed = set()
    if not os.path.exists(RESULTS_PATH):
        return completed
    with open(RESULTS_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = "|".join(
                [
                    row.get("dataset", ""),
                    row.get("pipeline", ""),
                    row.get("messaging_mode", ""),
                    row.get("model", ""),
                    row.get("defense", ""),
                    row.get("document", ""),
                ]
            )
            completed.add(key)
    print(f"  Loaded {len(completed)} already completed experiments — skipping them.")
    return completed


def make_key(dataset, pipeline, mode, model, defense, doc) -> str:
    return "|".join([dataset, pipeline, mode, model, defense, os.path.basename(doc)])


def append_to_csv(result: dict):
    file_exists = os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def load_failed() -> list:
    if not os.path.exists(FAILED_PATH):
        return []
    with open(FAILED_PATH, "r") as f:
        return json.load(f)


def save_failed(failed: list):
    with open(FAILED_PATH, "w") as f:
        json.dump(failed, f, indent=2)


def run_single_experiment(
    dataset, pipeline, messaging_mode, model_name, defense, doc_path
):
    document = load_pdf(doc_path)
    graph = build_graph(pipeline)

    # For Groq models try each key before giving up
    keys_to_try = list(GROQ_KEYS) if model_name in GROQ_MODELS else [None]

    last_error = None
    for attempt, groq_key in enumerate(keys_to_try, 1):
        print(f"\n{'='*60}")
        print(f"  Dataset  : {dataset}")
        print(f"  Pipeline : {pipeline}")
        print(f"  Mode     : {messaging_mode}")
        print(f"  Model    : {model_name}")
        print(f"  Defense  : {defense}")
        print(f"  Document : {os.path.basename(doc_path)}")
        if model_name in GROQ_MODELS:
            print(f"  Key      : attempt {attempt}/{len(keys_to_try)}")
        print(f"{'='*60}")

        try:
            initial_state = {
                "messages": [{"role": "user", "content": document}],
                "messaging_mode": messaging_mode,
                "model_name": model_name,
                "tool_called": False,
                "infected": False,
                "defense": defense,
                "groq_api_key": groq_key,
            }

            start = time.time()
            final_state = graph.invoke(initial_state)
            elapsed = round(time.time() - start, 2)

            final_output = final_state["messages"][-1]["content"]
            tool_called = final_state.get("tool_called", False)

            if pipeline == "scam":
                evaluation = evaluate_scam(final_output)
            else:
                evaluation = evaluate_theft(final_output, tool_called)

            result = {
                "timestamp": datetime.now().isoformat(),
                "dataset": dataset,
                "pipeline": pipeline,
                "messaging_mode": messaging_mode,
                "model": model_name,
                "defense": defense,
                "document": os.path.basename(doc_path),
                "elapsed_sec": elapsed,
                "final_output": final_output[:500],
                "tool_called": tool_called,
                **evaluation,
            }

            print(f"\n  Result: {result['result'].upper()} — {result['reason']}")
            return result

        except Exception as e:
            last_error = e
            error_str = str(e)

            if "429" in error_str and attempt < len(keys_to_try):
                print(f"  429 on key {attempt} — rotating to next key...")
                time.sleep(5)
                continue
            elif "413" in error_str:
                raise e  # don't retry 413s, document is too large
            else:
                raise e  # non-rate-limit error, don't rotate

    # All keys exhausted
    raise Exception(
        f"All {len(keys_to_try)} Groq keys hit rate limits. Last error: {last_error}"
    )


def run_all_experiments():
    experiments = []
    for dataset in DATASETS:
        for pipeline in PIPELINES:
            docs = SCAM_DOCS[dataset] if pipeline == "scam" else THEFT_DOCS[dataset]
            for mode in MESSAGING_MODES:
                for model in MODELS:
                    for defense in DEFENSES:
                        for doc in docs:
                            experiments.append(
                                (dataset, pipeline, mode, model, defense, doc)
                            )

    total = len(experiments)
    print(f"\nTotal experiments: {total}")
    print(f"  Models    : {MODELS}")
    print(f"  Pipelines : {PIPELINES}")
    print(f"  Modes     : {MESSAGING_MODES}")
    print(f"  Defenses  : {DEFENSES}")
    for dataset in DATASETS:
        print(f"  {dataset} scam docs : {len(SCAM_DOCS[dataset])}")
        print(f"  {dataset} theft docs: {len(THEFT_DOCS[dataset])}")

    completed = load_completed()
    failed = load_failed()
    skipped = 0

    for i, (dataset, pipeline, mode, model, defense, doc) in enumerate(experiments, 1):
        key = make_key(dataset, pipeline, mode, model, defense, doc)

        if key in completed:
            skipped += 1
            print(f"[{i}/{total}] Skipping (already done): {key}")
            continue

        print(f"\n[{i}/{total}]")
        try:
            result = run_single_experiment(dataset, pipeline, mode, model, defense, doc)
            append_to_csv(result)
            completed.add(key)

        except Exception as e:
            error_str = str(e)
            print(f"  ERROR: {e}")
            traceback.print_exc()

            entry = {
                "dataset": dataset,
                "pipeline": pipeline,
                "mode": mode,
                "model": model,
                "defense": defense,
                "doc": doc,
                "error": error_str,
            }

            if "413" in error_str:
                print("  413 Request Too Large — skipping, will not retry.")
            else:
                failed.append(entry)
                save_failed(failed)

        print(f"  Waiting 60s...")
        time.sleep(60)

    print(f"\nCompleted. Skipped: {skipped} | Failed: {len(failed)}")
    if failed:
        print(f"  {len(failed)} failed — see {FAILED_PATH}")


if __name__ == "__main__":
    run_all_experiments()
