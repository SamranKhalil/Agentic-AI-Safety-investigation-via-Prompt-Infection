import os
import csv
import json
import time
import traceback
from datetime import datetime
from pypdf import PdfReader

from graph.pipeline import build_graph
from evaluation.evaluator import evaluate_scam, evaluate_theft

FAILED_PATH = "evaluation/failed_experiments.json"
RESULTS_PATH = "evaluation/experiment_results.csv"
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
GROQ_MODELS = ["qwen", "openai20b", "openai120b"]


def load_pdf(path: str, model_name: str = "", defense: str = "none") -> str:
    reader = PdfReader(path)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    if model_name == "qwen":
        return text[:2000]
    elif "sandwich" in defense:
        return text[:3000]
    return text[:6000]


def load_failed() -> list:
    if not os.path.exists(FAILED_PATH):
        print("No failed_experiments.json found.")
        return []
    try:
        with open(FAILED_PATH, "r") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        return []


def save_failed(failed: list):
    with open(FAILED_PATH, "w") as f:
        json.dump(failed, f, indent=2)


def append_to_csv(result: dict):
    file_exists = os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def run_single(entry: dict) -> dict:
    pipeline = entry["pipeline"]
    messaging_mode = entry["mode"]
    model_name = entry["model"]
    doc_path = entry["doc"]
    defense = entry.get("defense", "none")
    dataset = "non_replicating" if "non_replicating" in doc_path else "replicating"

    document = load_pdf(doc_path, model_name=model_name, defense=defense)

    # Try each Groq key in sequence on 429, gemini uses None
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
            graph = build_graph(pipeline)

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
                raise Exception(f"413_SKIP: {e}")
            elif "400" in error_str and "tool call validation" in error_str:
                raise Exception(f"400_SKIP: {e}")
            else:
                raise e

    raise Exception(f"All {len(keys_to_try)} Groq keys exhausted. Last: {last_error}")


def retry_all():
    failed = load_failed()
    if not failed:
        print("Nothing to retry.")
        return

    total = len(failed)
    still_failing = []

    print(f"\nRetrying {total} failed experiments...")
    print(f"Models in queue: {set(e['model'] for e in failed)}")

    for i, entry in enumerate(failed, 1):
        print(f"\n[{i}/{total}]")

        error_msg = entry.get("error", "")

        # Skip permanently unfixable errors without retrying
        if "413" in error_msg or ("400" in error_msg and "tool call" in error_msg):
            print(f"  Skipping permanently — {error_msg[:80]}")
            continue

        # Parse 429 wait time and sleep before retrying
        if "429" in error_msg and "Please try again in" in error_msg:
            try:
                time_str = (
                    error_msg.split("Please try again in")[1].split(".")[0].strip()
                )
                mins = int(time_str.split("m")[0]) if "m" in time_str else 0
                secs = (
                    int(time_str.split("m")[1].replace("s", ""))
                    if "m" in time_str
                    else int(time_str.replace("s", ""))
                )
                wait = mins * 60 + secs + 10
                wait = 50
                print(f"  429 detected — waiting {wait}s...")
                time.sleep(wait)
            except Exception:
                print(f"  429 detected — waiting 60s (default)...")
                time.sleep(60)

        # # Skip daily token limit errors — won't resolve until next day
        # if "tokens per day" in error_msg or "TPD" in error_msg:
        #     print(f"  Daily token limit — keeping for later retry.")
        #     still_failing.append(entry)
        #     continue

        try:
            result = run_single(entry)
            append_to_csv(result)
            print(f"  Success — appended to CSV")

        except Exception as e:
            error_str = str(e)
            print(f"  ERROR again: {error_str[:100]}")
            traceback.print_exc()

            # Don't re-add permanently unfixable errors
            if "413_SKIP" in error_str or "400_SKIP" in error_str:
                print(f"  Permanently skipping — not re-adding to failed list.")
            else:
                still_failing.append(
                    {
                        **entry,
                        "error": error_str,
                        "retry_time": datetime.now().isoformat(),
                    }
                )

        if i < total:
            print(f"  Waiting 30s...")
            time.sleep(30)

    save_failed(still_failing)

    succeeded = total - len(still_failing)
    print(f"\nDone. {succeeded}/{total} retried successfully.")
    print(f"{len(still_failing)} still failing — see {FAILED_PATH}")


if __name__ == "__main__":
    retry_all()
