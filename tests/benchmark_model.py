"""
Performance benchmark for the Movie Mood Recommender model.

Measures latency, token usage, and JSON validity across diverse test cases.
Requires HF_TOKEN (via .env or environment variable).

Usage:
    python tests/benchmark_model.py
"""

import json
import os
import sys
import time
import statistics

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from huggingface_hub import InferenceClient
from appDraft import build_messages

# ── Test cases ───────────────────────────────────────────────────────

TEST_CASES = [
    {
        "label": "Happy / Comedy",
        "answers": {
            "mood": "happy",
            "genres": ["Comedy"],
            "pace": "medium",
            "spectacle_vs_story": "balanced",
            "familiar_vs_new": "familiar",
            "open_ended": "I enjoyed The Grand Budapest Hotel.",
        },
    },
    {
        "label": "Scary / Horror+Thriller",
        "answers": {
            "mood": "scary",
            "genres": ["Horror", "Thriller"],
            "pace": "fast",
            "spectacle_vs_story": "story",
            "familiar_vs_new": "new",
            "open_ended": "I like tense movies with clever twists.",
        },
    },
    {
        "label": "Adventurous / Action+Sci-Fi",
        "answers": {
            "mood": "adventurous",
            "genres": ["Action", "Sci-Fi"],
            "pace": "fast",
            "spectacle_vs_story": "spectacle",
            "familiar_vs_new": "new",
            "open_ended": "I loved Interstellar.",
        },
    },
    {
        "label": "Sad / Drama",
        "answers": {
            "mood": "sad",
            "genres": ["Drama"],
            "pace": "slow",
            "spectacle_vs_story": "story",
            "familiar_vs_new": "familiar",
            "open_ended": "Something like Moonlight or Manchester by the Sea.",
        },
    },
    {
        "label": "Chill / Animation+Comedy",
        "answers": {
            "mood": "chill",
            "genres": ["Animation", "Comedy"],
            "pace": "medium",
            "spectacle_vs_story": "balanced",
            "familiar_vs_new": "no preference",
            "open_ended": "",
        },
    },
]

# ── Benchmark runner ─────────────────────────────────────────────────


def run_benchmark():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set. Add it to .env or set it as an env var.")
        sys.exit(1)

    client = InferenceClient(
        provider="together",
        token=token,
        model="Qwen/Qwen2.5-7B-Instruct",
    )

    results = []

    print(f"Running benchmark with {len(TEST_CASES)} test cases...\n")
    print("-" * 70)

    for i, case in enumerate(TEST_CASES, 1):
        label = case["label"]
        messages = build_messages(case["answers"])

        start = time.perf_counter()
        response = client.chat_completion(
            messages,
            max_tokens=256,
            temperature=0.02,
            top_p=0.95,
        )
        elapsed = time.perf_counter() - start

        raw = response.choices[0].message.content
        usage = response.usage

        # Check JSON validity
        parsed = None
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

        valid_json = parsed is not None
        has_required_keys = (
            valid_json
            and "recommended_movie" in parsed
            and "year" in parsed
            and "why" in parsed
        )

        result = {
            "label": label,
            "latency_s": elapsed,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
            "valid_json": valid_json,
            "has_required_keys": has_required_keys,
            "movie": parsed.get("recommended_movie", "N/A") if parsed else "PARSE FAIL",
            "raw": raw,
        }
        results.append(result)

        status = "OK" if has_required_keys else ("JSON OK" if valid_json else "FAIL")
        print(f"  [{i}/{len(TEST_CASES)}] {label:<30} {elapsed:5.2f}s  "
              f"tokens={usage.total_tokens if usage else '?':>4}  "
              f"{status:<8} -> {result['movie']}")

    # ── Summary ──────────────────────────────────────────────────────

    print("-" * 70)
    print("\n=== SUMMARY ===\n")

    latencies = [r["latency_s"] for r in results]
    print(f"Latency (seconds):")
    print(f"  Min:    {min(latencies):.2f}")
    print(f"  Max:    {max(latencies):.2f}")
    print(f"  Mean:   {statistics.mean(latencies):.2f}")
    print(f"  Median: {statistics.median(latencies):.2f}")
    if len(latencies) > 1:
        print(f"  Stdev:  {statistics.stdev(latencies):.2f}")

    prompt_tokens = [r["prompt_tokens"] for r in results if r["prompt_tokens"] is not None]
    completion_tokens = [r["completion_tokens"] for r in results if r["completion_tokens"] is not None]
    total_tokens = [r["total_tokens"] for r in results if r["total_tokens"] is not None]

    if total_tokens:
        print(f"\nToken usage:")
        print(f"  Prompt tokens (mean):     {statistics.mean(prompt_tokens):.0f}")
        print(f"  Completion tokens (mean):  {statistics.mean(completion_tokens):.0f}")
        print(f"  Total tokens (mean):       {statistics.mean(total_tokens):.0f}")
        print(f"  Total tokens (sum):        {sum(total_tokens)}")

    valid_count = sum(1 for r in results if r["valid_json"])
    schema_count = sum(1 for r in results if r["has_required_keys"])
    total = len(results)
    print(f"\nReliability:")
    print(f"  Valid JSON:        {valid_count}/{total} ({valid_count/total*100:.0f}%)")
    print(f"  Correct schema:    {schema_count}/{total} ({schema_count/total*100:.0f}%)")


if __name__ == "__main__":
    run_benchmark()
