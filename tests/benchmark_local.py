"""
Local performance benchmark for the Movie Mood Recommender model.

Downloads the model and runs inference locally. Measures latency,
token usage, memory consumption, and JSON validity.

Requires: pip install transformers torch psutil accelerate

Usage:
    python tests/benchmark_local.py                          # default small model
    python tests/benchmark_local.py --model Qwen/Qwen2.5-7B-Instruct  # full model (needs GPU)
"""

import argparse
import json
import os
import sys
import time
import statistics

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from appDraft import build_messages

# ── Test cases (same as API benchmark) ───────────────────────────────

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

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# ── Helpers ──────────────────────────────────────────────────────────


def get_gpu_memory_mb():
    """Return current GPU memory allocated in MB, or None if no CUDA."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return None


def get_ram_usage_mb():
    """Return current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


# ── Benchmark runner ─────────────────────────────────────────────────


def run_benchmark(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:  {device}")
    if device == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"Model:   {model_name}\n")

    # ── Load model ───────────────────────────────────────────────────
    ram_before_load = get_ram_usage_mb()
    gpu_before_load = get_gpu_memory_mb()

    print("Loading model and tokenizer...")
    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    load_elapsed = time.perf_counter() - load_start

    ram_after_load = get_ram_usage_mb()
    gpu_after_load = get_gpu_memory_mb()

    print(f"Model loaded in {load_elapsed:.1f}s")
    print(f"  RAM usage:  {ram_before_load:.0f} MB -> {ram_after_load:.0f} MB "
          f"(+{ram_after_load - ram_before_load:.0f} MB)")
    if gpu_after_load is not None:
        print(f"  GPU memory: {gpu_before_load:.0f} MB -> {gpu_after_load:.0f} MB "
              f"(+{gpu_after_load - gpu_before_load:.0f} MB)")
    print()

    # ── Run test cases ───────────────────────────────────────────────
    results = []
    print(f"Running benchmark with {len(TEST_CASES)} test cases...\n")
    print("-" * 70)

    for i, case in enumerate(TEST_CASES, 1):
        label = case["label"]
        messages = build_messages(case["answers"])

        # Apply the model's chat template
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        prompt_tokens = inputs["input_ids"].shape[1]

        # Measure inference
        ram_before = get_ram_usage_mb()
        gpu_before = get_gpu_memory_mb()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.02,
                top_p=0.95,
                do_sample=True,
            )
        elapsed = time.perf_counter() - start

        ram_after = get_ram_usage_mb()
        gpu_after = get_gpu_memory_mb()

        # Decode only the new tokens
        completion_tokens = outputs.shape[1] - prompt_tokens
        raw = tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)

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

        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

        result = {
            "label": label,
            "latency_s": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "tokens_per_sec": tokens_per_sec,
            "ram_delta_mb": ram_after - ram_before,
            "gpu_delta_mb": (gpu_after - gpu_before) if gpu_after is not None else None,
            "valid_json": valid_json,
            "has_required_keys": has_required_keys,
            "movie": parsed.get("recommended_movie", "N/A") if parsed else "PARSE FAIL",
            "raw": raw,
        }
        results.append(result)

        status = "OK" if has_required_keys else ("JSON OK" if valid_json else "FAIL")
        print(f"  [{i}/{len(TEST_CASES)}] {label:<30} {elapsed:6.2f}s  "
              f"{tokens_per_sec:5.1f} tok/s  "
              f"{status:<8} -> {result['movie']}")

    # ── Summary ──────────────────────────────────────────────────────
    print("-" * 70)
    print("\n=== SUMMARY ===\n")

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Model load time: {load_elapsed:.1f}s\n")

    latencies = [r["latency_s"] for r in results]
    print("Latency (seconds):")
    print(f"  Min:    {min(latencies):.2f}")
    print(f"  Max:    {max(latencies):.2f}")
    print(f"  Mean:   {statistics.mean(latencies):.2f}")
    print(f"  Median: {statistics.median(latencies):.2f}")
    if len(latencies) > 1:
        print(f"  Stdev:  {statistics.stdev(latencies):.2f}")

    tps = [r["tokens_per_sec"] for r in results]
    print(f"\nThroughput (tokens/sec):")
    print(f"  Min:    {min(tps):.1f}")
    print(f"  Max:    {max(tps):.1f}")
    print(f"  Mean:   {statistics.mean(tps):.1f}")

    prompt_tokens = [r["prompt_tokens"] for r in results]
    completion_tokens = [r["completion_tokens"] for r in results]
    total_tokens = [r["total_tokens"] for r in results]
    print(f"\nToken usage:")
    print(f"  Prompt tokens (mean):      {statistics.mean(prompt_tokens):.0f}")
    print(f"  Completion tokens (mean):   {statistics.mean(completion_tokens):.0f}")
    print(f"  Total tokens (mean):        {statistics.mean(total_tokens):.0f}")

    print(f"\nMemory:")
    print(f"  Model RAM footprint:  +{ram_after_load - ram_before_load:.0f} MB")
    if gpu_after_load is not None:
        print(f"  Model GPU footprint:  +{gpu_after_load - gpu_before_load:.0f} MB")
        peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  Peak GPU memory:       {peak_gpu:.0f} MB")

    valid_count = sum(1 for r in results if r["valid_json"])
    schema_count = sum(1 for r in results if r["has_required_keys"])
    total = len(results)
    print(f"\nReliability:")
    print(f"  Valid JSON:        {valid_count}/{total} ({valid_count / total * 100:.0f}%)")
    print(f"  Correct schema:    {schema_count}/{total} ({schema_count / total * 100:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local model benchmark")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    run_benchmark(args.model)
