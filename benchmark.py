"""
Benchmark script for Movie Mood Recommender app.

Measures average response time and token usage across multiple test scenarios
by calling the HuggingFace Inference API (same model used in app.py).

Usage:
    python benchmark.py                  # Run with defaults (5 iterations)
    python benchmark.py --iterations 10  # Custom iteration count
    python benchmark.py --local          # Benchmark the local model instead
"""

import argparse
import json
import os
import statistics
import time

# Load .env before importing anything that needs env vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app import (
    API_MODEL_NAME,
    LOCAL_MODEL_NAME,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    FEW_SHOT_EXAMPLES,
    build_messages,
    clean_output,
    parse_recommendation,
)

# ---------------------------------------------------------------------------
# Test scenarios covering different preference combinations
# ---------------------------------------------------------------------------
TEST_SCENARIOS = [
    {
        "name": "Horror fan - dark & intense",
        "answers": {
            "mood": "Dark & Intense",
            "genres": ["Horror", "Thriller"],
            "pace": "Fast-paced",
            "viewing_context": "Solo",
            "era": "Recent",
            "open_ended": "I love psychological horror. Recommend me some movies.",
        },
    },
    {
        "name": "Family comedy night",
        "answers": {
            "mood": "Light & Fun",
            "genres": ["Comedy", "Animation"],
            "pace": "Fast-paced",
            "viewing_context": "Family",
            "era": "2010s",
            "open_ended": "Looking for something the whole family can watch and laugh at.",
        },
    },
    {
        "name": "Deep drama - solo viewing",
        "answers": {
            "mood": "Emotional & Deep",
            "genres": ["Drama"],
            "pace": "Slow & character-driven",
            "viewing_context": "Solo",
            "era": "Classic",
            "open_ended": "I want a movie that will make me think. Something like Schindler's List.",
        },
    },
    {
        "name": "Sci-fi suspense with friends",
        "answers": {
            "mood": "Suspenseful",
            "genres": ["Science-Fiction", "Thriller"],
            "pace": "Balanced",
            "viewing_context": "Friends",
            "era": "2000s",
            "open_ended": "Recommend sci-fi movies with a twist ending.",
        },
    },
    {
        "name": "Inspirational romance",
        "answers": {
            "mood": "Inspirational",
            "genres": ["Romance", "Drama"],
            "pace": "Slow & character-driven",
            "viewing_context": "Any",
            "era": "Any Era",
            "open_ended": "I want a feel-good love story that leaves me smiling.",
        },
    },
]


def estimate_token_count(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English text."""
    return len(text) // 4


def run_api_benchmark(messages: list[dict], max_tokens: int, temperature: float, top_p: float, hf_token: str):
    """Run a single API inference call and return (response_text, usage_dict, elapsed_seconds)."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=hf_token, model=API_MODEL_NAME)

    start = time.perf_counter()
    response = ""
    prompt_tokens = 0
    completion_tokens = 0

    for chunk in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        if chunk.choices and chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
        # Try to capture usage from the final chunk
        if hasattr(chunk, "usage") and chunk.usage:
            prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

    elapsed = time.perf_counter() - start

    # Fallback: estimate tokens if the API didn't report them
    if prompt_tokens == 0:
        prompt_text = " ".join(m["content"] for m in messages)
        prompt_tokens = estimate_token_count(prompt_text)
    if completion_tokens == 0:
        completion_tokens = estimate_token_count(response)

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    return response, usage, elapsed


def run_local_benchmark(messages: list[dict], max_tokens: int, temperature: float):
    """Run a single local model inference call and return (response_text, usage_dict, elapsed_seconds)."""
    from app import local_pipeline

    if local_pipeline is None:
        raise RuntimeError("Local model pipeline not loaded. Set LOCAL_MODEL=true in your .env")

    prompt_text = " ".join(m["content"] for m in messages)
    prompt_tokens = estimate_token_count(prompt_text)

    start = time.perf_counter()
    result = local_pipeline(
        messages,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=max(temperature, 0.01),
    )
    elapsed = time.perf_counter() - start

    response = result[0]["generated_text"][-1]["content"]
    completion_tokens = estimate_token_count(response)

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    return response, usage, elapsed


def run_benchmarks(iterations: int = 5, use_local: bool = False):
    """Run benchmarks across all test scenarios and print a summary report."""
    hf_token = os.environ.get("HF_TOKEN", "")
    if not use_local and not hf_token:
        print("ERROR: HF_TOKEN environment variable not set. Required for API benchmarks.")
        print("Set it in your .env file or export it: export HF_TOKEN=hf_...")
        return

    model_name = LOCAL_MODEL_NAME if use_local else API_MODEL_NAME
    print(f"{'=' * 70}")
    print(f"  Movie Mood Recommender - Benchmark")
    print(f"  Model:      {model_name}")
    print(f"  Mode:       {'Local' if use_local else 'API'}")
    print(f"  Iterations: {iterations}")
    print(f"  Scenarios:  {len(TEST_SCENARIOS)}")
    print(f"{'=' * 70}\n")

    all_times = []
    all_prompt_tokens = []
    all_completion_tokens = []
    all_total_tokens = []
    parse_successes = 0
    total_runs = 0
    results_per_scenario = {}

    for scenario in TEST_SCENARIOS:
        name = scenario["name"]
        answers = scenario["answers"]
        messages = build_messages(answers)

        scenario_times = []
        scenario_prompt_tokens = []
        scenario_completion_tokens = []
        scenario_parse_ok = 0

        print(f"--- Scenario: {name} ---")

        for i in range(iterations):
            total_runs += 1
            print(f"  Run {i + 1}/{iterations}...", end=" ", flush=True)

            try:
                if use_local:
                    raw_response, usage, elapsed = run_local_benchmark(
                        messages, max_tokens=512, temperature=0.3,
                    )
                else:
                    raw_response, usage, elapsed = run_api_benchmark(
                        messages, max_tokens=512, temperature=0.3, top_p=0.95, hf_token=hf_token,
                    )

                cleaned = clean_output(raw_response)
                rec = parse_recommendation(cleaned)
                parsed_ok = rec is not None and len(rec.get("recommendations", [])) > 0

                scenario_times.append(elapsed)
                scenario_prompt_tokens.append(usage["prompt_tokens"])
                scenario_completion_tokens.append(usage["completion_tokens"])

                if parsed_ok:
                    scenario_parse_ok += 1
                    parse_successes += 1

                num_recs = len(rec["recommendations"]) if parsed_ok else 0
                print(f"  {elapsed:.2f}s | "
                      f"tokens: {usage['total_tokens']} "
                      f"(prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']}) | "
                      f"parsed: {'OK' if parsed_ok else 'FAIL'} ({num_recs} recs)")

            except Exception as e:
                print(f"  ERROR: {e}")

        if scenario_times:
            avg_time = statistics.mean(scenario_times)
            avg_prompt = statistics.mean(scenario_prompt_tokens)
            avg_completion = statistics.mean(scenario_completion_tokens)
            print(f"  >> Avg: {avg_time:.2f}s | "
                  f"prompt: {avg_prompt:.0f} | completion: {avg_completion:.0f} | "
                  f"parse success: {scenario_parse_ok}/{iterations}\n")

            results_per_scenario[name] = {
                "avg_time": avg_time,
                "avg_prompt_tokens": avg_prompt,
                "avg_completion_tokens": avg_completion,
                "parse_success_rate": scenario_parse_ok / iterations,
            }

            all_times.extend(scenario_times)
            all_prompt_tokens.extend(scenario_prompt_tokens)
            all_completion_tokens.extend(scenario_completion_tokens)
            all_total_tokens.extend(
                [p + c for p, c in zip(scenario_prompt_tokens, scenario_completion_tokens)]
            )

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    if not all_times:
        print("No successful runs to report.")
        return

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY ({total_runs} total runs)")
    print(f"{'=' * 70}")
    print(f"  Response Time (seconds)")
    print(f"    Mean:   {statistics.mean(all_times):.2f}")
    print(f"    Median: {statistics.median(all_times):.2f}")
    print(f"    Stdev:  {statistics.stdev(all_times):.2f}" if len(all_times) > 1 else "")
    print(f"    Min:    {min(all_times):.2f}")
    print(f"    Max:    {max(all_times):.2f}")

    print(f"\n  Token Usage")
    print(f"    Avg Prompt Tokens:     {statistics.mean(all_prompt_tokens):.0f}")
    print(f"    Avg Completion Tokens: {statistics.mean(all_completion_tokens):.0f}")
    print(f"    Avg Total Tokens:      {statistics.mean(all_total_tokens):.0f}")

    print(f"\n  Parse Success Rate: {parse_successes}/{total_runs} "
          f"({parse_successes / total_runs * 100:.1f}%)")

    print(f"\n  Per-Scenario Breakdown:")
    print(f"  {'Scenario':<35} {'Avg Time':>10} {'Prompt Tok':>12} {'Compl Tok':>12} {'Parse %':>10}")
    print(f"  {'-' * 79}")
    for name, data in results_per_scenario.items():
        print(f"  {name:<35} {data['avg_time']:>9.2f}s {data['avg_prompt_tokens']:>11.0f} "
              f"{data['avg_completion_tokens']:>11.0f} {data['parse_success_rate']:>9.0%}")
    print(f"{'=' * 70}")

    # Save results to JSON
    report = {
        "model": model_name,
        "mode": "local" if use_local else "api",
        "iterations_per_scenario": iterations,
        "total_runs": total_runs,
        "summary": {
            "avg_response_time_s": round(statistics.mean(all_times), 3),
            "median_response_time_s": round(statistics.median(all_times), 3),
            "min_response_time_s": round(min(all_times), 3),
            "max_response_time_s": round(max(all_times), 3),
            "avg_prompt_tokens": round(statistics.mean(all_prompt_tokens)),
            "avg_completion_tokens": round(statistics.mean(all_completion_tokens)),
            "avg_total_tokens": round(statistics.mean(all_total_tokens)),
            "parse_success_rate": round(parse_successes / total_runs, 3),
        },
        "per_scenario": results_per_scenario,
    }
    output_path = "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Movie Mood Recommender")
    parser.add_argument("--iterations", type=int, default=5, help="Runs per scenario (default: 5)")
    parser.add_argument("--local", action="store_true", help="Benchmark local model instead of API")
    args = parser.parse_args()

    run_benchmarks(iterations=args.iterations, use_local=args.local)
