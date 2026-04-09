"""
Query multiple LLMs with financial advice scenarios and collect responses.

Usage:
    python scripts/query_llms.py --scenarios data/scenarios/pilot_scenarios.json --models all
    python scripts/query_llms.py --scenarios data/scenarios/pilot_scenarios.json --models openai,anthropic
    python scripts/query_llms.py --scenarios data/scenarios/pilot_scenarios.json --models openai --runs 3
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful assistant. The user is asking for personal financial advice. "
    "Answer their question as helpfully as you can."
)

# Model configurations: (provider, model_id, display_name)
MODEL_REGISTRY = {
    "gpt-4o": ("openai", "gpt-4o", "GPT-4o"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini", "GPT-4o Mini"),
    "o3-mini": ("openai", "o3-mini", "o3-mini"),
    "claude-sonnet": ("anthropic", "claude-sonnet-4-20250514", "Claude Sonnet 4"),
    "claude-haiku": ("anthropic", "claude-haiku-4-20250414", "Claude Haiku 4"),
    "gemini-2.0-flash": ("google", "gemini-2.0-flash", "Gemini 2.0 Flash"),
    "gemini-2.5-pro": ("google", "gemini-2.5-pro-preview-05-06", "Gemini 2.5 Pro"),
    # Open-source via OpenRouter
    "llama-3.3-70b": ("openrouter", "meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B"),
    "deepseek-v3": ("openrouter", "deepseek/deepseek-chat-v3-0324", "DeepSeek V3"),
    "qwen-2.5-72b": ("openrouter", "qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B"),
}

PROVIDER_GROUPS = {
    "openai": [k for k, v in MODEL_REGISTRY.items() if v[0] == "openai"],
    "anthropic": [k for k, v in MODEL_REGISTRY.items() if v[0] == "anthropic"],
    "google": [k for k, v in MODEL_REGISTRY.items() if v[0] == "google"],
    "openrouter": [k for k, v in MODEL_REGISTRY.items() if v[0] == "openrouter"],
}


def query_openai(model_id: str, question: str) -> dict:
    from openai import OpenAI

    client = OpenAI()
    start = time.time()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=2000,
    )
    elapsed = time.time() - start
    return {
        "text": response.choices[0].message.content,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
        "latency_seconds": round(elapsed, 2),
    }


def query_anthropic(model_id: str, question: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    start = time.time()
    response = client.messages.create(
        model=model_id,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
        temperature=0.7,
        max_tokens=2000,
    )
    elapsed = time.time() - start
    return {
        "text": response.content[0].text,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "latency_seconds": round(elapsed, 2),
    }


def query_google(model_id: str, question: str) -> dict:
    from google import genai

    client = genai.Client()
    start = time.time()
    response = client.models.generate_content(
        model=model_id,
        contents=f"{SYSTEM_PROMPT}\n\n{question}",
    )
    elapsed = time.time() - start
    return {
        "text": response.text,
        "usage": {
            "input_tokens": getattr(response.usage_metadata, "prompt_token_count", None),
            "output_tokens": getattr(response.usage_metadata, "candidates_token_count", None),
        },
        "latency_seconds": round(elapsed, 2),
    }


def query_openrouter(model_id: str, question: str) -> dict:
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    start = time.time()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=2000,
    )
    elapsed = time.time() - start
    return {
        "text": response.choices[0].message.content,
        "usage": {
            "input_tokens": getattr(response.usage, "prompt_tokens", None),
            "output_tokens": getattr(response.usage, "completion_tokens", None),
        },
        "latency_seconds": round(elapsed, 2),
    }


QUERY_FNS = {
    "openai": query_openai,
    "anthropic": query_anthropic,
    "google": query_google,
    "openrouter": query_openrouter,
}


def query_model(model_key: str, question: str) -> dict:
    provider, model_id, display_name = MODEL_REGISTRY[model_key]
    query_fn = QUERY_FNS[provider]
    try:
        result = query_fn(model_id, question)
        result["model_key"] = model_key
        result["model_id"] = model_id
        result["display_name"] = display_name
        result["provider"] = provider
        result["error"] = None
        return result
    except Exception as e:
        return {
            "model_key": model_key,
            "model_id": model_id,
            "display_name": display_name,
            "provider": provider,
            "text": None,
            "usage": None,
            "latency_seconds": None,
            "error": str(e),
        }


def resolve_models(model_arg: str) -> list[str]:
    if model_arg == "all":
        return list(MODEL_REGISTRY.keys())
    models = []
    for token in model_arg.split(","):
        token = token.strip()
        if token in PROVIDER_GROUPS:
            models.extend(PROVIDER_GROUPS[token])
        elif token in MODEL_REGISTRY:
            models.append(token)
        else:
            print(f"Warning: unknown model/provider '{token}', skipping")
    return models


def main():
    parser = argparse.ArgumentParser(description="Query LLMs with financial advice scenarios")
    parser.add_argument("--scenarios", required=True, help="Path to scenarios JSON file")
    parser.add_argument("--models", default="all", help="Comma-separated model keys, provider names, or 'all'")
    parser.add_argument("--output", default="data/responses", help="Output directory for responses")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per scenario per model (for consistency testing)")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep between API calls (seconds)")
    args = parser.parse_args()

    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    models = resolve_models(args.models)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    all_responses = []

    total = len(scenarios) * len(models) * args.runs
    done = 0

    for scenario in scenarios:
        for model_key in models:
            for run_idx in range(args.runs):
                done += 1
                print(f"[{done}/{total}] {scenario['id']} x {model_key} (run {run_idx + 1})")

                result = query_model(model_key, scenario["question"])
                result["scenario_id"] = scenario["id"]
                result["category"] = scenario["category"]
                result["question"] = scenario["question"]
                result["run_index"] = run_idx
                result["timestamp"] = datetime.now(timezone.utc).isoformat()

                all_responses.append(result)

                if result["error"]:
                    print(f"  ERROR: {result['error']}")
                else:
                    word_count = len(result["text"].split())
                    print(f"  OK: {word_count} words, {result['latency_seconds']}s")

                time.sleep(args.sleep)

    out_path = output_dir / f"responses_{timestamp}.json"
    out_path.write_text(json.dumps(all_responses, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(all_responses)} responses to {out_path}")

    # Summary
    errors = sum(1 for r in all_responses if r["error"])
    print(f"Success: {len(all_responses) - errors}, Errors: {errors}")


if __name__ == "__main__":
    main()
