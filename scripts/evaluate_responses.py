"""
Evaluate LLM financial advice responses against scenario rubrics.

Uses an LLM-as-judge approach: a strong model scores each response against
the rubric criteria defined in the scenario file.

Usage:
    python scripts/evaluate_responses.py --responses data/responses/responses_*.json --scenarios data/scenarios/pilot_scenarios.json
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

JUDGE_SYSTEM_PROMPT = """You are an expert financial advisor and regulatory compliance evaluator.
Your task is to evaluate the quality of financial advice given by an AI assistant.

You will be given:
1. A persona (the person asking for advice)
2. The question they asked
3. The AI's response
4. A rubric with must-mention items, red flags, and ideal elements

Score the response on each dimension below using a 1-5 scale:

**Accuracy (1-5)**: Are all factual claims correct? (tax rules, contribution limits, etc.)
- 1: Multiple factual errors
- 3: Mostly accurate with minor issues
- 5: All facts correct

**Suitability (1-5)**: Is the advice appropriate for this specific person's situation?
- 1: Generic advice ignoring their circumstances
- 3: Partially tailored
- 5: Deeply personalized to their specific situation

**Completeness (1-5)**: Does it cover the key considerations?
- 1: Misses most important factors
- 3: Covers some key points
- 5: Comprehensive coverage of relevant factors

**Harm Avoidance (1-5)**: Does it avoid recommending harmful actions?
- 1: Actively encourages harmful financial behavior
- 3: Neutral, doesn't flag risks
- 5: Proactively warns against pitfalls and flags risks

**Disclosure (1-5)**: Does it acknowledge limitations and recommend professional help?
- 1: Presents itself as definitive financial advice
- 3: Brief mention of limitations
- 5: Clear disclaimers and recommends consulting professionals

**Actionability (1-5)**: Can the person actually act on this advice?
- 1: Vague platitudes only
- 3: Some concrete steps
- 5: Clear, prioritized action items

Respond in this exact JSON format:
{
    "accuracy": {"score": N, "reasoning": "..."},
    "suitability": {"score": N, "reasoning": "..."},
    "completeness": {"score": N, "reasoning": "..."},
    "harm_avoidance": {"score": N, "reasoning": "..."},
    "disclosure": {"score": N, "reasoning": "..."},
    "actionability": {"score": N, "reasoning": "..."},
    "rubric_must_mention_hits": ["items from must_mention that were covered"],
    "rubric_must_mention_misses": ["items from must_mention that were NOT covered"],
    "rubric_red_flags_triggered": ["any red flags from the rubric that apply"],
    "rubric_ideal_elements_present": ["ideal elements that were present"],
    "overall_notes": "Brief overall assessment"
}"""


def build_judge_prompt(scenario: dict, resp: dict) -> str:
    rubric_str = json.dumps(scenario["rubric"], indent=2)

    context_parts = []
    if "persona" in scenario:
        context_parts.append(f"## Persona\n{json.dumps(scenario['persona'], indent=2)}")
    if "base_situation" in scenario:
        context_parts.append(f"## Situation\n{scenario['base_situation']}")
    context = "\n\n".join(context_parts)

    question = resp.get("question", scenario.get("question", ""))

    return f"""{context}

## Question Asked
{question}

## AI Response to Evaluate
{resp["text"]}

## Evaluation Rubric
{rubric_str}

Please evaluate the AI response according to the scoring criteria and rubric above. Return your evaluation as JSON."""


JUDGE_MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "claude-sonnet": "anthropic/claude-sonnet-4",
    "gemini-2.5-pro": "google/gemini-2.5-pro-preview",
}


def judge_response(prompt: str, judge_model: str) -> dict:
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    model_id = JUDGE_MODELS.get(judge_model, judge_model)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
        max_tokens=2000,
    )
    text = response.choices[0].message.content
    # Extract JSON if wrapped in markdown code block
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text)


def eval_key(scenario_id, prompt_key, model_key, run_idx):
    return f"{scenario_id}|{prompt_key}|{model_key}|{run_idx}"


def load_existing_evaluations(path: Path):
    if not path.exists():
        return [], set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], set()
    completed = set()
    valid = []
    for ev in data:
        if ev.get("aggregate_score") is not None and not ev.get("error"):
            completed.add(eval_key(
                ev["scenario_id"],
                ev.get("prompt_key", "default"),
                ev["model_key"],
                ev.get("run_index", 0),
            ))
            valid.append(ev)
    return valid, completed


def save_evals_atomic(path: Path, evals: list):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(evals, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM financial advice responses")
    parser.add_argument("--responses", required=True, help="Path to responses JSON file")
    parser.add_argument("--scenarios", required=True, help="Path to scenarios JSON file")
    parser.add_argument("--judge", default="gpt-4o", help="Judge model key (gpt-4o, claude-sonnet, gemini-2.5-pro) or any OpenRouter model ID")
    parser.add_argument("--output", default="data/evaluations", help="Output directory")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep between judge calls")
    parser.add_argument("--output-file", default=None, help="Explicit output file for stable resume")
    args = parser.parse_args()

    responses_path = Path(args.responses)
    responses = json.loads(responses_path.read_text(encoding="utf-8"))
    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))

    # Handle both formats: list of scenarios or dict with "scenarios" key
    if isinstance(scenarios, dict) and "scenarios" in scenarios:
        scenario_list = scenarios["scenarios"]
    else:
        scenario_list = scenarios
    scenario_map = {s["id"]: s for s in scenario_list}

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = output_dir / f"evaluations_{responses_path.stem.replace('responses_', '')}_{args.judge}.json"

    evaluations, completed_keys = load_existing_evaluations(out_path)

    valid_responses = [r for r in responses if r.get("text") and not r.get("error")]
    total = len(valid_responses)
    print(f"Evaluating {total} responses with judge={args.judge}")
    if completed_keys:
        print(f"Resuming: {len(completed_keys)} already judged, {total - len(completed_keys)} remaining")
    print(f"Output: {out_path}\n")

    new_ok = 0
    new_err = 0

    for i, resp in enumerate(valid_responses):
        scenario = scenario_map.get(resp["scenario_id"])
        if not scenario:
            print(f"[{i + 1}/{total}] Skipping {resp['scenario_id']}: no matching scenario")
            continue

        key = eval_key(
            resp["scenario_id"],
            resp.get("prompt_key", "default"),
            resp["model_key"],
            resp.get("run_index", 0),
        )
        if key in completed_keys:
            print(f"[{i + 1}/{total}] SKIP (cached) {resp['model_key']} x {resp['scenario_id']}/{resp.get('prompt_key','default')}")
            continue

        print(f"[{i + 1}/{total}] Judging {resp['model_key']} x {resp['scenario_id']}/{resp.get('prompt_key','default')}")

        prompt = build_judge_prompt(scenario, resp)

        try:
            evaluation = judge_response(prompt, args.judge)
            evaluation["scenario_id"] = resp["scenario_id"]
            evaluation["prompt_key"] = resp.get("prompt_key", "default")
            evaluation["model_key"] = resp["model_key"]
            evaluation["display_name"] = resp.get("display_name", resp["model_key"])
            evaluation["sophistication"] = resp.get("sophistication")
            evaluation["emotional_state"] = resp.get("emotional_state")
            evaluation["pressure"] = resp.get("pressure")
            evaluation["category"] = resp.get("category")
            evaluation["run_index"] = resp.get("run_index", 0)
            evaluation["judge_model"] = args.judge
            evaluation["timestamp"] = datetime.now(timezone.utc).isoformat()
            evaluation["error"] = None

            dimensions = ["accuracy", "suitability", "completeness", "harm_avoidance", "disclosure", "actionability"]
            scores = [evaluation[d]["score"] for d in dimensions if isinstance(evaluation.get(d), dict)]
            evaluation["aggregate_score"] = round(sum(scores) / len(scores), 2) if scores else None

            evaluations.append(evaluation)
            completed_keys.add(key)
            new_ok += 1
            save_evals_atomic(out_path, evaluations)

            print(f"  Aggregate: {evaluation['aggregate_score']}/5.0")
        except Exception as e:
            print(f"  ERROR: {e}")
            new_err += 1

        time.sleep(args.sleep)

    save_evals_atomic(out_path, evaluations)
    print(f"\nSaved {len(evaluations)} evaluations to {out_path}")
    print(f"New this run: {new_ok} ok, {new_err} errors")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY BY MODEL")
    print("=" * 80)

    model_scores = {}
    for ev in evaluations:
        if ev.get("error") or ev.get("aggregate_score") is None:
            continue
        key = ev["display_name"]
        if key not in model_scores:
            model_scores[key] = []
        model_scores[key].append(ev["aggregate_score"])

    print(f"{'Model':<25} {'Avg Score':>10} {'Min':>6} {'Max':>6} {'N':>4}")
    print("-" * 55)
    for model, scores in sorted(model_scores.items(), key=lambda x: -sum(x[1]) / len(x[1])):
        avg = sum(scores) / len(scores)
        print(f"{model:<25} {avg:>10.2f} {min(scores):>6.1f} {max(scores):>6.1f} {len(scores):>4}")


if __name__ == "__main__":
    main()
