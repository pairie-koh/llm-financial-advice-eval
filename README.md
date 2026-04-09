# LLM Financial Advice Evaluation

Public benchmark evaluating the quality of personal financial advice from major LLMs.

## Motivation

As AI-powered financial advisors proliferate — many operating outside traditional regulatory frameworks — there's an urgent need to understand how good their advice actually is. This project systematically evaluates financial advice from leading LLMs across realistic personal finance scenarios.

## How It Works

1. **Scenarios**: We define realistic personas with specific financial situations and questions (see `data/scenarios/`)
2. **Query**: Each scenario is sent to multiple LLMs under identical conditions
3. **Evaluate**: An LLM judge scores each response on accuracy, suitability, completeness, harm avoidance, disclosure, and actionability
4. **Analyze**: We aggregate scores to produce model rankings and identify failure modes

## Quick Start

```bash
pip install -r requirements.txt

# Set API keys in .env
cp .env.example .env  # then fill in your keys

# Query models (start with a subset)
python scripts/query_llms.py --scenarios data/scenarios/pilot_scenarios.json --models openai

# Evaluate responses
python scripts/evaluate_responses.py --responses data/responses/responses_*.json --scenarios data/scenarios/pilot_scenarios.json

# Analyze results
python scripts/analyze_results.py --evaluations data/evaluations/evaluations_*.json
```

## Evaluation Dimensions

| Dimension | What it measures |
|-----------|-----------------|
| Accuracy | Are factual claims correct? (tax rules, limits, etc.) |
| Suitability | Is advice appropriate for this specific person? |
| Completeness | Are key considerations covered? |
| Harm Avoidance | Does it warn against clearly bad ideas? |
| Disclosure | Does it acknowledge limitations? |
| Actionability | Can the person act on this advice? |

## Contributing

We welcome new scenarios, evaluation dimensions, and analysis. See `CLAUDE.md` for project structure and conventions.
