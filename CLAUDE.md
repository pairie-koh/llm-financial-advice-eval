# LLM Financial Advice Evaluation

## Project Overview

Public benchmark evaluating the quality of personal financial advice from major LLMs (ChatGPT, Claude, Gemini, Llama, etc.). Motivated by the rapid growth of unregulated AI financial advisors and policy interest in understanding their impact on retail investors.

## Research Questions

1. **Quality**: How accurate and appropriate is the financial advice from different LLMs across common personal finance scenarios?
2. **Suitability**: Do LLMs tailor advice to individual circumstances (age, income, risk tolerance, goals), or give generic one-size-fits-all recommendations?
3. **Alignment**: Are LLMs aligned with the user's financial interests, or do they exhibit biases (e.g., favoring specific products, being overly conservative/aggressive)?
4. **Regulatory Compliance**: Would the advice violate SEC/FINRA suitability rules if given by a licensed advisor?
5. **Consistency**: How much does advice vary across repeated prompts or slight rephrasing?

## Collaborators

- Andy Hall (Free Systems Lab)
- Pairie Koh

## Methodology

### Scenario-Based Evaluation

We create realistic personal finance scenarios (personas with specific demographics, income, goals, constraints) and prompt each LLM for advice. Scenarios span:

- **Retirement planning** (401k allocation, savings rate, Social Security timing)
- **Debt management** (student loans, credit cards, mortgage decisions)
- **Investment allocation** (portfolio construction, risk tolerance mapping)
- **Tax optimization** (Roth conversions, tax-loss harvesting, HSA strategy)
- **Major life events** (home buying, job loss, inheritance, divorce)
- **Edge cases** (crypto speculation, concentrated stock positions, penny stocks)

### Evaluation Dimensions

1. **Accuracy** - Are factual claims correct (tax brackets, contribution limits, etc.)?
2. **Suitability** - Is advice appropriate for the persona's specific situation?
3. **Completeness** - Does it address key considerations, or miss important factors?
4. **Harm avoidance** - Does it steer away from clearly bad advice?
5. **Disclosure** - Does it acknowledge limitations and recommend professional help when appropriate?
6. **Consistency** - Same scenario, same advice across runs?

### Models Under Evaluation

- OpenAI: GPT-4o, GPT-4o-mini, o1, o3
- Anthropic: Claude Sonnet 4, Claude Opus 4
- Google: Gemini 2.0 Flash, Gemini 2.5 Pro
- Meta: Llama 3.3 70B, Llama 4 Scout
- Open-source: Mixtral, Qwen, DeepSeek

## Repo Structure

```
data/
  scenarios/       # JSON files defining evaluation scenarios (personas + questions)
  responses/       # Raw LLM responses (gitignored, large)
  evaluations/     # Scored evaluations
scripts/
  generate_scenarios.py   # Create evaluation scenarios
  query_llms.py           # Send scenarios to each LLM API
  evaluate_responses.py   # Score responses against rubrics
  analyze_results.py      # Aggregate analysis and visualizations
notebooks/
  exploration.ipynb       # Ad-hoc analysis
```

## Running

```bash
pip install -r requirements.txt

# Generate scenarios
python scripts/generate_scenarios.py

# Query all LLMs (requires API keys in .env)
python scripts/query_llms.py --models all --scenarios data/scenarios/

# Evaluate responses
python scripts/evaluate_responses.py

# Analyze
python scripts/analyze_results.py
```

## API Keys

Store in `.env` (gitignored):
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...  # for open-source models
```
