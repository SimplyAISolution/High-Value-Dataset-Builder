# High-Value-Dataset-Builder

Easy tools and scripts for collecting, preparing, and cleaning datasets for LLM fine-tuning.

**Overview**
High-Value-Dataset-Builder helps you: (1) collect raw data from various sources, (2) clean and normalize it into consistent JSON/CSV/Parquet, (3) validate quality, and (4) export ready-to-train datasets for both local and cloud AI models.

**Features**
- Modular, composable Python scripts
- Simple CLI workflows
- Optional run usage logging (usage.log)
- Reproducible configs via YAML/JSON
- Basic schema validation hooks

**Quickstart**
```bash
# 1) Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 2) Install dependencies
pip install -r requirements.txt  # if present
# Or minimal deps
pip install pandas pyyaml tqdm requests

# 3) Run the collection + cleaning workflow
python collect_and_clean.py \
  --input ./data/raw \
  --output ./data/processed \
  --config ./config/example.yml \
  --log-usage
```

**Project Structure**
```
.
├─ collect_and_clean.py       # Main workflow runner (collect + clean)
├─ data/
│  ├─ raw/                    # Place raw inputs here
│  └─ processed/              # Cleaned outputs land here
├─ config/
│  └─ example.yml             # Example configuration (sources, cleaning rules)
├─ usage.log                  # Optional: appended run metadata
└─ README.md
```

**Usage: Collecting and Cleaning Data (Step-by-step)**
1) Prepare config
- Define source(s), file patterns, and cleaning rules in config/example.yml
- Example config:
```yaml
sources:
  - path: ./data/raw
    include: ["*.jsonl", "*.json", "*.csv"]
cleaning:
  drop_duplicates: true
  trim_whitespace: true
  normalize_unicode: true
  remove_empty: true
  lower_keys: false
export:
  format: parquet  # options: csv|jsonl|parquet
  output_dir: ./data/processed
```

2) Place raw data
- Put your files into data/raw (or paths declared in config)

3) Run workflow
```bash
python collect_and_clean.py --config ./config/example.yml --input ./data/raw --output ./data/processed --log-usage
```

4) Inspect outputs
- Check data/processed for cleaned dataset (e.g., dataset.parquet)
- Review console summary and usage.log for run details

**Enable Basic Usage Tracking (usage.log)**
- Pass --log-usage to append run metadata (timestamp, args, counts) to usage.log
- Minimal example snippet inside collect_and_clean.py:
```python
import json, os, time

def log_usage(enabled: bool, info: dict, path: str = "usage.log"):
    if not enabled:
        return
    record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Example call after a run:
log_usage(args.log_usage, {
    "input": args.input,
    "output": args.output,
    "n_records_in": n_in,
    "n_records_out": n_out,
    "config": args.config,
})
```
- CLI flag pattern:
```python
import argparse
p = argparse.ArgumentParser()
p.add_argument("--log-usage", action="store_true")
args = p.parse_args()
```

**Workflow Optimization Tips**
- Modularize: keep collectors (fetch/load) and cleaners (transform) in separate functions/classes
- Reproducibility: store configs in version control; pin package versions (requirements.txt); seed randomness
- Validation: add schema checks (pydantic/cerberus) and simple unit tests for transforms
- Incremental runs: write idempotent steps; cache intermediate artifacts
- Monitor quality: compute basic stats (missingness, length, dedupe rate) after cleaning

**Using Generated Datasets with Local and Cloud Models**
1) Local models (e.g., llama.cpp, vLLM, Ollama)
```bash
# Example: fine-tune with local framework (pseudo)
python finetune_local.py \
  --train ./data/processed/dataset.parquet \
  --model ./models/base  \
  --out ./models/finetuned
```

2) Cloud AI APIs (e.g., OpenAI, Anthropic, Mistral)
```python
import json, os
import requests

API_URL = os.getenv("API_URL")  # e.g., https://api.openai.com/v1/chat/completions
API_KEY = os.getenv("API_KEY")

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Convert a processed row to your provider's schema
payload = {
  "model": "gpt-4o-mini",   # example
  "messages": [{"role": "user", "content": "Hello"}],
}
resp = requests.post(API_URL, headers=headers, json=payload)
print(resp.json())
```
- Tips:
  - Use Parquet/Arrow for speed and memory efficiency
  - Batch requests and add retries/backoff
  - Never hardcode API keys; use env vars or secret managers

**FAQ**
- Q: Where do I put raw data?
  A: data/raw or as configured in config.
- Q: How do I turn on logging?
  A: Add --log-usage when running the script.

**License**
MIT (see LICENSE if present)
