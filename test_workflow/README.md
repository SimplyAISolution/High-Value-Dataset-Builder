# Test Workflow

This directory contains test files and examples to demonstrate the High-Value-Dataset-Builder tool.

## Files Included

- `test_example.yml` - Configuration file demonstrating cleaning settings
- `input/` - Directory containing sample input files:
  - `sample.csv` - CSV with whitespace, mixed case, and duplicates
  - `sample.jsonl` - JSONL with whitespace, mixed case, and duplicates
- `output/` - Directory where cleaned outputs will be generated
- `test_run.sh` - Automated test script to validate the pipeline

## How to Use

1. **Install dependencies** (from project root):
   ```bash
   pip install pandas pyyaml
   ```

2. **Run the cleaning pipeline**:
   ```bash
   cd test_workflow
   python ../collect_and_clean.py --config test_example.yml
   ```

3. **Check the output**:
   - Look in the `output/` directory for the processed dataset
   - The output should be deduplicated, have trimmed whitespace, and normalized Unicode
   - Original messy data: 4 records → Clean data: 2 records (after deduplication)

4. **Run automated test** (optional):
   ```bash
   chmod +x test_run.sh
   ./test_run.sh
   ```

## Expected Results

The sample files contain:
- Mixed case field names and values
- Leading/trailing whitespace
- Duplicate records
- Empty/whitespace-only values

After processing, you should see:
- ✅ Duplicates removed
- ✅ Whitespace trimmed
- ✅ Unicode normalized
- ✅ Empty records filtered out
- ✅ Consistent data format

The test validates that the output contains exactly 2 unique, clean records.
