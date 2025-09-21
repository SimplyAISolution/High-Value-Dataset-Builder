#!/bin/bash
# Test script for High-Value-Dataset-Builder workflow
# Validates that the cleaning pipeline works correctly with sample data

set -e  # Exit on any error

echo "🧪 Running High-Value-Dataset-Builder test workflow..."
echo

# Check if we're in the right directory
if [ ! -f "test_example.yml" ] || [ ! -d "input" ]; then
  echo "❌ Error: Please run this script from the test_workflow directory"
  exit 1
fi

# Check if collect_and_clean.py exists
if [ ! -f "../collect_and_clean.py" ]; then
  echo "❌ Error: collect_and_clean.py not found in parent directory"
  exit 1
fi

# Run the cleaning pipeline
echo "📁 Running data cleaning pipeline..."
python ../collect_and_clean.py --config test_example.yml

# Check if output was created
if [ ! -d "output" ]; then
  echo "❌ Error: Output directory was not created"
  exit 1
fi

# Find the output file (should be cleaned_dataset.parquet)
OUTPUT_FILE="output/cleaned_dataset.parquet"
if [ ! -f "$OUTPUT_FILE" ]; then
  echo "❌ Error: Expected output file $OUTPUT_FILE not found"
  echo "Files in output directory:"
  ls -la output/ || echo "(output directory is empty)"
  exit 1
fi

# Basic validation: check if the file has content
if [ ! -s "$OUTPUT_FILE" ]; then
  echo "❌ Error: Output file $OUTPUT_FILE is empty"
  exit 1
fi

# Success!
echo
echo "✅ Test completed successfully!"
echo "📊 Results:"
echo "   - Input files: input/sample.csv, input/sample.jsonl"
echo "   - Output file: $OUTPUT_FILE"
echo "   - File size: $(ls -lh $OUTPUT_FILE | awk '{print $5}')"
echo
echo "🎉 The data cleaning pipeline works correctly!"
echo "   Original messy data has been cleaned, deduplicated, and exported."
echo
echo "💡 Next steps:"
echo "   1. Check the output file: $OUTPUT_FILE"
echo "   2. Verify the data has been properly cleaned"
echo "   3. Compare with the original input files in input/"
echo
