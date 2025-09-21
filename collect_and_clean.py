#!/usr/bin/env python3
"""
High-Value Dataset Builder - Data Collection and Cleaning Script

This script provides a customizable framework for collecting, processing,
and cleaning datasets for LLM fine-tuning.

Customize the functions below to suit your specific data collection needs.
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any

# Configuration
DATA_DIR = Path('./data')
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

def setup_directories():
    """Create necessary data directories if they don't exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories created: {RAW_DATA_DIR}, {PROCESSED_DATA_DIR}")

def collect_data() -> List[Dict[str, Any]]:
    """
    Collect raw data from your sources.
    
    TODO: Customize this function for your data collection needs:
    - Web scraping
    - API calls  
    - File parsing
    - Database queries
    
    Returns:
        List of dictionaries containing raw data
    """
    print("📥 Starting data collection...")
    
    # Example placeholder data - replace with your collection logic
    sample_data = [
        {
            "id": 1,
            "text": "This is sample training text for LLM fine-tuning.",
            "label": "example",
            "source": "placeholder"
        },
        {
            "id": 2, 
            "text": "Add your own data collection logic here.",
            "label": "instruction",
            "source": "placeholder"
        }
    ]
    
    print(f"✓ Collected {len(sample_data)} raw data points")
    return sample_data

def clean_data(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean and preprocess the collected data.
    
    Args:
        raw_data: List of raw data dictionaries
        
    Returns:
        List of cleaned data dictionaries
    """
    print("🧹 Starting data cleaning...")
    
    cleaned_data = []
    
    for item in raw_data:
        # TODO: Add your cleaning logic here:
        # - Text normalization
        # - Deduplication
        # - Quality filtering
        # - Format standardization
        
        # Example cleaning - customize as needed
        cleaned_item = {
            "id": item["id"],
            "text": item["text"].strip().lower(),  # Basic text cleaning
            "label": item["label"],
            "source": item["source"],
            "cleaned": True
        }
        
        # Filter out empty or invalid entries
        if cleaned_item["text"] and len(cleaned_item["text"]) > 10:
            cleaned_data.append(cleaned_item)
    
    print(f"✓ Cleaned data: {len(cleaned_data)} valid entries")
    return cleaned_data

def save_data(data: List[Dict[str, Any]], filename: str, data_dir: Path):
    """
    Save data to JSON and CSV formats.
    
    Args:
        data: List of data dictionaries
        filename: Base filename (without extension)
        data_dir: Directory to save the files
    """
    # Save as JSON
    json_path = data_dir / f"{filename}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Save as CSV
    csv_path = data_dir / f"{filename}.csv"
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    print(f"💾 Data saved to {json_path} and {csv_path}")

def main():
    """
    Main execution function.
    Run the complete data collection and cleaning pipeline.
    """
    print("🚀 High-Value Dataset Builder Starting...")
    
    # Setup
    setup_directories()
    
    # Collect raw data
    raw_data = collect_data()
    save_data(raw_data, "raw_dataset", RAW_DATA_DIR)
    
    # Clean the data
    cleaned_data = clean_data(raw_data)
    save_data(cleaned_data, "cleaned_dataset", PROCESSED_DATA_DIR)
    
    print(f"✅ Pipeline complete! Generated {len(cleaned_data)} clean training samples.")
    print(f"📁 Check {PROCESSED_DATA_DIR} for your LLM training data.")

if __name__ == "__main__":
    main()
