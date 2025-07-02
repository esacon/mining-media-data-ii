#!/usr/bin/env python3
"""
Script to analyze the churn distribution in Game 2 datasets.
"""

import json
import os
from collections import Counter

def analyze_churn_distribution(file_path):
    """Analyze churn distribution in a JSONL file."""
    churned_count = 0
    not_churned_count = 0
    total_count = 0
    
    print(f"Analyzing: {file_path}")
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                total_count += 1
                
                if data['churned']:
                    churned_count += 1
                else:
                    not_churned_count += 1
    
    print(f"Total players: {total_count:,}")
    print(f"Churned players: {churned_count:,} ({churned_count/total_count*100:.2f}%)")
    print(f"Not churned players: {not_churned_count:,} ({not_churned_count/total_count*100:.2f}%)")
    print(f"Churn rate: {churned_count/total_count*100:.2f}%")
    print("-" * 50)
    
    return {
        'total': total_count,
        'churned': churned_count,
        'not_churned': not_churned_count,
        'churn_rate': churned_count/total_count*100
    }

def main():
    """Main function to analyze both Game 2 datasets."""
    base_path = "src/data/processed"
    
    datasets = [
        ("game2_DS1_labeled.jsonl", "Game 2 Training Dataset (DS1)"),
        ("game2_DS2_labeled.jsonl", "Game 2 Evaluation Dataset (DS2)")
    ]
    
    results = {}
    
    print("=" * 60)
    print("GAME 2 CHURN DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    for filename, description in datasets:
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            print(f"\n{description}")
            print("=" * len(description))
            results[filename] = analyze_churn_distribution(file_path)
        else:
            print(f"File not found: {file_path}")
    
    # Combined analysis
    if len(results) == 2:
        print("\nCOMBINED ANALYSIS")
        print("=" * 17)
        total_combined = sum(r['total'] for r in results.values())
        churned_combined = sum(r['churned'] for r in results.values())
        not_churned_combined = sum(r['not_churned'] for r in results.values())
        
        print(f"Combined total players: {total_combined:,}")
        print(f"Combined churned players: {churned_combined:,} ({churned_combined/total_combined*100:.2f}%)")
        print(f"Combined not churned players: {not_churned_combined:,} ({not_churned_combined/total_combined*100:.2f}%)")
        print(f"Overall churn rate: {churned_combined/total_combined*100:.2f}%")

if __name__ == "__main__":
    main() 