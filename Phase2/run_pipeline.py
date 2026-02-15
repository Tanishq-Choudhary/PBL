#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path
import shutil

def run_command(cmd, description):
    print("\n" + "="*60)
    print("STEP: " + description)
    print("="*60)
    print("Command: " + cmd + "\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(description + " - SUCCESS")
        return True
    else:
        print(description + " - FAILED")
        return False

def main():
    
    print("\nAI TEXT DETECTION - AUTOMATED PIPELINE")
    print("Generate Results Fast and Professionally\n")
    
    print("Setting up directories...")
    dirs = ['datasets_raw', 'datasets_parsed', 'data_augmented', 'results', 'website_output']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    print("Directories created\n")
    
    print("\n" + "="*60)
    print("OPTION 1: Download Free Datasets (HC3 - Human vs ChatGPT)")
    print("OPTION 2: Use your existing data folder")
    print("="*60)
    
    choice = input("\nDownload datasets? (y/n) [default: n]: ").strip().lower()
    
    if choice == 'y':
        run_command(
            "python download_datasets.py",
            "Downloading HC3 Dataset"
        )
        data_source = "datasets_parsed"
    else:
        data_source = input("Enter your data directory path: ").strip()
        if not data_source or not os.path.exists(data_source):
            data_source = "Data"
        print("Using existing data: " + data_source)
    
    print("\n" + "="*60)
    create_para = input("Create paraphrased versions? (y/n) [default: y]: ").strip().lower()
    
    if create_para != 'n':
        run_command(
            "python paraphraser.py " + data_source + " data_augmented",
            "Creating Paraphrased Versions"
        )
        analysis_dir = "data_augmented"
    else:
        analysis_dir = data_source
    
    run_command(
        "python enhanced_features.py " + analysis_dir + " enhanced_features.csv",
        "Extracting Enhanced Features (30+ metrics)"
    )
    
    run_command(
        "python analyze_results.py enhanced_features.csv results",
        "Statistical Analysis and Visualization Generation"
    )
    
    print("\n" + "="*60)
    print("Generating website files...")
    
    result_files = [
        'results/top_features_boxplot.png',
        'results/effect_sizes.png',
        'results/statistical_tests.csv',
        'results/summary_report.json'
    ]
    
    for file in result_files:
        if os.path.exists(file):
            shutil.copy(file, 'website_output/')
            print("Copied " + file)
    
    if os.path.exists('enhanced_features.csv'):
        shutil.copy('enhanced_features.csv', 'website_output/')
        print("Copied enhanced_features.csv")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\nYour results are ready:")
    print("Results: results/")
    print("Visualizations: results/*.png")
    print("Statistics: results/statistical_tests.csv")
    print("Website files: website_output/")
    print("\nNext steps:")
    print("1. Check results/ folder for all outputs")
    print("2. Copy website_output/ files to your GitHub Pages")
    print("3. Update your PBL presentation HTML")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print("\n\nPipeline failed: " + str(e))
        sys.exit(1)
