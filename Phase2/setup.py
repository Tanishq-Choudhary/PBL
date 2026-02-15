#!/usr/bin/env python3

import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        return True
    except:
        return False

def download_nltk_data():
    import nltk
    resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'universal_tagset']
    
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            print(resource + " already downloaded")
        except LookupError:
            try:
                print("Downloading " + resource + "...")
                nltk.download(resource, quiet=True)
                print(resource + " downloaded")
            except:
                print("Failed to download " + resource)

def main():
    
    print("\nSETUP - AI TEXT DETECTION PIPELINE\n")
    
    if sys.version_info < (3, 7):
        print("Python 3.7 or higher required")
        sys.exit(1)
    print("Python " + str(sys.version_info.major) + "." + str(sys.version_info.minor))
    
    print("\nInstalling Dependencies")
    packages = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
        'nltk',
        'tqdm'
    ]
    
    for pkg in packages:
        print("Installing " + pkg + "...", end=' ')
        if install_package(pkg):
            print("Done")
        else:
            print("Skipped or already installed")
    
    print("\nDownloading NLTK Resources")
    import nltk
    download_nltk_data()
    
    print("\nCreating Directory Structure")
    import os
    from pathlib import Path
    
    dirs = [
        'datasets_raw',
        'datasets_parsed', 
        'data_augmented',
        'results',
        'website_output',
        'Data'
    ]
    
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(d + "/")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Put your .txt files in Data/ folder")
    print("2. Run: python run_pipeline.py")
    print("3. Follow the prompts")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup interrupted")
        sys.exit(1)
    except Exception as e:
        print("\nSetup failed: " + str(e))
        sys.exit(1)
