import os
import urllib.request
import zipfile
import json
from pathlib import Path

def download_file(url, output_path):
    print(f"Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded to: {output_path}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

def download_datasets(output_dir="datasets"):
    
    Path(output_dir).mkdir(exist_ok=True)
    
    datasets = []
    
    print("\n=== Downloading HC3 Dataset ===")
    hc3_files = [
        ("https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/reddit_eli5.jsonl", "hc3_reddit.jsonl"),
        ("https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/open_qa.jsonl", "hc3_openqa.jsonl"),
    ]
    
    for url, filename in hc3_files:
        output = os.path.join(output_dir, filename)
        if download_file(url, output):
            datasets.append(output)
    
    print(f"\nDownloaded {len(datasets)} dataset files")
    print(f"Location: {output_dir}/")
    return datasets

def parse_hc3_to_txt(jsonl_file, output_dir, max_samples=100):
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nParsing {jsonl_file}...")
    
    human_count = 0
    ai_count = 0
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                    
                try:
                    data = json.loads(line)
                    question = data.get('question', '')
                    
                    if 'human_answers' in data:
                        for j, answer in enumerate(data['human_answers'][:2]):
                            text = f"{question}\n\n{answer}"
                            filename = f"sample_{i}_{j}_human.txt"
                            with open(output_path / filename, 'w', encoding='utf-8') as out:
                                out.write(text)
                            human_count += 1
                    
                    if 'chatgpt_answers' in data:
                        for j, answer in enumerate(data['chatgpt_answers'][:2]):
                            text = f"{question}\n\n{answer}"
                            filename = f"sample_{i}_{j}_chatgpt.txt"
                            with open(output_path / filename, 'w', encoding='utf-8') as out:
                                out.write(text)
                            ai_count += 1
                            
                except Exception as e:
                    continue
        
        print(f"Created {human_count} human samples")
        print(f"Created {ai_count} AI samples")
        print(f"Location: {output_path}/")
        
    except Exception as e:
        print(f"Error parsing: {e}")

if __name__ == "__main__":
    datasets = download_datasets("datasets_raw")
    
    for dataset_file in datasets:
        basename = os.path.splitext(os.path.basename(dataset_file))[0]
        output_dir = f"datasets_parsed/{basename}"
        parse_hc3_to_txt(dataset_file, output_dir, max_samples=50)
    
    print("\n" + "="*50)
    print("ALL DONE")
    print("Your data is ready in: datasets_parsed/")
    print("="*50)
