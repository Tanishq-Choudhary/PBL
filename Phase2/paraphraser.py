import random
import re
import os
from pathlib import Path
import nltk

SYNONYMS = {
    'good': ['great', 'excellent', 'fine', 'nice', 'positive'],
    'bad': ['poor', 'negative', 'unfavorable', 'inferior'],
    'big': ['large', 'huge', 'enormous', 'massive', 'substantial'],
    'small': ['tiny', 'little', 'minute', 'compact'],
    'important': ['significant', 'crucial', 'vital', 'essential', 'critical'],
    'show': ['demonstrate', 'indicate', 'reveal', 'display', 'exhibit'],
    'use': ['utilize', 'employ', 'apply', 'implement'],
    'make': ['create', 'produce', 'generate', 'form', 'construct'],
    'think': ['believe', 'consider', 'regard', 'suppose'],
    'know': ['understand', 'recognize', 'realize', 'comprehend'],
    'see': ['observe', 'notice', 'view', 'perceive'],
    'get': ['obtain', 'acquire', 'receive', 'gain'],
    'very': ['extremely', 'highly', 'remarkably', 'particularly'],
    'also': ['additionally', 'furthermore', 'moreover', 'likewise'],
    'however': ['nevertheless', 'nonetheless', 'yet', 'still'],
    'because': ['since', 'as', 'due to', 'owing to'],
    'many': ['numerous', 'various', 'multiple', 'several'],
    'different': ['various', 'diverse', 'distinct', 'separate'],
    'new': ['novel', 'recent', 'modern', 'fresh'],
    'first': ['initial', 'primary', 'foremost'],
}

def synonym_paraphrase(text, replacement_rate=0.3):
    words = text.split()
    paraphrased = []
    
    for word in words:
        word_lower = word.lower().strip('.,!?;:')
        
        if word_lower in SYNONYMS and random.random() < replacement_rate:
            synonym = random.choice(SYNONYMS[word_lower])
            if word[0].isupper():
                synonym = synonym.capitalize()
            punct = ''.join([c for c in word if c in '.,!?;:'])
            paraphrased.append(synonym + punct)
        else:
            paraphrased.append(word)
    
    return ' '.join(paraphrased)

def sentence_shuffle_paraphrase(text):
    sentences = re.split(r'([.!?]+)', text)
    pairs = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            pairs.append(sentences[i] + sentences[i+1])
    
    if len(pairs) > 2:
        middle = pairs[1:-1]
        random.shuffle(middle)
        pairs = [pairs[0]] + middle + [pairs[-1]]
    
    return ' '.join(pairs)

def light_paraphrase(text):
    return synonym_paraphrase(text, replacement_rate=0.3)

def heavy_paraphrase(text):
    text = synonym_paraphrase(text, replacement_rate=0.5)
    return sentence_shuffle_paraphrase(text)

def augment_with_paraphrasing(input_dir, output_dir):
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    (output_path / "original").mkdir(parents=True, exist_ok=True)
    (output_path / "paraphrased_light").mkdir(parents=True, exist_ok=True)
    (output_path / "paraphrased_heavy").mkdir(parents=True, exist_ok=True)
    
    txt_files = list(input_path.rglob("*.txt"))
    print(f"Found {len(txt_files)} files to paraphrase")
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            
            if len(text) < 50:
                continue
            
            filename = txt_file.name
            
            with open(output_path / "original" / filename, 'w', encoding='utf-8') as f:
                f.write(text)
            
            light = light_paraphrase(text)
            with open(output_path / "paraphrased_light" / filename, 'w', encoding='utf-8') as f:
                f.write(light)
            
            heavy = heavy_paraphrase(text)
            with open(output_path / "paraphrased_heavy" / filename, 'w', encoding='utf-8') as f:
                f.write(heavy)
            
            print(f"{filename}")
            
        except Exception as e:
            print(f"{txt_file.name}: {e}")
    
    print("\nParaphrasing complete")
    print(f"Output: {output_path}/")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python paraphraser.py <input_dir> <output_dir>")
        sys.exit(1)
    
    augment_with_paraphrasing(sys.argv[1], sys.argv[2])
