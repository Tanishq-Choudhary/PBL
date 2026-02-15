import os
import re
import math
import zlib
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import nltk

def safe_word_tokenize(text):
    try:
        return nltk.word_tokenize(text)
    except:
        return re.findall(r'\w+', text)

def safe_sent_tokenize(text):
    try:
        return nltk.sent_tokenize(text)
    except:
        return re.split(r'[.!?]+', text)

def safe_pos_tag(tokens):
    try:
        return nltk.pos_tag(tokens, tagset='universal')
    except:
        return []

def get_syllables(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if not word:
        return 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

FUNCTION_WORDS = set([
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
    'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me'
])

def extract_enhanced_features(filepath):
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except:
        return None

    clean_text = re.sub(r'\s+', ' ', text).strip()
    tokens = safe_word_tokenize(clean_text)
    sentences = [s for s in safe_sent_tokenize(clean_text) if s.strip()]
    
    if len(tokens) < 20:
        return None

    tokens_lower = [t.lower() for t in tokens if t.isalpha()]
    n_tokens = len(tokens)
    n_words = len(tokens_lower)
    n_chars = len(clean_text)
    n_sentences = len(sentences) if len(sentences) > 0 else 1
    
    if n_words == 0:
        return None

    filename = os.path.basename(filepath)
    stem = os.path.splitext(filename)[0]
    parts = stem.split('_')
    source = parts[-1].lower() if len(parts) > 1 else 'unknown'
    is_ai = 0 if source in ['wikipedia', 'human', 'wiki'] else 1
    
    features = {
        'filename': filename,
        'source': source,
        'is_ai': is_ai
    }
    
    types = set(tokens_lower)
    n_types = len(types)
    word_counts = Counter(tokens_lower)
    hapax = sum(1 for c in word_counts.values() if c == 1)
    dis_legomena = sum(1 for c in word_counts.values() if c == 2)
    
    features['lex_ttr'] = n_types / n_words
    features['lex_rttr'] = n_types / math.sqrt(n_words)
    features['lex_log_ttr'] = math.log(n_types) / math.log(n_words) if n_words > 1 else 0
    features['lex_hapax_ratio'] = hapax / n_words
    features['lex_dis_legomena_ratio'] = dis_legomena / n_words
    features['lex_avg_word_len'] = sum(len(w) for w in tokens_lower) / n_words
    features['lex_word_len_std'] = np.std([len(w) for w in tokens_lower])
    
    sent_lens = [len(safe_word_tokenize(s)) for s in sentences]
    features['struct_avg_sent_len'] = np.mean(sent_lens) if sent_lens else 0
    features['struct_sent_len_std'] = np.std(sent_lens) if len(sent_lens) > 1 else 0
    features['struct_burstiness'] = features['struct_sent_len_std']
    features['struct_max_sent_len'] = max(sent_lens) if sent_lens else 0
    features['struct_min_sent_len'] = min(sent_lens) if sent_lens else 0
    features['struct_sent_len_range'] = features['struct_max_sent_len'] - features['struct_min_sent_len']
    
    syllables = sum(get_syllables(w) for w in tokens_lower)
    features['read_flesch'] = 206.835 - 1.015 * (n_tokens / n_sentences) - 84.6 * (syllables / n_tokens) if n_tokens > 0 else 0
    features['read_flesch_kincaid'] = 0.39 * (n_tokens / n_sentences) + 11.8 * (syllables / n_tokens) - 15.59 if n_tokens > 0 else 0
    features['read_avg_syllables_per_word'] = syllables / n_words if n_words > 0 else 0
    features['read_complex_word_ratio'] = sum(1 for w in tokens_lower if get_syllables(w) >= 3) / n_words
    
    char_probs = [clean_text.count(c) / n_chars for c in set(clean_text)]
    features['ent_shannon'] = -sum(p * math.log2(p) for p in char_probs if p > 0)
    features['ent_word_entropy'] = -sum((c/n_words) * math.log2(c/n_words) for c in word_counts.values())
    encoded = clean_text.encode('utf-8')
    features['ent_zlib_ratio'] = len(zlib.compress(encoded)) / n_chars
    features['ent_char_diversity'] = len(set(clean_text)) / n_chars
    
    pos_tags = safe_pos_tag(tokens)
    if pos_tags:
        pos_counts = Counter(tag for w, tag in pos_tags)
        for tag in ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'PRON', 'ADP', 'CONJ']:
            features[f'pos_{tag}'] = pos_counts.get(tag, 0) / n_tokens
    else:
        for tag in ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'PRON', 'ADP', 'CONJ']:
            features[f'pos_{tag}'] = 0
    
    func_word_count = sum(1 for w in tokens_lower if w in FUNCTION_WORDS)
    features['func_word_ratio'] = func_word_count / n_words
    features['func_word_diversity'] = len(set(w for w in tokens_lower if w in FUNCTION_WORDS)) / len(FUNCTION_WORDS)
    features['content_word_ratio'] = 1 - features['func_word_ratio']
    
    punct = '.,!?;:'
    features['punct_comma_ratio'] = text.count(',') / n_chars
    features['punct_period_ratio'] = text.count('.') / n_chars
    features['punct_question_ratio'] = text.count('?') / n_chars
    features['punct_exclaim_ratio'] = text.count('!') / n_chars
    features['punct_total_ratio'] = sum(text.count(p) for p in punct) / n_chars
    
    return features

def analyze_directory(input_dir, output_csv="enhanced_features.csv"):
    
    input_path = Path(input_dir)
    txt_files = list(input_path.rglob("*.txt"))
    
    print(f"Analyzing {len(txt_files)} files...")
    
    data = []
    for filepath in txt_files:
        row = extract_enhanced_features(filepath)
        if row:
            data.append(row)
            if len(data) % 10 == 0:
                print(f"Processed {len(data)} files...")
    
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No valid data extracted!")
        return None
    
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} samples to: {output_csv}")
    return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_features.py <input_dir> [output.csv]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "enhanced_features.csv"
    
    analyze_directory(input_dir, output_csv)
