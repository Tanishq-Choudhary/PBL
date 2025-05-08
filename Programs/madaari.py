import os
import sys
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from nltk.tokenize import sent_tokenize, RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams, pos_tag, FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import math
import textstat
import matplotlib
matplotlib.use('Agg')
import csv
import string
import re

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")
    print("If you already have the data installed, this should be fine.")

DEFAULT_FUNCTION_WORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
    'can', 'could', 
    'did', 'do', 'does', 'doing', 'down', 'during', 
    'each', 
    'few', 'for', 'from', 'further', 
    'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 
    'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 
    'just', 
    'me', 'more', 'most', 'my', 'myself', 
    'no', 'nor', 'not', 'now', 
    'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 
    'same', 'she', 'should', 'so', 'some', 'such', 
    'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 
    'under', 'until', 'up', 
    'very', 
    'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 
    'you', 'your', 'yours', 'yourself', 'yourselves'
])

def validate_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return False
    
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    if not txt_files:
        print(f"Error: No text files found in '{directory_path}'.")
        return False
    
    return True

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None

def preprocess_text(text):
    if not text:
        return {
            'original_text': "", 'sentences': [], 'words_in_sentences': [],
            'tokens_with_stopwords': [], 'tokens_cleaned': [], 'characters': []
        }
        
    word_tokenizer = RegexpTokenizer(r'\w+')
    
    try:
        sentences_raw = sent_tokenize(text)
    except Exception as e:
        print(f"Warning: NLTK sent_tokenize failed ({e}), using basic split.")
        sentences_raw = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    sentences = [s for s in sentences_raw if s]
    words_in_sentences = [word_tokenizer.tokenize(s.lower()) for s in sentences]
    
    words = word_tokenizer.tokenize(text.lower())
    characters = list(text)
    
    tokens_with_stopwords = words
    
    try:
        stop_words = set(stopwords.words('english'))
        tokens_cleaned = [word for word in words if word not in stop_words]
    except LookupError:
        print("Warning: NLTK stopwords not found. Proceeding without stopword removal.")
        tokens_cleaned = words
        stop_words = set()
    
    return {
        'original_text': text,
        'sentences': sentences,
        'words_in_sentences': words_in_sentences,
        'tokens_with_stopwords': tokens_with_stopwords,
        'tokens_cleaned': tokens_cleaned,
        'characters': characters,
        'stopwords_set': stop_words
    }

def compute_basic_stats(preprocessed_data):
    original_text = preprocessed_data['original_text']
    sentences = preprocessed_data['sentences']
    tokens_with_stopwords = preprocessed_data['tokens_with_stopwords']
    characters = preprocessed_data['characters']
    
    word_count = len(tokens_with_stopwords)
    sentence_count = len(sentences)
    char_count = len(characters)
    unique_words = len(set(tokens_with_stopwords))
    
    avg_word_length = sum(len(word) for word in tokens_with_stopwords) / word_count if word_count > 0 else 0
    avg_sentence_length_words = word_count / sentence_count if sentence_count > 0 else 0
    avg_sentence_length_chars = char_count / sentence_count if sentence_count > 0 else 0
    
    lexical_diversity = unique_words / word_count if word_count > 0 else 0
    
    sentence_lengths = [len(words) for words in preprocessed_data['words_in_sentences']]
    sentence_length_std_dev = np.std(sentence_lengths) if sentence_lengths else 0
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'character_count': char_count,
        'unique_word_count': unique_words,
        'avg_word_length': avg_word_length,
        'avg_sentence_length_words': avg_sentence_length_words,
        'avg_sentence_length_chars': avg_sentence_length_chars,
        'lexical_diversity': lexical_diversity,
        'sentence_length_std_dev': sentence_length_std_dev
    }

def calculate_entropy(tokens):
    if not tokens: return 0
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    probabilities = [count / total_tokens for count in token_counts.values()]
    return -np.sum(probabilities * np.log2(probabilities))

def analyze_readability(text):
    if not text:
        return {'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0, 'gunning_fog': 0, 'smog_index': 0, 'coleman_liau_index': 0, 'automated_readability_index': 0}
    try:
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'smog_index': textstat.smog_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'automated_readability_index': textstat.automated_readability_index(text)
        }
    except Exception as e:
        print(f"Warning: Error calculating readability scores: {e}")
        return {'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0, 'gunning_fog': 0, 'smog_index': 0, 'coleman_liau_index': 0, 'automated_readability_index': 0}

def analyze_frequency_distribution(tokens, top_n=30):
    if not tokens:
        return {'counter': Counter(), 'most_common': [], 'frequencies': [], 'ranks': []}
    
    counter = Counter(tokens)
    most_common = counter.most_common(top_n)
    frequencies = sorted(counter.values(), reverse=True)
    ranks = list(range(1, len(frequencies) + 1))
    
    return {'counter': counter, 'most_common': most_common, 'frequencies': frequencies, 'ranks': ranks}

def perform_pos_analysis(tokens):
    if not tokens:
        return {'pos_tags': [], 'pos_counts': Counter(), 'pos_distribution': {}}
        
    try:
        pos_tags = pos_tag(tokens)
        pos_counts = Counter(tag for word, tag in pos_tags)
        total_tags = sum(pos_counts.values())
        pos_distribution = {tag: count / total_tags for tag, count in pos_counts.items()} if total_tags > 0 else {}
    except LookupError:
        print("Warning: NLTK POS tagger model not found. Skipping POS analysis.")
        pos_tags = []
        pos_counts = Counter()
        pos_distribution = {}
    except Exception as e:
        print(f"Warning: Error in POS tagging: {e}")
        pos_tags = []
        pos_counts = Counter()
        pos_distribution = {}

    return {'pos_tags': pos_tags, 'pos_counts': pos_counts, 'pos_distribution': pos_distribution}

def perform_ngram_analysis(tokens, n_values=[2, 3, 4], top_n=20):
    results = {}
    for n in n_values:
        if len(tokens) < n:
            results[f'{n}-grams'] = {'n_grams': [], 'n_gram_freq': [], 'repetition_rate': 0}
            continue
        
        n_grams_list = list(ngrams(tokens, n))
        n_gram_freq = Counter(n_grams_list).most_common(top_n)
        
        repetition_rate = 0
        if len(n_grams_list) > 0:
            unique_ngrams = len(set(n_grams_list))
            repetition_rate = 1.0 - (unique_ngrams / len(n_grams_list))
            
        results[f'{n}-grams'] = {
            'n_grams': n_grams_list, 
            'n_gram_freq': n_gram_freq,
            'repetition_rate': repetition_rate
        }
    return results

def perform_sentiment_analysis(text):
    if not text:
        return {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0}
    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)
    except LookupError:
        print("Warning: VADER lexicon not found. Skipping sentiment analysis.")
        sentiment = {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0}
    except Exception as e:
        print(f"Warning: Error in sentiment analysis: {e}")
        sentiment = {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0}
    return sentiment

def analyze_punctuation(characters, top_n=10):
    if not characters:
        return {'punctuation_counts': Counter(), 'punctuation_distribution': {}, 'most_common_punctuation': []}
        
    punct_counts = Counter(c for c in characters if c in string.punctuation)
    total_punct = sum(punct_counts.values())
    punct_distribution = {p: count / total_punct for p, count in punct_counts.items()} if total_punct > 0 else {}
    most_common = punct_counts.most_common(top_n)
    
    return {
        'punctuation_counts': punct_counts, 
        'punctuation_distribution': punct_distribution,
        'most_common_punctuation': most_common
    }

def analyze_function_words(tokens, function_word_set=DEFAULT_FUNCTION_WORDS):
    if not tokens: return {'function_word_count': 0, 'function_word_ratio': 0}
    
    func_word_count = sum(1 for token in tokens if token in function_word_set)
    total_tokens = len(tokens)
    ratio = func_word_count / total_tokens if total_tokens > 0 else 0
    
    return {'function_word_count': func_word_count, 'function_word_ratio': ratio}

def calculate_mattr(tokens, window_size=50):
    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
        
    ttrs = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]
        ttr = len(set(window)) / window_size
        ttrs.append(ttr)
        
    return np.mean(ttrs) if ttrs else 0

def perform_character_ngram_analysis(text, n_values=[3, 4, 5], top_n=20):
    results = {}
    text_processed = ''.join(filter(lambda x: x.isalnum() or x.isspace(), text.lower())).replace(" ", "_")
    
    for n in n_values:
        if len(text_processed) < n:
            results[f'char_{n}-grams'] = {'n_grams': [], 'n_gram_freq': []}
            continue
            
        char_ngrams_list = list(ngrams(text_processed, n))
        char_ngram_freq = Counter(char_ngrams_list).most_common(top_n)
        
        results[f'char_{n}-grams'] = {
            'n_grams': [''.join(ng) for ng in char_ngrams_list],
            'n_gram_freq': [(''.join(ng), count) for ng, count in char_ngram_freq]
        }
    return results

def create_visualizations(analysis_results, output_dir, filename_base):
    plots_dir = os.path.join(output_dir, 'plots', filename_base)
    os.makedirs(plots_dir, exist_ok=True)
    plot_paths = {}

    plt.style.use('seaborn-v0_8-darkgrid')
    
    tokens_ws = analysis_results['preprocessed']['tokens_with_stopwords']
    sentences_ws = analysis_results['preprocessed']['words_in_sentences']

    plt.figure(figsize=(10, 6))
    word_lengths = [len(word) for word in tokens_ws]
    if word_lengths:
        sns.histplot(word_lengths, kde=True, bins=max(1, min(max(word_lengths), 30)))
        plt.title('Word Length Distribution')
        plt.xlabel('Word Length')
        plt.ylabel('Frequency')
    else:
        plt.text(0.5, 0.5, "No data available", ha='center', va='center')
    path = os.path.join(plots_dir, f"{filename_base}_word_length.png")
    plt.savefig(path)
    plt.close()
    plot_paths['word_length_plot'] = path
    
    plt.figure(figsize=(10, 6))
    sentence_lengths = [len(words) for words in sentences_ws]
    if sentence_lengths:
        sns.histplot(sentence_lengths, kde=True, bins=max(1, min(max(sentence_lengths)//2, 50)))
        plt.title('Sentence Length Distribution (Words per Sentence)')
        plt.xlabel('Sentence Length (Words)')
        plt.ylabel('Frequency')
    else:
        plt.text(0.5, 0.5, "No data available", ha='center', va='center')
    path = os.path.join(plots_dir, f"{filename_base}_sentence_length.png")
    plt.savefig(path)
    plt.close()
    plot_paths['sentence_length_plot'] = path

    plt.figure(figsize=(10, 6))
    ranks = analysis_results['frequency_distribution']['ranks']
    frequencies = analysis_results['frequency_distribution']['frequencies']
    if ranks and frequencies:
        plot_ranks = ranks[:min(500, len(ranks))]
        plot_freqs = frequencies[:min(500, len(frequencies))]
        plt.loglog(plot_ranks, plot_freqs, marker='.', linestyle='None', markersize=4)
        plt.title("Zipf's Law (Word Frequency vs. Rank)")
        plt.xlabel('Rank (Log Scale)')
        plt.ylabel('Frequency (Log Scale)')
        plt.grid(True, which="both", ls="-", alpha=0.5)
    else:
        plt.text(0.5, 0.5, "No data available", ha='center', va='center')
    path = os.path.join(plots_dir, f"{filename_base}_zipf.png")
    plt.savefig(path)
    plt.close()
    plot_paths['zipf_plot'] = path
    
    plt.figure(figsize=(12, 7))
    pos_counts = analysis_results['pos_analysis']['pos_counts']
    if pos_counts:
        pos_labels = list(pos_counts.keys())
        pos_values = list(pos_counts.values())
        sorted_indices = np.argsort(pos_values)[::-1]
        pos_labels = [pos_labels[i] for i in sorted_indices][:25] 
        pos_values = [pos_values[i] for i in sorted_indices][:25]
        
        plt.bar(pos_labels, pos_values)
        plt.title('Part-of-Speech Distribution (Top 25)')
        plt.xlabel('POS Tag')
        plt.ylabel('Count')
        plt.xticks(rotation=60, ha='right')
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "No POS data available", ha='center', va='center')
    path = os.path.join(plots_dir, f"{filename_base}_pos.png")
    plt.savefig(path)
    plt.close()
    plot_paths['pos_plot'] = path

    plt.figure(figsize=(8, 8))
    sentiment = analysis_results['sentiment']
    labels = ['Positive', 'Neutral', 'Negative']
    values = [sentiment['pos'], sentiment['neu'], sentiment['neg']]
    if sum(values) > 0:
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99','#ff9999'])
        plt.title('Sentiment Distribution (VADER)')
    else:
        plt.text(0.5, 0.5, "No sentiment data", ha='center', va='center')
    path = os.path.join(plots_dir, f"{filename_base}_sentiment.png")
    plt.savefig(path)
    plt.close()
    plot_paths['sentiment_plot'] = path

    plt.figure(figsize=(10, 6))
    punct_counts = analysis_results['punctuation']['punctuation_counts']
    if punct_counts:
        punct_labels = list(punct_counts.keys())
        punct_values = list(punct_counts.values())
        sorted_indices = np.argsort(punct_values)[::-1]
        punct_labels = [punct_labels[i] for i in sorted_indices]
        punct_values = [punct_values[i] for i in sorted_indices]

        plt.bar(punct_labels, punct_values)
        plt.title('Punctuation Frequency')
        plt.xlabel('Punctuation Mark')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "No punctuation data available", ha='center', va='center')
    path = os.path.join(plots_dir, f"{filename_base}_punctuation.png")
    plt.savefig(path)
    plt.close()
    plot_paths['punctuation_plot'] = path

    return plot_paths

def generate_pdf_report(analysis_results, plots, output_file):
    doc = SimpleDocTemplate(output_file, pagesize=letter, topMargin=36, bottomMargin=36, leftMargin=54, rightMargin=54)
    styles = getSampleStyleSheet()
    elements = []
    
    title_style = styles['Title']
    h1_style = styles['Heading1']
    h2_style = styles['Heading2']
    normal_style = styles['Normal']
    code_style = styles['Code']
    
    elements.append(Paragraph(f"Text Analysis Report: {os.path.basename(output_file).replace('_analysis.pdf', '')}", title_style))
    elements.append(Spacer(1, 18))

    elements.append(Paragraph("1. Basic Statistics", h1_style))
    stats = analysis_results['basic_stats']
    stats_data = [
        ['Metric', 'Value'],
        ['Word Count', f"{stats['word_count']}"],
        ['Sentence Count', f"{stats['sentence_count']}"],
        ['Character Count', f"{stats['character_count']}"],
        ['Unique Word Count', f"{stats['unique_word_count']}"],
        ['Avg Word Length (chars)', f"{stats['avg_word_length']:.2f}"],
        ['Avg Sentence Length (words)', f"{stats['avg_sentence_length_words']:.2f}"],
        ['Avg Sentence Length (chars)', f"{stats['avg_sentence_length_chars']:.2f}"],
        ['Sentence Length Std Dev (words)', f"{stats['sentence_length_std_dev']:.2f}"],
        ['Lexical Diversity (TTR)', f"{stats['lexical_diversity']:.4f}"],
        ['MATTR (Window=50)', f"{analysis_results['mattr']:.4f}"]
    ]
    table = Table(stats_data, colWidths=[200, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("2. Entropy & Complexity", h1_style))
    complexity_data = [
        ['Metric', 'Value'],
        ['Shannon Entropy (Words)', f"{analysis_results['word_entropy']:.4f} bits"],
        ['Shannon Entropy (Chars)', f"{analysis_results['char_entropy']:.4f} bits"],
        ['Function Word Ratio', f"{analysis_results['function_words']['function_word_ratio']:.4f}"]
    ]
    table = Table(complexity_data, colWidths=[200, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("3. Readability Scores", h1_style))
    readability = analysis_results['readability']
    readability_data = [
        ['Index', 'Score'],
        ['Flesch Reading Ease', f"{readability['flesch_reading_ease']:.2f}"],
        ['Flesch-Kincaid Grade Level', f"{readability['flesch_kincaid_grade']:.2f}"],
        ['Gunning Fog Index', f"{readability['gunning_fog']:.2f}"],
        ['SMOG Index', f"{readability['smog_index']:.2f}"],
        ['Coleman-Liau Index', f"{readability['coleman_liau_index']:.2f}"],
        ['Automated Readability Index', f"{readability['automated_readability_index']:.2f}"]
    ]
    table = Table(readability_data, colWidths=[200, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("4. Visualizations", h1_style))
    elements.append(Spacer(1, 6))

    def add_plot(title, plot_key):
        if plot_key in plots and os.path.exists(plots[plot_key]):
            elements.append(Paragraph(title, h2_style))
            elements.append(Spacer(1, 6))
            try:
                img = Image(plots[plot_key], width=450, height=270)
                img.hAlign = 'CENTER'
                elements.append(img)
            except Exception as e:
                elements.append(Paragraph(f"Error loading image {plots[plot_key]}: {e}", normal_style))
            elements.append(Spacer(1, 12))
        else:
             elements.append(Paragraph(f"{title}: Plot not available.", normal_style))
             elements.append(Spacer(1, 12))

    add_plot("Word Length Distribution", 'word_length_plot')
    add_plot("Sentence Length Distribution", 'sentence_length_plot')
    add_plot("Zipf's Law Analysis (Word Frequency vs Rank)", 'zipf_plot')
    add_plot("Part-of-Speech Distribution (Top 25)", 'pos_plot')
    add_plot("Sentiment Distribution (VADER)", 'sentiment_plot')
    add_plot("Punctuation Frequency", 'punctuation_plot')
    
    elements.append(Paragraph("5. Frequency Analysis", h1_style))
    elements.append(Paragraph("Most Common Words (Top 15, with stopwords):", h2_style))
    most_common_words = analysis_results['frequency_distribution']['most_common']
    if most_common_words:
        words_text = ", ".join([f"{word} ({count})" for word, count in most_common_words[:15]])
        elements.append(Paragraph(words_text, normal_style))
    else:
        elements.append(Paragraph("No data available", normal_style))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Most Common Punctuation:", h2_style))
    most_common_punct = analysis_results['punctuation']['most_common_punctuation']
    if most_common_punct:
        punct_text = ", ".join([f"'{p}' ({count})" for p, count in most_common_punct])
        elements.append(Paragraph(punct_text, normal_style))
    else:
        elements.append(Paragraph("No data available", normal_style))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("6. N-gram Analysis", h1_style))
    for n in [2, 3, 4]:
        key = f'{n}-grams'
        if key in analysis_results['ngram_analysis']:
            ngram_data = analysis_results['ngram_analysis'][key]
            elements.append(Paragraph(f"Common {n}-grams (Top 10):", h2_style))
            if ngram_data['n_gram_freq']:
                ngram_text = ", ".join([f"'{' '.join(ng)}' ({count})" for ng, count in ngram_data['n_gram_freq'][:10]])
                elements.append(Paragraph(ngram_text, normal_style))
                elements.append(Paragraph(f"{n}-gram Repetition Rate: {ngram_data['repetition_rate']:.4f}", normal_style))
            else:
                 elements.append(Paragraph(f"No {n}-gram data available.", normal_style))
            elements.append(Spacer(1, 6))
    
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("Character N-gram Analysis:", h1_style))
    for n in [3, 4, 5]:
        key = f'char_{n}-grams'
        if key in analysis_results['char_ngram_analysis']:
             char_ngram_data = analysis_results['char_ngram_analysis'][key]
             elements.append(Paragraph(f"Common Character {n}-grams (Top 10):", h2_style))
             if char_ngram_data['n_gram_freq']:
                 char_ngram_text = ", ".join([f"'{ng}' ({count})" for ng, count in char_ngram_data['n_gram_freq'][:10]])
                 elements.append(Paragraph(char_ngram_text, code_style))
             else:
                 elements.append(Paragraph(f"No character {n}-gram data available.", normal_style))
             elements.append(Spacer(1, 6))
             
    try:
        doc.build(elements)
    except Exception as e:
        print(f"Error building PDF report {output_file}: {e}")

def run_full_analysis(text):
    preprocessed_data = preprocess_text(text)
    
    basic_stats = compute_basic_stats(preprocessed_data)
    word_entropy = calculate_entropy(preprocessed_data['tokens_with_stopwords'])
    char_entropy = calculate_entropy(preprocessed_data['characters'])
    readability_scores = analyze_readability(preprocessed_data['original_text'])
    frequency_distribution = analyze_frequency_distribution(preprocessed_data['tokens_with_stopwords'])
    pos_analysis = perform_pos_analysis(preprocessed_data['tokens_with_stopwords'])
    ngram_analysis = perform_ngram_analysis(preprocessed_data['tokens_with_stopwords'], n_values=[2, 3, 4])
    sentiment = perform_sentiment_analysis(preprocessed_data['original_text'])
    punctuation = analyze_punctuation(preprocessed_data['characters'])
    function_words = analyze_function_words(preprocessed_data['tokens_with_stopwords'])
    mattr = calculate_mattr(preprocessed_data['tokens_with_stopwords'])
    char_ngram_analysis = perform_character_ngram_analysis(preprocessed_data['original_text'], n_values=[3, 4, 5])

    analysis_results = {
        'preprocessed': preprocessed_data,
        'basic_stats': basic_stats,
        'word_entropy': word_entropy,
        'char_entropy': char_entropy,
        'readability': readability_scores,
        'frequency_distribution': frequency_distribution,
        'pos_analysis': pos_analysis,
        'ngram_analysis': ngram_analysis,
        'sentiment': sentiment,
        'punctuation': punctuation,
        'function_words': function_words,
        'mattr': mattr,
        'char_ngram_analysis': char_ngram_analysis
    }
    return analysis_results

def flatten_results(filename, results):
    flat = {'filename': filename}
    flat.update(results['basic_stats'])
    flat['word_entropy'] = results['word_entropy']
    flat['char_entropy'] = results['char_entropy']
    flat.update({f"readability_{k}": v for k, v in results['readability'].items()})
    flat.update({f"sentiment_{k}": v for k, v in results['sentiment'].items()})
    flat['pos_tag_count'] = len(results['pos_analysis']['pos_counts'])
    flat['function_word_ratio'] = results['function_words']['function_word_ratio']
    flat['mattr'] = results['mattr']
    for n in [2, 3, 4]:
        key = f'{n}-grams'
        if key in results['ngram_analysis']:
             flat[f'{n}gram_repetition_rate'] = results['ngram_analysis'][key]['repetition_rate']
        else:
             flat[f'{n}gram_repetition_rate'] = 0
    
    return flat

def main():
    if len(sys.argv) != 2:
        print("Usage: python advanced_text_analyzer.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    if not validate_directory(directory_path):
        sys.exit(1)
    
    output_base_dir = os.path.join(directory_path, 'analysis_output')
    os.makedirs(output_base_dir, exist_ok=True)
    
    all_results_data = []
    
    txt_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.txt')])

    for filename in txt_files:
        file_path = os.path.join(directory_path, filename)
        print(f"Processing {filename}...")
        
        text = read_text_file(file_path)
        if text is None or not text.strip():
            print(f"Skipping empty or unreadable file: {filename}")
            continue
        
        try:
            analysis_results = run_full_analysis(text)
            
            filename_base = os.path.splitext(filename)[0]
            
            individual_output_dir = os.path.join(output_base_dir, 'individual_reports')
            os.makedirs(individual_output_dir, exist_ok=True)

            plots = create_visualizations(analysis_results, individual_output_dir, filename_base)
            
            pdf_output = os.path.join(individual_output_dir, f"{filename_base}_analysis.pdf")
            generate_pdf_report(analysis_results, plots, pdf_output)
            
            flat_data = flatten_results(filename, analysis_results)
            all_results_data.append(flat_data)
            
            print(f"Analysis complete for {filename}. Report saved to {pdf_output}")
            
        except Exception as e:
            print(f"!!! Critical error processing {filename}: {e}. Skipping file.")
            import traceback
            traceback.print_exc()
            
    if not all_results_data:
        print("No files were successfully processed. No summary generated.")
        sys.exit(0)

    summary_csv_path = os.path.join(output_base_dir, 'summary_analysis_results.csv')
    print(f"\nGenerating summary CSV file: {summary_csv_path}")
    
    fieldnames = list(all_results_data[0].keys())
    
    try:
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results_data)
        print("Summary CSV generated successfully.")
    except Exception as e:
        print(f"Error writing summary CSV file: {e}")

    print("\nProcessing finished.")

if __name__ == "__main__":
    main()