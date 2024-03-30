import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer

checkpoint = 'dccuchile/bert-base-spanish-wwm-uncased'

# Function to remove pattern occurrences of '@Username' from a sentence
def remove_pattern(sentence):
    return re.sub(r'@[\w]+', '', sentence)

# Function to remove numbers and URLs from a sentence
def remove_numbers_and_urls(sentence):
    # Remove numbers
    sentence = re.sub(r'\d+', '', sentence)
    # Remove URLs
    sentence = re.sub(r'http[s]?://\S+', '', sentence)
    return sentence

# Function to remove characters except specified punctuation marks and spaces
def remove_chars_except_punctuations(sentence):
    # Keep words, '?', '!', ',', '.', and spaces
    sentence = re.sub(r'[^\w\s?!,.]', '', sentence)
    return sentence

def remove_newline_pattern(sentence):
    # Remove '\n' pattern
    sentence = re.sub(r'\n', '', sentence)
    return sentence

def get_sentence_embeddings():
    model = SentenceTransformer(checkpoint).to('cuda')
    df = pd.read_csv('/content/homo-mex-2024/data/public_data_dev_phase/track_1_dev.csv')
    df['content'] = df['content'].apply(remove_pattern)
    df['content'] = df['content'].apply(remove_numbers_and_urls)
    df['content'] = df['content'].apply(remove_chars_except_punctuations)
    df['content'] = df['content'].apply(remove_newline_pattern)
    sentences = []
    for text in df['content']:
        sentences.append(text)
    embeddings = model.encode(sentences)
    embeddings_df = pd.DataFrame(embeddings,columns = [f"feat_{i+1}" for i in range(embeddings.shape[1])])
    embeddings_df['label'] = df['label']
    return embeddings_df
