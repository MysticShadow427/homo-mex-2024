import pandas as pd
import numpy as np
import re

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
