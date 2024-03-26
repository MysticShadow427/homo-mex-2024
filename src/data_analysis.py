import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import defaultdict
from textwrap import wrap
import argparse
import os
from wordcloud import WordCloud

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def get_distribution(df,artipath):
    label_counts = df['label'].value_counts()

    plt.figure(figsize=(8, 6))
    label_counts.plot(kind='bar')

    plt.title('Distribution of Data Points by Class')
    plt.xlabel('Class Labels')
    plt.ylabel('Number of Data Points')

    for i, count in enumerate(label_counts):
        plt.text(i, count + 0.1, str(count), ha='center')

    plt.tight_layout()

    plt.show()

def get_word_cloud(df,artipath):
    text_combined = ' '.join(df['content'])

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def main(args):
    filename = os.path.splitext(os.path.basename(args.datapath))[0]

    try:
        globals()[filename] = pd.read_csv(args.datapath)
    except FileNotFoundError:
        print(f"File '{args.path}' not found.")
        return
    
    get_distribution(globals()[filename],args.artifactpath)
    get_word_cloud(globals()[filename],args.artifactpath)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to the CSV file of data")
    parser.add_argument("--artifactpath",type=str,help="Path to save artifacts")
    args = parser.parse_args()
    if args:
        main(args)
    else:
        parser.print_help()