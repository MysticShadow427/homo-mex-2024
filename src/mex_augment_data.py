# so we have very imbalanced data so we need to augment and best i can see is back - transalation technique 
import pandas as pd
import numpy as np

def random_oversample(df):
    """First technique to try is random oversampling"""

    class_1_df = df[df['label'] == 'NP']
    class_2_df = df[df['label'] == 'NR']
    class_3_df = df[df['label'] == 'P']

    desired_samples = 4000

    if len(class_2_df) < desired_samples:
        oversampled_class_2_df = class_2_df.sample(n=desired_samples, replace=True)
    else:
        oversampled_class_2_df = class_2_df.sample(frac=1.0)  

    if len(class_3_df) < desired_samples:
        oversampled_class_3_df = class_3_df.sample(n=desired_samples, replace=True)
    else:
        oversampled_class_3_df = class_3_df.sample(frac=1.0)  

    augmented_df = pd.concat([class_1_df, oversampled_class_2_df, oversampled_class_3_df])

    augmented_df = augmented_df.sample(frac=1.0).reset_index(drop=True)

    return augmented_df

def noise_augment_embeddings():
    """Add noise to the embeddings to create new ones"""
    pass

def smote_augment_embeddings():
    """Use SMOTE on the embeddings"""
    pass

def cbos_augment_embeddings():
    """Use CBOS(Cluster-based oversampling (CBOS)) on the embeddings"""
    pass

def we_augment_embeddings():
    """Use WE(Wilsons editing) on the embeddings"""
    pass

def oss_augment_embeddings():
    """Use OSS(One-sides Sections) on the embeddings"""
    


