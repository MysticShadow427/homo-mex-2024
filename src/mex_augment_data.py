# so we have very imbalanced data so we need to augment and best i can see is back - transalation technique 
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.under_sampling import OneSidedSelection
from sklearn.preprocessing import LabelEncoder

def random_oversample(df,random_seed = 42,desired_samples = 5400):
    """First technique to try is random oversampling"""
    np.random.seed(random_seed)
    class_1_df = df[df['label'] == 'NP']
    class_2_df = df[df['label'] == 'NR']
    class_3_df = df[df['label'] == 'P']

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

def random_oversample_track_3(df,random_seed = 42):
    """First technique to try is random oversampling"""
    np.random.seed(random_seed)
    class_1_df = df[df['label'] == 'NP']
    class_3_df = df[df['label'] == 'P']

    desired_samples = 560 

    if len(class_3_df) < desired_samples:
        oversampled_class_3_df = class_3_df.sample(n=desired_samples, replace=True)
    else:
        oversampled_class_3_df = class_3_df.sample(frac=1.0)  

    augmented_df = pd.concat([class_1_df, oversampled_class_3_df])

    augmented_df = augmented_df.sample(frac=1.0).reset_index(drop=True)

    return augmented_df


def noise_augment_embeddings():
    """Add noise to the embeddings to create new ones"""
    pass

def smote_augment_embeddings(df):
    """Use SMOTE on the embeddings"""
    X, y = df.drop('label',axis = 1).values,df['label'].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    oversample = SMOTE(random_state = 42)
    X, y = oversample.fit_resample(X, y)
    augmented_df = pd.DataFrame(X, columns=[f"feat_{i+1}" for i in range(768)])
    augmented_df['label'] = le.inverse_transform(y)
    return augmented_df
    
def augment_data_with_adasyn(df):
    """Use ADASYN on the embeddings"""
    X, y = df.drop('label',axis = 1).values,df['label'].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    oversample = ADASYN(random_state = 42)
    X, y = oversample.fit_resample(X, y)
    augmented_df = pd.DataFrame(X, columns=[f"feat_{i+1}" for i in range(768)])
    augmented_df['label'] = le.inverse_transform(y)
    return augmented_df

def augment_data_with_oss(df):
    X, y = df.drop('label',axis = 1).values,df['label'].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    undersample = OneSidedSelection(random_state = 42)
    X, y = undersample.fit_resample(X, y)
    augmented_df = pd.DataFrame(X, columns=[f"feat_{i+1}" for i in range(768)])
    augmented_df['label'] = le.inverse_transform(y)
    return augmented_df





