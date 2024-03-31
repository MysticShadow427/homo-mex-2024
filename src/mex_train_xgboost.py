# here i will take out the sentence embeddings and train xgboost on this as we also have a variety of augmentation techniques for the numerical data
# give the data path as drive 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
import optuna
from mex_eval import get_classification_report,get_confusion_matrix,get_scores
from utils import save_xgb
import argparse
from mex_augment_data import random_oversample,smote_augment_embeddings,augment_data_with_adasyn,augment_data_with_oss,noise_augment_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", type=bool, help="hyperparameter tune or not")
    parser.add_argument("--embeddings", type=str, help="which embeddings to train xgboost")
    parser.add_argument("--augmentation", type=str, help="which augmentation technique to use")
    args = parser.parse_args()

    tune = args.tune
    embeddings = args.embeddings
    aug = args.augmentation

    if embeddings == 'jina':
        df = pd.read_csv('/content/drive/MyDrive/homo-mex-2024-jina-embeddings.csv')
     # change the path according to which augmented version you are taking as all augmented df are hosted on drive 
    else:
        df = pd.read_csv('/content/drive/MyDrive/homo-mex-2024-spanish-bert-embeddings.csv')
    
    if aug =='random':
        df = random_oversample(df)
    elif aug =='smote':
        df = smote_augment_embeddings(df)
    elif aug =='adasyn':
        df = augment_data_with_adasyn(df)
    else:
        df = augment_data_with_oss(df)

    # le = LabelEncoder()
    # df['label'] = le.fit_transform(df['label'])
    labels_df = pd.DataFrame(df['label'].values,columns = ['label'])
    df = df.drop('label',axis = 1)
    class_names = ['NP','NR','P']

    X_train,X_test,y_train,y_test = train_test_split(df,labels_df,test_size = 0.20,random_state = 42,stratify=labels_df)
    
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced',y=y_train['label'])

    print('\033[96m' + X_train.shape,X_test.shape+ '\033[0m')

    if tune:
        print('\033[96m' + 'Hyperparameter Tuning...'+ '\033[0m')
        def objective(trial):
            """Define the objective function"""

            params = {
                'max_depth': trial.suggest_int('max_depth', 1, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0,log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0,log=True),
                'subsample': trial.suggest_float('subsample', 0.01, 1.0,log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0,log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0,log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0,log=True),
                'eval_metric': 'mlogloss',
                'use_label_encoder': False
            }

            optuna_model = XGBClassifier(objective = 'multi:softmax',**params)
            optuna_model.fit(X_train,y_train)

            y_pred = optuna_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            return accuracy
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)
        print('\033[96m' + 'Number of finished trials: {}'.format(len(study.trials))+ '\033[0m')
        print('\033[96m' + 'Best trial:'+ '\033[0m')
        trial = study.best_trial

        print('  Value: {}'.format(trial.value))
        print('  Params: ')

        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))
        
        params = trial.params
        model = XGBClassifier(objective = 'multi:softmax',**params)
        print('\033[96m' + 'Starting model training...'+ '\033[0m')
        print()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print('\033[96m' + 'Predictions :'+ '\033[0m')
        get_classification_report(y_test,y_pred)
        get_confusion_matrix(y_test,y_pred)
        get_scores(y_test,y_pred)
        save_xgb(model)
    else:
        xgb = XGBClassifier(objective = 'multi:softmax',eval_metric = 'mlogloss',use_label_encoder=False)
        print('\033[96m' + 'Starting model training...'+ '\033[0m')
        print()
        xgb.fit(X_train,y_train)
        y_pred = xgb.predict(X_test)
        print('\033[96m' + 'Predictions :'+ '\033[0m')
        get_classification_report(y_test,y_pred)
        get_confusion_matrix(y_test,y_pred)
        get_scores(y_test,y_pred)
        save_xgb(xgb)
    
    




