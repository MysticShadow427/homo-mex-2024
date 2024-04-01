# homo-mex-2024
**You can run the code either on Local Machine with a GPU or *Google Colab/Kaggle Environment* by cloning the repo and running the commands on the terminal or best way is to follow the commands given in `mex_main.py` and `mex_train_xgboost.py.`**

<hr>

- `notebooks/` have the messy code for reference 
- `artifacts/` have some images and plots for the EDA.
- `data/` has following two subfolders :
    - `public_data_dev_phase/` contains the original competition data (train + dev in a single csv file with labels).
    - `sentence_embeddings/` is not hosted on github as their is file size limit, you can host it on your Drive and change the path accordingly.
- `src/` is the main folder having the python scripts to run.
    - `mex_main.py` is the main file to run for *LoRA fine tuning* or *Full fine tuning* the *Spanish BERT*.
    - `load_lora_llm.py` and its `MexSpanClassifierLoRA` class is having bugs as during training with this class loss remains same but accuracy changes , need to fix it hence avoid using this for LoRA fine tuning.
    - For LoRA fine tuning run the `train_lora_llm.py` and the model training will start with no bugs.
    - `mex_train_xgboost.py` is the file you need to run to train a *XgBoost* model on the sentence embeddings of the **Mex Span Dataset**.
    - *Sentence Embeddings* were taken as we need to augment the data and techniques like *SMOTE* and  *ADASYN* work only on numerical data.*Back Translation* technique (which is more robust) was also tried on the original dataset but it is giving OOM errors hence need to fix it.

</hr>

**Instructions to run the Python Scripts** - 
    - `mex_main.py` has 3 arguments to pass on - 
        1. epochs
        2. learning rate 
        3. batch size
    - `train_lora_llm.py` can be run directly without any argument but if you want to change some parameters you can edit on the machine you are hosting.
    - `mex_train_xgboost.py` has also 3 arguments - 
        1. tune -  a boolean if passed then the Xgboost will undergo hyperparameter tuning for finding the best parameters before training , this takes around 20-25 mins.
        2. embeddings - this is a 'str' type argument which can take 2 values - 
            1. "jina" - for using *jina-sentence-embeddings*
            2. "spanish-bert" - to use *spanish-bert-sentence-embeddings*
        3. augmentation - this is also a 'str' type which can take 4 values - 
            1. "smote" - for smote augmentation on your sentence embeddings.
            2. "adasyn" - for adasyn augmentation on your sentence embeddings.
            3. "random" - for random oversampling augmentation on your sentence embeddings.
            4. "oss" - for oss undersampling augmentation on your sentence embeddings.

*I think the best way to run the scripts is on Google Colab,host your data their and just run the `homo-mex-2024-main.ipynb`.*

<details>
  <summary>Next approaches planned</summary>
  - Using features from Spanish BERT and instead of directly applying a softmax layer we can train a another LSTM or Transformer.
  - Use contrastive losses to enhance the feature representations and then using some classifiers on top of it.
</details>