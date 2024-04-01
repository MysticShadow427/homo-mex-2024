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
    - `mex_train_lstm.py` is used to train a lstm on top of the Spanish BERT extracted features.

</hr>

Sure, here are the instructions formatted for better readability:

**Instructions to run the Python Scripts**

- **`mex_main.py`**

    This script requires three arguments to be passed:

    1. `epochs`
    2. `learning rate`
    3. `batch size`

- **`train_lora_llm.py`**

    This script can be run directly without any arguments. However, if you wish to change some parameters, you can edit them on the machine where you are hosting the script.

- **`mex_train_xgboost.py`**

    This script requires three arguments:

    1. `tune`: A boolean value. If passed as `True`, XGBoost will undergo hyperparameter tuning to find the best parameters before training. This process takes around 20-25 minutes.
    2. `embeddings`: A string argument with two possible values:
        - `"jina"`: Use Jina sentence embeddings.
        - `"spanish-bert"`: Use Spanish BERT sentence embeddings.
    3. `augmentation`: Another string argument with four possible values:
        - `"smote"`: Apply SMOTE augmentation on your sentence embeddings.
        - `"adasyn"`: Apply ADASYN augmentation on your sentence embeddings.
        - `"random"`: Apply random oversampling augmentation on your sentence embeddings.
        - `"oss"`: Apply OSS undersampling augmentation on your sentence embeddings.

- **`mex_train_lstm.py`**

    This script requires seven arguments:

    1. `epochs`: A integer value for number of epochs of training.
    2. `learning_rate`: A float value for learning rate of the optimizer
    3. `batch_size`: A integer value specifying the batch size.
    4. `dropout`: A float value for dropout prob between 2 layers.
    5. `num_layers`:A integer value for setting the number of LSTM layers.
    6. `bidirectional` : A integer value for setting the LSTM to be bidirectional or not , if **0** then no BiLSTM.
    7. `hidden_size` : A integer value for dimensionality of the LSTM hidden space.
<hr>

*I think the best way to run the scripts is on Google Colab,host your data their and just run the `homo-mex-2024-main.ipynb`.*
<hr>
<details>
  <summary>Next approaches planned</summary>
  - Using features from Spanish BERT and instead of directly applying a softmax layer we can train a another LSTM or Transformer.
  - Use contrastive losses to enhance the feature representations and then using some classifiers on top of it.
</details>