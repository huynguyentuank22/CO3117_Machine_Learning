# **Vietnamese Sentiment Analysis**

## 📝 **Introduction**
This project focuses on Vietnamese sentiment analysis, aiming to classify textual data into sentiment categories such as positive and negative. To achieve this, we implement and compare the performance of three distinct models:

- **TextCNN**: A convolutional neural network architecture effective for capturing local features in text sequences.

- **PhoBERT**: A state-of-the-art pre-trained language model tailored for Vietnamese, based on the RoBERTa architecture, known for its superior performance in various Vietnamese NLP tasks.

- **ViSoBERT**: A BERT-base variant further pretrained on 7 GB of Vietnamese social-media text (tweets, forums, Facebook), using whole-word and dynamic masking to model informal, noisy language—delivering strong gains on sentiment classification and slang detection.

- **Support Vector Machine (SVM)**: A traditional machine learning algorithm that excels in high-dimensional spaces, often used for text classification tasks.

- **RNN**: A vanilla recurrent neural network that processes tokens one at a time, maintaining a hidden state to capture sequential dependencies; it is simple and fast but can struggle to learn long‐range patterns due to vanishing gradients.

- **GRU**: A gated recurrent unit network that introduces update and reset gates to control information flow, offering many of LSTM’s benefits (remembering long‐term dependencies) with fewer parameters and faster training.

- **LSTM**: A long short-term memory network equipped with input, forget, and output gates, explicitly designed to retain information over long sequences and overcome the vanishing-gradient problem in RNNs.

- **Transformer**: A self-attention–based architecture built from multi-head attention and position-wise feed-forward layers, which captures global context in parallel and has become the dominant model for modern NLP tasks.

- **Naïve Bayes**: A probabilistic classifier applying Bayes’ theorem under a strong feature-independence assumption; prized for its simplicity, interpretability, and surprisingly robust baseline performance in text classification.

## 📁 **Dataset**

- **Source**: NTC-SCV is a dataset of blogs on the website https://streetcodevn.com/blog/dataset

- **Description**: The dataset consists of 50,001 user comments collected from Foody.vn, evenly split between positive and negative sentiments. Sentiment labels are assigned based on rating scores: 0.1-5.7 for negative and 8.5-10.0 for positive. Each entry includes only the raw text, making it suitable for preprocessing and training sentiment analysis models.

## 🛠️ Text Preprocessing

1. **Tokenization**: Using `pyvi` for Vietnamese word segmentation
2. **Teencode Normalization**: Maps informal Vietnamese (e.g., "ko" → "không")
3. **URL and HTML Removal**: Removes web links and HTML tags
4. **Emoji Handling**: Converts emojis to textual representations
5. **Language Detection**: Filters out non-Vietnamese content

## 🧠 **Methodology**

- **TextCNN**
    
    - **Embedding Layer**: Maps tokenized words to dense vectors (vocab size: 1000, embedding dim: 100) using an embedding matrix.
    - **Convolutional Layers**: Applies multiple 1D convolutions with kernel sizes [2, 3, 4], each with 100 filters, to capture local features.
    - **Pooling**: Uses max pooling over each convolutional output to extract the most salient features.
    - **Fully Connected Layer**: Concatenates pooled features and feeds them into a linear layer for binary classification (positive/negative).
    - **Training**: Uses Adam optimizer (lr=0.001) and CrossEntropyLoss, with tokenized sentences padded to uniform length.

- **PhoBert**

    - **Pre-trained Model**: Loads vinai/phobert-base with a tokenizer and sequence classification head (2 labels).
    - **Tokenization**: Sentences are tokenized with padding and truncation (max length: 256).
    - **Fine-tuning**: The model is trained on a custom dataset using AdamW optimizer (lr=2e-5) and a linear learning rate scheduler for 3 epochs.

- **ViSoBERT**

    - **Model & Tokenizer Initialization**  
        - Load the `uitnlp/visobert` weights via HuggingFace’s `AutoModelForSequenceClassification` (num_labels=2).  
        - Use the matching SentencePiece tokenizer for subword tokenization.  

    - **Fine-tuning Configuration**  
        - Optimizer: AdamW with `lr = 2e-5`.
        - Learning-rate scheduler: linear warm-up over the first 500 steps then linear decay. 
        - Loss: `CrossEntropyLoss` on model logits.

    - **Training & Evaluation**  
        - Shuffle training data each epoch and evaluate on a held-out validation set after each epoch.  
        - Save the checkpoint that achieves the highest validation accuracy.  

- **RNN**

    - **Data Preprocessing**  
        - Normalize text (lowercase, strip punctuation) and tokenize on whitespace.  
        - Build a vocabulary and convert each token to its integer ID.  
        - Pad or truncate all sequences to a fixed length (max_length = 300).

    - **Model Architecture**  
        - **Embedding Layer**: Maps each token ID to a 128-dimensional vector.  
        - **Recurrent Layer**: A 2-layer vanilla RNN with 256 hidden units per layer, bidirectional, and 0.4 dropout between layers.  
        - **Sequence Pooling**: Take the final hidden state from both directions and concatenate → a 512-dim vector.  
        - **Classification Head**: Linear(512 → 2) producing logits for positive/negative.

    - **Training Setup**  
        - Optimizer: AdamW, learning rate = 2×10⁻³.  
        - Loss: CrossEntropyLoss.  
        - Batch size: 64; epochs: 10; no LR scheduler.  
        - Shuffle training data each epoch and evaluate on a held-out validation split.

---

- **GRU**

    - **Data Preprocessing**  
        - Same as RNN: tokenize, pad/truncate to length 300.

    - **Model Architecture**  
        - **Embedding Layer**: 128-dim embeddings.  
        - **Recurrent Layer**: 2-layer GRU with 256 hidden units, bidirectional, dropout = 0.4.  
        - **Sequence Pooling**: Concatenate final forward and backward hidden states (512-dim).  
        - **Classification Head**: Linear(512 → 2).

    - **Training Setup**  
        - AdamW optimizer, lr = 2e-3; CrossEntropyLoss; batch_size = 64; epochs = 10; no scheduler.  
        - Monitor validation accuracy and pick the best checkpoint.

---

- **LSTM**

    - **Data Preprocessing**  
        - Identical to the RNN/GRU pipelines.

    - **Model Architecture**  
        - **Embedding Layer**: 128-demb.  
        - **Recurrent Layer**: 2-layer LSTM with 256 hidden units in each direction, dropout = 0.4 on outputs.  
        - **Sequence Pooling**: Concatenate the last hidden states of both directions (512-d).  
        - **Classification Head**: Linear(512 → 2).

    - **Training Setup**  
        - Use AdamW (lr=2e-3), CrossEntropyLoss, batch_size=64, epochs=10, no LR scheduler.  
        - Early-stop if validation loss plateaus for 3 epochs.

---

- **Transformer (trained from scratch)**

    - **Data Preprocessing**  
        - Tokenize with a learned Subword vocabulary and pad/truncate to max_length = 300.  
        - Add special `[CLS]` tokens as needed.

    - **Model Architecture**  
        - **Embedding & Positional Encoding**: Map tokens into 128-d vectors and add learned/sinusoidal positional embeddings.  
        - **Encoder Stack**: 6 Transformer encoder layers, each with  
            - 8 attention heads over 128-d model,  
            - a 256-unit feed-forward sublayer (with ReLU),  
            - dropout = 0.1 on attention weights and FF outputs.  
        - **Pooling**: Take the `[CLS]` token’s final embedding as sequence representation.  
        - **Classification Head**: Linear(128 → 1).

    - **Training Setup**  
        - Optimizer: AdamW, lr = 2×10^{-5}; no LR scheduler.  
        - Loss: CrossEntropyLoss; batch_size = 64; epochs = 30.  
        - Select the model checkpoint achieving highest validation accuracy.  


- **SVM**

    - **Feature Extraction**: Sentences are tokenized using ViTokenizer and converted to TF-IDF vectors (max features: 500) to represent word importance.
    - **Training**: An SVM with a linear kernel (C=1) is trained on the TF-IDF features using scikit-learn. The dataset is split into 80% training and 20% testing.

## 📊 **Result**

| Models | Accuracy |
|--------|----------|
| Transformer (PhoBERT pretrained) | 0.9189 |
| Transformer (VisoBERT pretrained) | 0.9111 |
| GRU | 0.898 |
| TextCNN |  0.8814 | 
| LSTM | 0.8778 |
| Transformer | 0.8739 |
| SVM | 0.8698 | 
| RNN | 0.7134 | 

## 🚀 **Usage**

1. Clone the repository:

        git clone https://github.com/huynguyentuank22/CO3117_Machine_Learning.git
        cd CO3117_Machine_Learning.git

2. Environment setup:

        # Create a virtual environment
        python -m venv myenv

        # Activate the environment
        # On Windows
        myenv\Scripts\activate
        # On macOS/Linux
        source myenv/bin/activate

        # Install dependencies
        pip install -r requirements.txt

3. How to train and evaluate the model?
    - For Naive Bayes, SVM, PhoBERT, VisoBERT: run the corresponding notebook
    - For RNN, GRU, LSTM, Transformer trained from scratch: run the corresponding scripts int `scripts/`
        - You can train the hyperparameters in `config/` folder
        - To evaluate, change the path to checkpoint and choose the correponding model in `evaluate.py` before running `evaluate.sh`

## 🗂️ **Project Structure**

```
CO3117_Machine_Learning/
├── data/                     # Dataset storage
├── cleaned_data/             # Cleaned dataset storage
├── config                    # Contain configurations for trainer, models, optimizers, data
├── src
    ├── models                # Contain RNN, LSTM, GRU models
    ├── data.py
    ├── evaluate.py
    ├── main.py
    ├── train.py
    ├── utils.py
├── scripts                   # Contain training and evaluating scripts
├── modelTextCNN/             # Saved TextCNN model
├── modelTransformer/         # Saved PhoBert model
├── preprocessing.ipynb       # Jupyter notebooks for text preprocessing
├── textCNN.ipynb             # Jupyter notebooks for training and evaluate TextCNN model
├── SVM.ipynb                 # Jupyter notebooks for training and evaluate SVM model
├── transformer.ipynb         # Jupyter notebooks for training and evaluate PhoBert model
├── transformer_visobert.ipynb         # Jupyter notebooks for training and evaluate PhoBert model
├── .gitignore
├── README.md
└── requirements.txt          # Python dependencies
```

## 📚 References

[1] Zhang, Y., & Wallace, B. (2015). A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification. arXiv preprint arXiv:1510.03820.

[2] Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. arXiv preprint arXiv:2003.00744.