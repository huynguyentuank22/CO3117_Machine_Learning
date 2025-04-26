# **Vietnamese Sentiment Analysis**

## 📝 **Introduction**
This project focuses on Vietnamese sentiment analysis, aiming to classify textual data into sentiment categories such as positive and negative. To achieve this, we implement and compare the performance of three distinct models:

- **TextCNN**: A convolutional neural network architecture effective for capturing local features in text sequences.

- **PhoBERT**: A state-of-the-art pre-trained language model tailored for Vietnamese, based on the RoBERTa architecture, known for its superior performance in various Vietnamese NLP tasks.

- **Support Vector Machine (SVM)**: A traditional machine learning algorithm that excels in high-dimensional spaces, often used for text classification tasks.

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

- **SVM**

    - **Feature Extraction**: Sentences are tokenized using ViTokenizer and converted to TF-IDF vectors (max features: 500) to represent word importance.
    - **Training**: An SVM with a linear kernel (C=1) is trained on the TF-IDF features using scikit-learn. The dataset is split into 80% training and 20% testing.

## 📊 **Result**

| Models | Accuracy |
|--------|----------|
| PhoBERT | 0.9189   |
| TextCNN |  0.8814 | 
| SVM | 0.8698 | 

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

## 🗂️ **Project Structure**

```
CO3117_Machine_Learning/
├── data/                     # Dataset storage
├── cleaned_data/             # Cleaned dataset storage
├── modelTextCNN/             # Saved TextCNN model
├── modelTransformer/         # Saved PhoBert model
├── preprocessing.ipynb       # Jupyter notebooks for text preprocessing
├── textCNN.ipynb             # Jupyter notebooks for training and evaluate TextCNN model
├── SVM.ipynb                 # Jupyter notebooks for training and evaluate SVM model
├── transformer.ipynb         # Jupyter notebooks for training and evaluate PhoBert model
├── .gitignore
├── README.md
└── requirements.txt          # Python dependencies
```

## 📚 References

[1] Zhang, Y., & Wallace, B. (2015). A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification. arXiv preprint arXiv:1510.03820.

[2] Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. arXiv preprint arXiv:2003.00744.