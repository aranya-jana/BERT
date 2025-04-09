from LoadDatasets import load_datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model

# Step 1: Load data
context = load_datasets()

print("\033[95m\033[1m========== CNN ==========\033[0m")

for dataset_name, df in context.items():
    print(f"\033[93m===== Processing dataset: {dataset_name} =====\033[0m")
    
    X = df["text"]
    y = df.drop(columns=["text"])

    # Text tokenization
    max_words = 10000
    max_len = 100

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post', truncating='post')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_pad, y.values, test_size=0.2, random_state=42)

    # CNN model
    inputs = Input(shape=(max_len,))
    x = Embedding(input_dim=max_words, output_dim=128, input_length=max_len)(inputs)
    x = Conv1D(128, kernel_size=5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(y.shape[1], activation='sigmoid')(x)  # Sigmoid for multi-label

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

    # Predict
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Evaluate
    print("\033[94m=== Classification Report (per label) ===\033[0m")
    print(classification_report(y_test, y_pred, target_names=y.columns))

    label_accuracies = (y_pred == y_test).mean(axis=0)
    print("\033[94m\n=== Accuracy Per Label ===\033[0m")
    for label, acc in zip(y.columns, label_accuracies):
        print(f"{label}: \033[92m{acc:.2f}\033[0m")

    subset_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== Subset Accuracy (exact match of all labels): \033[92m{subset_accuracy:.2f}\033[0m")

    hloss = hamming_loss(y_test, y_pred)
    print(f"\033[91mHamming Loss: {hloss:.4f}\033[0m")