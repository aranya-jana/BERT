from LoadDatasets import load_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

merged_df = load_datasets()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    merged_df['text'], merged_df['label'], test_size=0.2, random_state=42
)

# vertorized
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# CNN Model
# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Label encoding (in case labels aren't 0/1)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# CNN architecture
cnn_model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_len),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
cnn_model.fit(X_train_pad, y_train_enc, epochs=5, batch_size=64, verbose=1, validation_split=0.1)

# Evaluate
loss, accuracy_cnn = cnn_model.evaluate(X_test_pad, y_test_enc, verbose=0)
print("CNN Accuracy : {:.2f}".format(accuracy_cnn))