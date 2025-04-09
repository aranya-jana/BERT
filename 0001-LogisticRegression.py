from LoadDatasets import load_datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

# Step 1: Load data
context = load_datasets()

for dataset_name, df in context.items():
    print(f"\033[93m===== Processing dataset: {dataset_name} =====\033[0m")
    # Step 2: Separate features and target
    X = df["text"]
    y = df.drop(columns=["text"])  # Multi-label targets

    # Step 3: Vectorize text
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X_vectorized = vectorizer.fit_transform(X)

    # Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # Step 5: Train model
    lr = LogisticRegression(max_iter=1000)
    model = MultiOutputClassifier(lr)
    model.fit(X_train, y_train)

    # Step 6: Make predictions
    y_pred = model.predict(X_test)

    # Step 7: Evaluation
    print("=== Classification Report (per label) ===")
    print(classification_report(y_test, y_pred, target_names=y.columns))

    # Accuracy per label
    label_accuracies = (y_pred == y_test).mean(axis=0)
    print("\n=== Accuracy Per Label ===")
    for label, acc in zip(y.columns, label_accuracies):
        print(f"\033[92m{label}\033[0m: \033[94m{acc:.2f}\033[0m")

    # Subset Accuracy (exact match)
    subset_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== Subset Accuracy (exact match of all labels): {subset_accuracy:.2f}")

    # Hamming Loss
    hloss = hamming_loss(y_test, y_pred)
    print(f"\033[91mHamming Loss: {hloss:.4f}\033[0m")