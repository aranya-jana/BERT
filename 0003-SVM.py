from LoadDatasets import load_datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

# Step 1: Load data
context = load_datasets()

print("\033[95m\033[1m========== SVM ==========\033[0m")

for dataset_name, df in context.items():
    print(f"\033[93m===== Processing dataset: {dataset_name} =====\033[0m")
    
    # Step 2: Separate features and targets
    X = df["text"]
    y = df.drop(columns=["text"])  # Multi-label targets

    # Step 3: Vectorize text
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X_vectorized = vectorizer.fit_transform(X)

    # Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    # Step 5: Train model using Linear SVM
    svm = LinearSVC(max_iter=2000)
    model = MultiOutputClassifier(svm)
    model.fit(X_train, y_train)

    # Step 6: Make predictions
    y_pred = model.predict(X_test)

    # Step 7: Evaluation
    print("\033[94m=== Classification Report (per label) ===\033[0m")
    print(classification_report(y_test, y_pred, target_names=y.columns))

    # Accuracy per label
    label_accuracies = (y_pred == y_test).mean(axis=0)
    print("\033[94m\n=== Accuracy Per Label ===\033[0m")
    for label, acc in zip(y.columns, label_accuracies):
        print(f"{label}: \033[92m{acc}\033[0m")

    # Subset Accuracy (exact match)
    subset_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== Subset Accuracy (exact match of all labels): \033[92m{subset_accuracy}\033[0m")

    # Hamming Loss
    hloss = hamming_loss(y_test, y_pred)
    print(f"\033[91mHamming Loss: {hloss:.4f}\033[0m")