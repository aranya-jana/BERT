from LoadDatasets import load_datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

merged_df = load_datasets()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    merged_df['text'], merged_df['label'], test_size=0.2, random_state=42
)

# vertorized
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Random Forest Model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_tfidf, y_train)

# Accuracy
y_pred_rf = rf_model.predict(X_test_tfidf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy : {:.2f}".format(accuracy_rf))
