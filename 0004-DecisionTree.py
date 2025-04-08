from LoadDatasets import load_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

merged_df = load_datasets()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    merged_df['text'], merged_df['label'], test_size=0.2, random_state=42
)

# vertorized
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_tfidf, y_train)

# Accuracy
y_pred_dt = dt_model.predict(X_test_tfidf)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy : {:.2f}".format(accuracy_dt))