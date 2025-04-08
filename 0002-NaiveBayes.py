from LoadDatasets import load_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

merged_df = load_datasets()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    merged_df['text'], merged_df['label'], test_size=0.2, random_state=42
)

# vertorized
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Naïve Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Accuracy
y_pred_nb = nb_model.predict(X_test_tfidf)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naïve Bayes Accuracy : {:.2f}".format(accuracy_nb))