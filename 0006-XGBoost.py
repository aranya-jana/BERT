from LoadDatasets import load_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

merged_df = load_datasets()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    merged_df['text'], merged_df['label'], test_size=0.2, random_state=42
)

# vertorized
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# XGBoost Model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_tfidf, y_train)

# Accuracy
y_pred_xgb = xgb_model.predict(X_test_tfidf)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy : {:.2f}".format(accuracy_xgb))
