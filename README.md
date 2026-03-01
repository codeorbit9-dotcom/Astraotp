
# AI Guardian Text Classifier

A simple text classification system to detect **scam vs ham messages** using **TF-IDF vectorization** and **Logistic Regression**. Includes a **Flask API** for predictions.

---

## **1. Project Structure**
AI-Guardian/ ├── app.py                       # Flask API for predictions ├── aiguardian_clf.pkl           # Trained Logistic Regression model ├── aiguardian_vectorizer.pkl    # TF-IDF vectorizer ├── aiguardian_dataset.csv       # Dataset with 'message' and 'label' columns ├── README.md
Copy code

---

## **2. Requirements**

Install Python dependencies:

```bash
pip install pandas scikit-learn joblib flask
3. Dataset Format
Your dataset CSV should have two columns:
message
label
"This is a safe message"
ham
"You won a free gift!"
scam
4. Training the Model
Python
Copy code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("aiguardian_dataset.csv")
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression
clf = LogisticRegression(max_iter=500, n_jobs=-1)
clf.fit(X_train_tfidf, y_train)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report
y_pred = clf.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(clf, "aiguardian_clf.pkl")
joblib.dump(vectorizer, "aiguardian_vectorizer.pkl")
5. Running the Flask App
Place app.py, aiguardian_clf.pkl, and aiguardian_vectorizer.pkl in the same folder.
Run the app:
Bash
Copy code
python app.py
API endpoints:
Home: GET / → Check if API is running.
Predict: POST /predict → Predict message class.
Request Example:
Bash
Copy code
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"text":"You won a free gift!"}'
Response Example:
JSON
Copy code
{
  "text": "You won a free gift!",
  "prediction": "scam",
  "confidence": 0.9325
}
6. Notes & Tips
Model: Logistic Regression is lightweight and works well for CPU training.
TF-IDF: Adjust max_features or ngram_range for larger datasets.
Scaling: For 600k+ messages, consider using LightGBM for faster CPU training.
Deployment: You can wrap this API in a web dashboard, mobile app, or integrate into other services.
7. License
MIT License
