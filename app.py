# app.py
from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load saved model and TF-IDF vectorizer
model = joblib.load("aiguardian_clf.pkl")
vectorizer = joblib.load("aiguardian_vectorizer.pkl")

@app.route('/')
def home():
    return "AI Guardian Text Classifier API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        # Transform text using TF-IDF vectorizer
        text_vector = vectorizer.transform([text])
        
        # Predict class
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector).max()  # highest probability
        
        # Return result
        return jsonify({
            'text': text,
            'prediction': prediction,
            'confidence': round(float(probability), 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
