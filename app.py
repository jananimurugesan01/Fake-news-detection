from flask import Flask, request, render_template
import pickle

# Load the trained model and the vectorizer from the saved pickle files
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize the Flask app
app = Flask(__name__)

# Route to display the home page (input form)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and return the prediction result
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input news text from the form
    news = request.form['news']

    # Transform the news text using the loaded vectorizer
    news_tfidf = vectorizer.transform([news])

    # Use the model to make a prediction
    prediction = model.predict(news_tfidf)

    # Determine whether the news is "Fake" or "Real"
    result = "Real" if prediction[0] == 1 else "Fake"

    # Render the result page and pass the prediction result to it
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)


import pickle
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming you have your trained model and vectorizer
model = PassiveAggressiveClassifier(max_iter=50)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# After training your model, you can save it as follows:
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)


